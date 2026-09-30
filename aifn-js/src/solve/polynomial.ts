/**
 * Polynomial roots as the eigenvalues of the companion matrix, as numpy's `roots` computes them. `aifn/linalg` has only
 * a symmetric eigensolver, so the general (Hessenberg) eigenvalue problem is solved here by balancing and the shifted
 * QR algorithm (EISPACK's `balanc` and `hqr`, in the form of Press et al., 2007, "Numerical Recipes", 3rd ed., §11.6–
 * 11.7). The companion matrix is already upper Hessenberg, and balancing keeps it so.
 */

import { fromData, toFlat, type Tensor } from 'aifn/tensor'
import { toF64, type VectorLike } from './vector'

/** The roots of a polynomial: real and imaginary parts, conjugate pairs adjacent. */
export type PolynomialRoots = {
  /** Real parts (length = degree after stripping leading zeros). */
  real: Tensor
  /** Imaginary parts. */
  imag: Tensor
  /** False when the QR iteration did not converge within its iteration budget (the roots are then unreliable). */
  converged: boolean
}

/** Balance a (1-based, n×n) matrix in place with powers of 2 (Parlett & Reinsch, 1969). */
function balance(a: number[][], n: number): void {
  const RADIX = 2
  const sqrdx = RADIX * RADIX
  let done = false
  while (!done) {
    done = true
    for (let i = 1; i <= n; i++) {
      let r = 0
      let c = 0
      for (let j = 1; j <= n; j++)
        if (j !== i) {
          c += Math.abs(a[j][i])
          r += Math.abs(a[i][j])
        }
      if (c !== 0 && r !== 0) {
        let g = r / RADIX
        let f = 1
        const s = c + r
        while (c < g) {
          f *= RADIX
          c *= sqrdx
        }
        g = r * RADIX
        while (c > g) {
          f /= RADIX
          c /= sqrdx
        }
        if ((c + r) / f < 0.95 * s) {
          done = false
          g = 1 / f
          for (let j = 1; j <= n; j++) a[i][j] *= g
          for (let j = 1; j <= n; j++) a[j][i] *= f
        }
      }
    }
  }
}

/**
 * Eigenvalues of an upper Hessenberg matrix (1-based, destroyed) by the shifted QR algorithm with Francis double
 * shifts and exceptional shifts at iterations 10 and 20 (EISPACK `hqr`). Returns false if an eigenvalue needed more
 * than 60 iterations.
 */
function hqr(a: number[][], n: number, wr: number[], wi: number[]): boolean {
  let anorm = 0
  for (let i = 1; i <= n; i++) for (let j = Math.max(i - 1, 1); j <= n; j++) anorm += Math.abs(a[i][j])
  let nn = n
  let t = 0
  let p = 0
  let q = 0
  let r = 0
  let s = 0
  let w = 0
  let x = 0
  let y = 0
  let z = 0
  while (nn >= 1) {
    let its = 0
    let l: number
    do {
      for (l = nn; l >= 2; l--) {
        s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l])
        if (s === 0) s = anorm
        if (Math.abs(a[l][l - 1]) + s === s) {
          a[l][l - 1] = 0
          break
        }
      }
      x = a[nn][nn]
      if (l === nn) {
        // One root found.
        wr[nn] = x + t
        wi[nn] = 0
        nn--
      } else {
        y = a[nn - 1][nn - 1]
        w = a[nn][nn - 1] * a[nn - 1][nn]
        if (l === nn - 1) {
          // Two roots found: the eigenvalues of the trailing 2×2 block.
          p = 0.5 * (y - x)
          q = p * p + w
          z = Math.sqrt(Math.abs(q))
          x += t
          if (q >= 0) {
            z = p + (p >= 0 ? Math.abs(z) : -Math.abs(z))
            wr[nn - 1] = wr[nn] = x + z
            if (z) wr[nn] = x - w / z
            wi[nn - 1] = wi[nn] = 0
          } else {
            wr[nn - 1] = wr[nn] = x + p
            wi[nn - 1] = -z
            wi[nn] = z
          }
          nn -= 2
        } else {
          if (its === 60) return false
          if (its === 10 || its === 20) {
            // Exceptional shift.
            t += x
            for (let i = 1; i <= nn; i++) a[i][i] -= x
            s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2])
            y = x = 0.75 * s
            w = -0.4375 * s * s
          }
          ++its
          let m: number
          for (m = nn - 2; m >= l; m--) {
            z = a[m][m]
            r = x - z
            s = y - z
            p = (r * s - w) / a[m + 1][m] + a[m][m + 1]
            q = a[m + 1][m + 1] - z - r - s
            r = a[m + 2][m + 1]
            s = Math.abs(p) + Math.abs(q) + Math.abs(r)
            p /= s
            q /= s
            r /= s
            if (m === l) break
            const u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r))
            const v = Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z) + Math.abs(a[m + 1][m + 1]))
            if (u + v === v) break
          }
          for (let i = m + 2; i <= nn; i++) {
            a[i][i - 2] = 0
            if (i !== m + 2) a[i][i - 3] = 0
          }
          // Double QR step on rows l..nn and columns m..nn.
          for (let k = m; k <= nn - 1; k++) {
            if (k !== m) {
              p = a[k][k - 1]
              q = a[k + 1][k - 1]
              r = 0
              if (k !== nn - 1) r = a[k + 2][k - 1]
              if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0) {
                p /= x
                q /= x
                r /= x
              }
            }
            const root = Math.sqrt(p * p + q * q + r * r)
            if ((s = p >= 0 ? root : -root) !== 0) {
              if (k === m) {
                if (l !== m) a[k][k - 1] = -a[k][k - 1]
              } else a[k][k - 1] = -s * x
              p += s
              x = p / s
              y = q / s
              z = r / s
              q /= p
              r /= p
              for (let j = k; j <= nn; j++) {
                p = a[k][j] + q * a[k + 1][j]
                if (k !== nn - 1) {
                  p += r * a[k + 2][j]
                  a[k + 2][j] -= p * z
                }
                a[k + 1][j] -= p * y
                a[k][j] -= p * x
              }
              const mmin = nn < k + 3 ? nn : k + 3
              for (let i = l; i <= mmin; i++) {
                p = x * a[i][k] + y * a[i][k + 1]
                if (k !== nn - 1) {
                  p += z * a[i][k + 2]
                  a[i][k + 2] -= p * r
                }
                a[i][k + 1] -= p * q
                a[i][k] -= p
              }
            }
          }
        }
      }
    } while (l < nn - 1)
  }
  return true
}

/**
 * The roots of p(x) = c₀xⁿ + c₁xⁿ⁻¹ + … + cₙ, coefficients highest degree first (numpy's convention), as the
 * eigenvalues of the companion matrix. Leading zeros are stripped; trailing zeros give roots at 0. Roots are sorted by
 * real part, then imaginary part, both descending. Accuracy degrades for clustered or multiple roots (a double root is
 * found to about √ε).
 */
export function polynomialRoots(coefficients: VectorLike): PolynomialRoots {
  const c = Array.from(toF64(coefficients, 'polynomialRoots'))
  if (!c.every(Number.isFinite)) throw new Error('polynomialRoots: coefficients must be finite')
  while (c.length && c[0] === 0) c.shift()
  let zeros = 0
  while (c.length && c[c.length - 1] === 0) {
    c.pop()
    zeros++
  }
  const n = Math.max(c.length - 1, 0)
  const wr = new Array<number>(n + 1).fill(0)
  const wi = new Array<number>(n + 1).fill(0)
  let converged = true
  if (n > 0) {
    // Companion matrix (1-based): first row −c_k/c₀, ones on the subdiagonal.
    const a = Array.from({ length: n + 1 }, () => new Array<number>(n + 1).fill(0))
    for (let j = 1; j <= n; j++) a[1][j] = -c[j] / c[0]
    for (let i = 2; i <= n; i++) a[i][i - 1] = 1
    balance(a, n)
    converged = hqr(a, n, wr, wi)
  }
  const roots = Array.from({ length: n }, (_, k) => [wr[k + 1], wi[k + 1]])
  for (let k = 0; k < zeros; k++) roots.push([0, 0])
  roots.sort((u, v) => v[0] - u[0] || v[1] - u[1])
  return {
    real: fromData(
      Float64Array.from(roots, (r) => r[0]),
      [roots.length],
    ),
    imag: fromData(
      Float64Array.from(roots, (r) => r[1]),
      [roots.length],
    ),
    converged,
  }
}

/**
 * p(x) by Horner's rule, coefficients highest degree first. A number gives a number; a tensor of points gives a tensor
 * of the same shape.
 */
export function polynomialValue<X extends number | Tensor>(coefficients: VectorLike, x: X): X {
  const c = toF64(coefficients, 'polynomialValue')
  const horner = (at: number) => {
    let v = 0
    for (let k = 0; k < c.length; k++) v = v * at + c[k]
    return v
  }
  if (typeof x === 'number') return horner(x) as X
  const t = x as Tensor
  return fromData(Float64Array.from(toFlat(t), horner), t.shape) as X
}
