/**
 * Thin singular value decomposition by one-sided Jacobi (Hestenes, 1958; Demmel and Veselić, 1992, "Jacobi's method
 * is more accurate than QR", SIAM J. Matrix Anal. Appl. 13(4)): plane rotations orthogonalise the columns of A in
 * place; their norms are then the singular values. Small singular values come out with high relative accuracy.
 * `pinv`, `lstsq` and `conditionNumber` are built on it.
 */

import { reshape, type Tensor } from 'aifn/foundation/tensor'
import { ShapeError } from 'aifn/foundation/errors'
import { dense, EPS, matrix, untraced, vector } from './dense'

/** The result of `svd`. */
export type SVD = {
  /** Left singular vectors as columns, m×k with k = min(m, n). */
  U: Tensor
  /** Singular values in descending order, length k. */
  S: Tensor
  /**
   * Right singular vectors as columns, n×k (so A = U diag(S) Vᵀ; NumPy returns Vᵀ as `vh`). Each pair (uⱼ, vⱼ) is
   * signed so that the largest-magnitude component of vⱼ (the first of equals) is positive.
   */
  V: Tensor
  /** Number of Jacobi sweeps used. */
  sweeps: number
  /** False when `maxSweeps` ran out before every pair of columns was orthogonal to working precision. */
  converged: boolean
}

type Raw = {
  U: Float64Array
  S: Float64Array
  V: Float64Array
  m: number
  n: number
  sweeps: number
  converged: boolean
}

/** One-sided Jacobi on an m×n matrix with m ≥ n (row-major in `a`, overwritten). */
function jacobi(a: Float64Array, m: number, n: number, maxSweeps: number): Raw {
  const V = new Float64Array(n * n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  let sweeps = 0
  let converged = false
  for (; sweeps < maxSweeps; sweeps++) {
    let rotated = false
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < m; i++) {
          const x = a[i * n + p]
          const y = a[i * n + q]
          alpha += x * x
          beta += y * y
          gamma += x * y
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue
        rotated = true
        const zeta = (beta - alpha) / (2 * gamma)
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = c * t
        for (let i = 0; i < m; i++) {
          const x = a[i * n + p]
          const y = a[i * n + q]
          a[i * n + p] = c * x - s * y
          a[i * n + q] = s * x + c * y
        }
        for (let i = 0; i < n; i++) {
          const x = V[i * n + p]
          const y = V[i * n + q]
          V[i * n + p] = c * x - s * y
          V[i * n + q] = s * x + c * y
        }
      }
    }
    if (!rotated) {
      converged = true
      break
    }
  }
  const S = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    let s = 0
    for (let i = 0; i < m; i++) s += a[i * n + j] * a[i * n + j]
    S[j] = Math.sqrt(s)
  }
  return { U: a, S, V, m, n, sweeps, converged }
}

/**
 * Normalise U's columns and complete those of zero singular values to an orthonormal set, sort by descending
 * singular value, and fix signs. Returns U (m×n), S (n), V (n×n).
 */
function finish({ U, S, V, m, n }: Raw): { U: Float64Array; S: Float64Array; V: Float64Array } {
  const order = Array.from({ length: n }, (_, j) => j).sort((i, j) => S[j] - S[i])
  const u = new Float64Array(m * n)
  const v = new Float64Array(n * n)
  const s = new Float64Array(n)
  const smax = S[order[0]] ?? 0
  order.forEach((src, col) => {
    s[col] = S[src]
    for (let i = 0; i < n; i++) v[i * n + col] = V[i * n + src]
    if (S[src] > 0 && S[src] > smax * EPS * EPS) for (let i = 0; i < m; i++) u[i * n + col] = U[i * n + src] / S[src]
    else fillOrthogonal(u, m, n, col)
    let big = 0
    for (let i = 0; i < n; i++) if (Math.abs(v[i * n + col]) > Math.abs(v[big * n + col])) big = i
    if (v[big * n + col] < 0) {
      for (let i = 0; i < n; i++) v[i * n + col] = -v[i * n + col]
      for (let i = 0; i < m; i++) u[i * n + col] = -u[i * n + col]
    }
  })
  return { U: u, S: s, V: v }
}

/** Set column `col` of u (m×n) to a unit vector orthogonal to columns 0…col−1, by Gram–Schmidt on basis vectors. */
function fillOrthogonal(u: Float64Array, m: number, n: number, col: number): void {
  let best = new Float64Array(m)
  let bestNorm = -1
  for (let e = 0; e < m; e++) {
    const w = new Float64Array(m)
    w[e] = 1
    for (let pass = 0; pass < 2; pass++) {
      for (let c = 0; c < col; c++) {
        let d = 0
        for (let i = 0; i < m; i++) d += u[i * n + c] * w[i]
        for (let i = 0; i < m; i++) w[i] -= d * u[i * n + c]
      }
    }
    const norm = Math.hypot(...w)
    if (norm > bestNorm) {
      best = w
      bestNorm = norm
    }
    if (norm > 0.5) break
  }
  for (let i = 0; i < m; i++) u[i * n + col] = best[i] / bestNorm
}

/**
 * Thin singular value decomposition A = U diag(S) Vᵀ of an m×n matrix: U is m×k, S has length k and V is n×k, with
 * k = min(m, n) and S descending. Left singular vectors of zero singular values are completed to an orthonormal set.
 */
export function svd(a: Tensor, { maxSweeps = 60 }: { maxSweeps?: number } = {}): SVD {
  untraced(a, 'svd')
  const { m, n, a: data } = dense(a, 'svd')
  if (m >= n) {
    const raw = jacobi(data, m, n, maxSweeps)
    const { U, S, V } = finish(raw)
    return { U: matrix(U, m, n), S: vector(S), V: matrix(V, n, n), sweeps: raw.sweeps, converged: raw.converged }
  }
  // A wide matrix: decompose Aᵀ = V diag(S) Uᵀ and swap the factors.
  const t = new Float64Array(n * m)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) t[j * m + i] = data[i * n + j]
  const raw = jacobi(t, n, m, maxSweeps)
  const { U, S, V } = finish(raw)
  // Signs were fixed on Aᵀ's right vectors (our U); refix them on our V for a consistent convention.
  const k = m
  for (let col = 0; col < k; col++) {
    let big = 0
    for (let i = 0; i < n; i++) if (Math.abs(U[i * k + col]) > Math.abs(U[big * k + col])) big = i
    if (U[big * k + col] < 0) {
      for (let i = 0; i < n; i++) U[i * k + col] = -U[i * k + col]
      for (let i = 0; i < m; i++) V[i * k + col] = -V[i * k + col]
    }
  }
  return { U: matrix(V, m, k), S: vector(S), V: matrix(U, n, k), sweeps: raw.sweeps, converged: raw.converged }
}

/** The default relative cutoff for treating a singular value as zero: max(m, n)·ε, as NumPy's `matrix_rank`. */
function defaultRtol(m: number, n: number): number {
  return Math.max(m, n) * EPS
}

/**
 * Moore–Penrose pseudo-inverse (n×m) of an m×n matrix from its SVD: singular values at most `rtol`·σ_max are
 * treated as zero (default max(m, n)·ε).
 */
export function pinv(a: Tensor, { rtol }: { rtol?: number } = {}): Tensor {
  const [m, n] = a.shape
  const { U, S, V } = svd(a)
  const k = S.shape[0]
  const cutoff = (rtol ?? defaultRtol(m, n)) * (S.data[0] ?? 0)
  const out = new Float64Array(n * m)
  for (let c = 0; c < k; c++) {
    const s = S.data[c]
    if (!(s > cutoff)) continue
    for (let i = 0; i < n; i++) {
      const vi = V.data[i * k + c] / s
      for (let j = 0; j < m; j++) out[i * m + j] += vi * U.data[j * k + c]
    }
  }
  return matrix(out, n, m)
}

/** The result of `lstsq`. */
export type LeastSquares = {
  /** The minimum-norm least-squares solution (n, or n×r for a matrix right-hand side). */
  x: Tensor
  /** Squared residual norm ‖Ax − b‖² (per column for a matrix right-hand side). */
  residuals: Tensor
  /** Numerical rank: the number of singular values above the cutoff. */
  rank: number
  /** Singular values of A, descending. */
  singularValues: Tensor
}

/**
 * Least squares: the x minimising ‖Ax − b‖, and among those the one of least norm, from the SVD of A (m×n) with
 * singular values at most `rtol`·σ_max treated as zero (default max(m, n)·ε). Rank deficiency is reported by `rank`.
 */
export function lstsq(a: Tensor, b: Tensor, { rtol }: { rtol?: number } = {}): LeastSquares {
  const [m, n] = a.shape
  const isVector = b.shape.length === 1
  const rhs = dense(isVector ? reshape(b, [-1, 1]) : b, 'lstsq')
  if (rhs.m !== m) throw new ShapeError('lstsq', `lstsq: A has ${m} rows but b has ${rhs.m}`)
  const r = rhs.n
  const { U, S, V } = svd(a)
  const k = S.shape[0]
  const cutoff = (rtol ?? defaultRtol(m, n)) * (S.data[0] ?? 0)
  const x = new Float64Array(n * r)
  let rank = 0
  for (let c = 0; c < k; c++) {
    const s = S.data[c]
    if (!(s > cutoff)) continue
    rank++
    for (let col = 0; col < r; col++) {
      let d = 0
      for (let j = 0; j < m; j++) d += U.data[j * k + c] * rhs.a[j * r + col]
      d /= s
      for (let i = 0; i < n; i++) x[i * r + col] += V.data[i * k + c] * d
    }
  }
  const residuals = new Float64Array(r)
  const A = dense(a, 'lstsq')
  for (let col = 0; col < r; col++) {
    for (let j = 0; j < m; j++) {
      let e = -rhs.a[j * r + col]
      for (let i = 0; i < n; i++) e += A.a[j * n + i] * x[i * r + col]
      residuals[col] += e * e
    }
  }
  return {
    x: isVector ? vector(x) : matrix(x, n, r),
    residuals: vector(residuals),
    rank,
    singularValues: S,
  }
}

/** The 2-norm condition number σ_max / σ_min (∞ when σ_min = 0) of an m×n matrix, over its min(m, n) singular values. */
export function conditionNumber(a: Tensor): number {
  const { S } = svd(a)
  const k = S.shape[0]
  if (k === 0) return 0
  const smin = S.data[k - 1]
  return smin === 0 ? Infinity : S.data[0] / smin
}
