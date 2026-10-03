/**
 * Penalised B-spline smoothing for the GAM chapter's core figures: equally spaced B-spline bases, derivative and
 * difference penalties, penalised least squares, its Bayesian posterior and the Demmler–Reinsch basis. Bases have at
 * most a few dozen columns, so dense O(p³) linear algebra is enough.
 */
import { bsplineRow, solve, type Matrix } from '../../regression/nonlinear-regression/_shared/splines'

export type { Matrix }

/** Equally spaced knots with spacing (hi − lo)/K, extended q spans beyond each end as in P-splines: K + q functions. */
export function uniformKnots(K: number, lo: number, hi: number, q = 3): number[] {
  const h = (hi - lo) / K
  return Array.from({ length: K + 2 * q + 1 }, (_, i) => lo + (i - q) * h)
}

/** Number of B-splines of degree q on knot vector t. */
export const basisSize = (t: number[], q = 3) => t.length - q - 1

/**
 * r-th derivatives at x of all degree-q B-splines on t, from de Boor's rule
 * B'_{i,q} = q (B_{i,q−1}/(t_{i+q} − t_i) − B_{i+1,q−1}/(t_{i+q+1} − t_{i+1})).
 */
export function derivativeRow(x: number, t: number[], q: number, r: number): number[] {
  if (r === 0) return bsplineRow(x, t, q)
  const lower = derivativeRow(x, t, q - 1, r - 1)
  return Array.from({ length: basisSize(t, q) }, (_, i) => {
    const a = t[i + q] - t[i]
    const b = t[i + q + 1] - t[i + 1]
    return q * ((a > 0 ? lower[i] / a : 0) - (b > 0 ? lower[i + 1] / b : 0))
  })
}

export const designMatrix = (xs: number[], t: number[], q = 3, r = 0): Matrix =>
  xs.map((x) => derivativeRow(x, t, q, r))

// Three-point Gauss–Legendre rule on [−1, 1]: exact for polynomials of degree ≤ 5, so for every product of two
// B-spline derivatives of a cubic basis within one knot span.
const GL_NODES = [-Math.sqrt(3 / 5), 0, Math.sqrt(3 / 5)]
const GL_WEIGHTS = [5 / 9, 8 / 9, 5 / 9]

/** S_kl = ∫_lo^hi B_k^{(m)}(x) B_l^{(m)}(x) dx, exact by Gauss–Legendre quadrature on each knot span. */
export function derivativePenalty(t: number[], q: number, m: number, lo: number, hi: number): Matrix {
  const p = basisSize(t, q)
  const S: Matrix = Array.from({ length: p }, () => Array(p).fill(0))
  for (let s = 0; s < t.length - 1; s++) {
    const a = Math.max(t[s], lo)
    const b = Math.min(t[s + 1], hi)
    if (b <= a) continue
    GL_NODES.forEach((u, g) => {
      const d = derivativeRow(((b - a) * u + a + b) / 2, t, q, m)
      const w = (GL_WEIGHTS[g] * (b - a)) / 2
      for (let k = 0; k < p; k++) if (d[k] !== 0) for (let l = 0; l < p; l++) S[k][l] += w * d[k] * d[l]
    })
  }
  return S
}

/** DᵀD for the d-th order difference matrix D (p − d rows), the P-spline penalty. */
export function differencePenalty(p: number, d: number): Matrix {
  // Rows of D are the binomial coefficients of (1 − E)^d with alternating signs.
  const c = [1]
  for (let j = 0; j < d; j++) c.push(0)
  for (let j = 1; j <= d; j++) for (let k = j; k > 0; k--) c[k] -= c[k - 1]
  const P: Matrix = Array.from({ length: p }, () => Array(p).fill(0))
  for (let r = 0; r + d < p; r++)
    for (let a = 0; a <= d; a++) for (let b = 0; b <= d; b++) P[r + a][r + b] += c[a] * c[b]
  return P
}

export const crossProduct = (B: Matrix): Matrix => {
  const p = B[0].length
  const G: Matrix = Array.from({ length: p }, () => Array(p).fill(0))
  for (const row of B)
    for (let j = 0; j < p; j++) if (row[j] !== 0) for (let k = 0; k < p; k++) G[j][k] += row[j] * row[k]
  return G
}

export const transposeTimes = (B: Matrix, y: number[]): number[] =>
  B[0].map((_, j) => B.reduce((acc, row, i) => acc + row[j] * y[i], 0))

export const times = (B: Matrix, v: number[]): number[] => B.map((row) => row.reduce((acc, b, j) => acc + b * v[j], 0))

export const quadForm = (A: Matrix, v: number[]) => times(A, v).reduce((acc, u, i) => acc + u * v[i], 0)

/** Inverse of a small nonsingular matrix, column by column. */
export function inverse(A: Matrix): Matrix {
  const p = A.length
  const cols = Array.from({ length: p }, (_, k) =>
    solve(
      A,
      A.map((_, i) => (i === k ? 1 : 0)),
    ),
  )
  return A.map((_, i) => cols.map((c) => c[i]))
}

/** Lower-triangular L with A = L Lᵀ. A tiny jitter guards against round-off in positive semi-definite input. */
export function cholesky(A: Matrix): Matrix {
  const p = A.length
  const L: Matrix = Array.from({ length: p }, () => Array(p).fill(0))
  for (let i = 0; i < p; i++)
    for (let j = 0; j <= i; j++) {
      let s = A[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]
    }
  return L
}

/** Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations, eigenvalues ascending. */
export function symmetricEigen(A: Matrix): { values: number[]; vectors: Matrix } {
  const p = A.length
  const M = A.map((r) => [...r])
  const V: Matrix = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)))
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += M[i][j] ** 2
    if (off < 1e-22) break
    for (let i = 0; i < p; i++)
      for (let j = i + 1; j < p; j++) {
        if (Math.abs(M[i][j]) < 1e-300) continue
        const theta = (M[j][j] - M[i][i]) / (2 * M[i][j])
        const tt = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(tt * tt + 1)
        const s = tt * c
        for (let k = 0; k < p; k++) {
          const a = M[k][i]
          const b = M[k][j]
          M[k][i] = c * a - s * b
          M[k][j] = s * a + c * b
        }
        for (let k = 0; k < p; k++) {
          const a = M[i][k]
          const b = M[j][k]
          M[i][k] = c * a - s * b
          M[j][k] = s * a + c * b
        }
        for (let k = 0; k < p; k++) {
          const a = V[k][i]
          const b = V[k][j]
          V[k][i] = c * a - s * b
          V[k][j] = s * a + c * b
        }
      }
  }
  const order = M.map((r, i) => [r[i], i]).sort((a, b) => a[0] - b[0])
  return { values: order.map(([v]) => v), vectors: V.map((row) => order.map(([, i]) => row[i])) }
}

export type PenalisedFit = {
  coef: number[]
  /** (BᵀB + λP)⁻¹. */
  inv: Matrix
  /** trace of F = (BᵀB + λP)⁻¹ BᵀB. */
  edf: number
  rss: number
  /** βᵀPβ, without λ. */
  penalty: number
}

/** Penalised least squares: minimise ‖y − Bβ‖² + λ βᵀPβ. */
export function penalisedFit(B: Matrix, BtB: Matrix, y: number[], P: Matrix, lambda: number): PenalisedFit {
  const A = BtB.map((row, j) => row.map((v, k) => v + lambda * P[j][k] + (j === k ? 1e-10 : 0)))
  const inv = inverse(A)
  const coef = times(inv, transposeTimes(B, y))
  let edf = 0
  for (let j = 0; j < BtB.length; j++) for (let k = 0; k < BtB.length; k++) edf += inv[j][k] * BtB[k][j]
  const fitted = times(B, coef)
  const rss = y.reduce((acc, yi, i) => acc + (yi - fitted[i]) ** 2, 0)
  return { coef, inv, edf, rss, penalty: quadForm(P, coef) }
}

/**
 * Demmler–Reinsch basis: with BᵀB = L Lᵀ and L⁻¹ P L⁻ᵀ = U D Uᵀ, the columns of B L⁻ᵀ U are orthonormal over the data
 * and the penalty is diagonal in them. Returns the eigenvalues d_k (ascending) and the map γ ↦ β = L⁻ᵀ U γ.
 */
export function demmlerReinsch(BtB: Matrix, P: Matrix): { d: number[]; toCoef: Matrix } {
  const p = BtB.length
  const L = cholesky(BtB.map((row, j) => row.map((v, k) => v + (j === k ? 1e-10 : 0))))
  const Linv = inverse(L)
  // M = L⁻¹ P L⁻ᵀ
  const LP = Linv.map((row) => P[0].map((_, k) => row.reduce((acc, v, j) => acc + v * P[j][k], 0)))
  const M = LP.map((row) => Linv.map((r) => row.reduce((acc, v, j) => acc + v * r[j], 0)))
  const { values, vectors } = symmetricEigen(M.map((row, i) => row.map((v, j) => (v + M[j][i]) / 2)))
  // L⁻ᵀ U
  const toCoef = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, k) => Linv.reduce((acc, row, j) => acc + row[i] * vectors[j][k], 0)),
  )
  return { d: values.map((v) => Math.max(v, 0)), toCoef }
}
