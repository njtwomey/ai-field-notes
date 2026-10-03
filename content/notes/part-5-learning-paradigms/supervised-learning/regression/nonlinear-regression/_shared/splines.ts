/**
 * Cubic B-spline bases and penalised least squares for the spline and additive-model figures. Bases have at most a
 * few dozen columns, so a dense Gaussian elimination is enough.
 */

export type Matrix = number[][]

/** Clamped knot vector: each boundary repeated degree + 1 times around the sorted interior knots. */
export function knotVector(interior: number[], lo: number, hi: number, degree = 3): number[] {
  const inner = [...interior].filter((k) => k > lo && k < hi).sort((a, b) => a - b)
  return [...Array(degree + 1).fill(lo), ...inner, ...Array(degree + 1).fill(hi)]
}

/** Values at x of all B-splines of the given degree on knot vector t (Cox–de Boor recursion). */
export function bsplineRow(x: number, t: number[], degree = 3): number[] {
  const m = t.length - 1
  const hi = t[m]
  // Degree 0: the indicator of the knot span holding x; the right boundary belongs to the last non-empty span.
  let b = Array.from({ length: m }, (_, i) =>
    (x >= t[i] && x < t[i + 1]) || (x === hi && t[i] < hi && t[i + 1] === hi) ? 1 : 0,
  )
  for (let d = 1; d <= degree; d++) {
    const next = Array(m - d).fill(0)
    for (let i = 0; i < m - d; i++) {
      const l = t[i + d] - t[i]
      const r = t[i + d + 1] - t[i + 1]
      next[i] = (l > 0 ? ((x - t[i]) / l) * b[i] : 0) + (r > 0 ? ((t[i + d + 1] - x) / r) * b[i + 1] : 0)
    }
    b = next
  }
  return b
}

/** Solve A u = v by Gaussian elimination with partial pivoting. A is small and dense; neither input is modified. */
export function solve(A: Matrix, v: number[]): number[] {
  const n = v.length
  const M = A.map((row, i) => [...row, v[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    const piv = M[c][c] || 1e-300
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const u = Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * u[k]
    u[r] = s / (M[r][r] || 1e-300)
  }
  return u
}

/** Roughness penalty Ω with Ω_jk = ∫ B_j''(x) B_k''(x) dx, by central differences on a fine grid. */
export function roughness(t: number[], lo: number, hi: number, degree = 3, steps = 400): Matrix {
  const h = (hi - lo) / steps
  const e = h / 4
  const p = t.length - degree - 1
  const O: Matrix = Array.from({ length: p }, () => Array(p).fill(0))
  for (let s = 0; s < steps; s++) {
    const x = lo + (s + 0.5) * h
    const a = bsplineRow(x - e, t, degree)
    const b = bsplineRow(x, t, degree)
    const c = bsplineRow(x + e, t, degree)
    const d2 = b.map((bj, j) => (a[j] - 2 * bj + c[j]) / (e * e))
    for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) O[j][k] += d2[j] * d2[k] * h
  }
  return O
}

export type Basis = {
  knots: number[]
  degree: number
  /** n × p design matrix. */
  B: Matrix
  /** Bᵀ B. */
  BtB: Matrix
  /** Roughness penalty Ω. */
  omega: Matrix
}

export type Smoother = Basis & {
  lambda: number
  /** Bᵀ B + λ Ω. */
  A: Matrix
  /** Effective degrees of freedom: the trace of the smoother matrix S = B A⁻¹ Bᵀ. */
  edf: number
}

/** Cubic B-spline basis on inputs x with the given interior knots, and its roughness penalty. */
export function makeBasis(x: number[], interior: number[], lo: number, hi: number): Basis {
  const degree = 3
  const knots = knotVector(interior, lo, hi, degree)
  const B = x.map((xi) => bsplineRow(xi, knots, degree))
  const p = B[0].length
  const BtB: Matrix = Array.from({ length: p }, (_, j) =>
    Array.from({ length: p }, (_, k) => B.reduce((s, row) => s + row[j] * row[k], 0)),
  )
  return { knots, degree, B, BtB, omega: roughness(knots, lo, hi, degree) }
}

/** The penalised smoother for smoothing parameter λ ≥ 0. */
export function penalise(basis: Basis, lambda: number): Smoother {
  const { BtB, omega } = basis
  const p = BtB.length
  // A tiny ridge keeps A invertible when a knot span holds no data.
  const A = BtB.map((row, j) => row.map((v, k) => v + lambda * omega[j][k] + (j === k ? 1e-9 : 0)))
  // tr(B A⁻¹ Bᵀ) = tr(A⁻¹ BᵀB): solve for each column of BᵀB and sum the diagonal.
  let edf = 0
  for (let k = 0; k < p; k++)
    edf += solve(
      A,
      BtB.map((row) => row[k]),
    )[k]
  return { ...basis, lambda, A, edf }
}

/** Penalised cubic regression spline on inputs x with the given interior knots and smoothing parameter λ. */
export function makeSmoother(x: number[], interior: number[], lo: number, hi: number, lambda: number): Smoother {
  return penalise(makeBasis(x, interior, lo, hi), lambda)
}

/** log |A| of a symmetric positive-definite matrix, by Gaussian elimination without pivoting. */
export function logDet(A: Matrix): number {
  const n = A.length
  const M = A.map((row) => [...row])
  let s = 0
  for (let c = 0; c < n; c++) {
    s += Math.log(M[c][c])
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c]
      for (let k = c; k < n; k++) M[r][k] -= f * M[c][k]
    }
  }
  return s
}

/** Fit the smoother to responses y: coefficients and fitted values at the training inputs. */
export function smooth(s: Smoother, y: number[]): { coef: number[]; fitted: number[] } {
  const p = s.A.length
  const Bty = Array.from({ length: p }, (_, j) => s.B.reduce((acc, row, i) => acc + row[j] * y[i], 0))
  const coef = solve(s.A, Bty)
  return { coef, fitted: s.B.map((row) => row.reduce((acc, b, j) => acc + b * coef[j], 0)) }
}

/** Evaluate the fitted spline at new inputs. */
export function evaluate(s: Smoother, coef: number[], xs: number[]): number[] {
  return xs.map((x) => bsplineRow(x, s.knots, s.degree).reduce((acc, b, j) => acc + b * coef[j], 0))
}
