/**
 * Two-level hierarchical Gaussian processes on one-dimensional inputs: a shared function g ~ GP(0, k_g) and replicate
 * functions f_i | g ~ GP(g, k_f), observed with Gaussian noise of variance τ⁻¹. Matrices stay small (under a few
 * hundred rows), so the dense helpers of the parent category are enough.
 */
import { addDiagonal, cholesky, cholSolve, forward, gram, logDet, type Kernel, type Matrix } from '../../_shared/gp'

export type Replicate = { x: number[]; y: number[] }

export type Band = { mean: number[]; variance: number[] }

export type HgpPosterior = {
  /** Posterior of the shared function g on the grid. */
  g: Band
  /** Posterior of each replicate function f_i on the grid, noise excluded. */
  f: Band[]
  /** ln p(ŷ | x̂, θ). */
  logMarginal: number
}

/**
 * Stacks every replicate into one vector ŷ with covariance Σ, block (i, j) = K_g + δ_ij (K_f + τ⁻¹I), and conditions
 * on it. cov(ŷ, g(s)) = k_g(x̂, s); cov(ŷ, f_j(s)) adds k_f(x̂, s) on the rows of replicate j.
 */
export function hgpPosterior(
  kg: Kernel,
  kf: Kernel,
  noiseVar: number,
  reps: Replicate[],
  grid: number[],
): HgpPosterior {
  const x = reps.flatMap((r) => r.x)
  const y = reps.flatMap((r) => r.y)
  const owner = reps.flatMap((r, i) => r.x.map(() => i))
  const sigma: Matrix = x.map((a, p) => x.map((b, q) => kg(a, b) + (owner[p] === owner[q] ? kf(a, b) : 0)))
  const l = cholesky(addDiagonal(sigma, noiseVar))
  const alpha = cholSolve(l, y)
  const z = forward(l, y)
  const logMarginal = -0.5 * z.reduce((s, v) => s + v * v, 0) - 0.5 * logDet(l) - 0.5 * x.length * Math.log(2 * Math.PI)

  const band = (cross: (s: number) => number[], prior: (s: number) => number): Band => {
    const mean: number[] = []
    const variance: number[] = []
    for (const s of grid) {
      const c = cross(s)
      const v = forward(l, c)
      mean.push(c.reduce((acc, ci, p) => acc + ci * alpha[p], 0))
      variance.push(Math.max(prior(s) - v.reduce((acc, u) => acc + u * u, 0), 0))
    }
    return { mean, variance }
  }

  const g = band(
    (s) => x.map((a) => kg(a, s)),
    (s) => kg(s, s),
  )
  const f = reps.map((_, j) =>
    band(
      (s) => x.map((a, p) => kg(a, s) + (owner[p] === j ? kf(a, s) : 0)),
      (s) => kg(s, s) + kf(s, s),
    ),
  )
  return { g, f, logMarginal }
}

/** ln p(ŷ | x̂, θ) alone, for hyperparameter search. */
export function hgpLogMarginal(kg: Kernel, kf: Kernel, noiseVar: number, reps: Replicate[]): number {
  const x = reps.flatMap((r) => r.x)
  const y = reps.flatMap((r) => r.y)
  const owner = reps.flatMap((r, i) => r.x.map(() => i))
  const sigma: Matrix = x.map((a, p) => x.map((b, q) => kg(a, b) + (owner[p] === owner[q] ? kf(a, b) : 0)))
  const l = cholesky(addDiagonal(sigma, noiseVar))
  const z = forward(l, y)
  return -0.5 * z.reduce((s, v) => s + v * v, 0) - 0.5 * logDet(l) - 0.5 * x.length * Math.log(2 * Math.PI)
}

export type ClassModel = {
  /** Predictive mean of a new replicate: E[g | Y] on the common grid. */
  mean: number[]
  /** Cholesky factor of the predictive covariance cov[g | Y] + K_f + τ⁻¹I of a new replicate. */
  chol: Matrix
}

/**
 * The predictive distribution of a new replicate when n training replicates share one grid x. The mean ȳ of the
 * replicates is sufficient for g: g | Y has mean K_g (K_g + A/n)⁻¹ ȳ and covariance K_g − K_g (K_g + A/n)⁻¹ K_g with
 * A = K_f + τ⁻¹I. A new replicate adds its own deviation and noise, A.
 */
export function classModel(kg: Kernel, kf: Kernel, noiseVar: number, x: number[], ys: number[][]): ClassModel {
  const n = ys.length
  const t = x.length
  const Kg = gram(kg, x, x)
  const A = addDiagonal(gram(kf, x, x), noiseVar)
  const ybar = x.map((_, k) => ys.reduce((s, y) => s + y[k], 0) / n)
  const lm = cholesky(Kg.map((row, i) => row.map((v, j) => v + A[i][j] / n)))
  const beta = cholSolve(lm, ybar)
  const mean = Kg.map((row) => row.reduce((s, v, j) => s + v * beta[j], 0))
  // K_g (K_g + A/n)⁻¹ K_g = Wᵀ W with W = L⁻¹ K_g.
  const w = Kg.map((col) => forward(lm, col))
  const cov: Matrix = Array.from({ length: t }, (_, i) =>
    Array.from({ length: t }, (_, j) => Kg[i][j] - w[i].reduce((s, u, m) => s + u * w[j][m], 0) + A[i][j]),
  )
  return { mean, chol: cholesky(cov) }
}

/** Anomaly score S(y*) = −ln p(y* | class model): the negative log predictive density of the whole curve. */
export function anomalyScore(model: ClassModel, y: number[]): number {
  const r = y.map((v, i) => v - model.mean[i])
  const z = forward(model.chol, r)
  return 0.5 * z.reduce((s, v) => s + v * v, 0) + 0.5 * logDet(model.chol) + 0.5 * y.length * Math.log(2 * Math.PI)
}
