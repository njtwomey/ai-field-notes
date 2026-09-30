/**
 * Backfitting (Buja, Hastie and Tibshirani, 1989, Annals of Statistics 17; Hastie and Tibshirani, 1990, "Generalized
 * Additive Models", §4.4) with local scoring for non-Gaussian families (§6.5), as a traceable algorithm over any
 * additive model η = α + Σⱼ Bⱼβⱼ with a quadratic penalty per term: each step updates the working response and weights
 * from the current linear predictor, then refits each term's penalised smoother to its partial residual in turn,
 * holding the others fixed. Shared by the generalised models; `gam` builds the term designs and penalties.
 */

import type { Status } from 'aifn/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import { cholesky, choleskySolve } from 'aifn/numerics/linalg'
import { gaussianFamily, link as linkByName, type Family, type Link, type LinkName } from 'aifn/probability/likelihoods'

/** The problem a backfitting run solves: one design block and one penalty per term. */
export type BackfitProblem = {
  /** Each term's design Bⱼ [n, pⱼ] (centred, so the intercept is identifiable). */
  designs: readonly Tensor[]
  /** Each term's penalty Sⱼ [pⱼ, pⱼ], smoothing parameters already applied. */
  penalties: readonly Tensor[]
  y: Tensor
  weights?: Tensor
  /** Default the Gaussian family. */
  family?: Family
  /** Default the family's default link. */
  link?: LinkName | Link
  /** Converged when the largest change of any fⱼ(xᵢ) is below tolerance × (1 + max |f|) (default 1e-8). */
  tolerance?: number
}

/** A backfitting state. */
export type BackfitState = Status & {
  /** Sweeps done. */
  t: number
  intercept: number
  /** Each term's fitted values at the data, [terms, n]. */
  contributions: Tensor
  /** Linear predictor η = α + Σ fⱼ, [n]. */
  eta: Tensor
  /** Deviance at η. */
  deviance: number
  /** Largest change of a term's fitted value in this sweep. */
  change: number
  converged: boolean
}

/** Backfitting as an `Algorithm` (no start: α begins at the weighted mean of g(μ₀), every fⱼ at 0). */
export function backfitting(problem: BackfitProblem): Algorithm<void, BackfitState> {
  const family = problem.family ?? gaussianFamily()
  const lk = typeof problem.link === 'object' ? problem.link : linkByName(problem.link ?? family.defaultLink)
  const y = dense.data(problem.y)
  const n = y.length
  const w = problem.weights ? dense.data(problem.weights) : new Float64Array(n).fill(1)
  const tolerance = problem.tolerance ?? 1e-8
  const designs = problem.designs.map((B) => dense.data(B))
  const sizes = problem.designs.map((B) => B.shape[1])
  const penalties = problem.penalties.map((S) => dense.data(S))
  const T = designs.length
  const deviance = (eta: Float64Array) => {
    const mu = lk.inverse(fromData(eta, [n])) as Tensor
    const u = dense.data(family.unitDeviance(problem.y, mu) as Tensor)
    return u.reduce((s, v, i) => s + w[i] * v, 0)
  }
  const stateOf = (alpha: number, f: Float64Array, t: number, change: number, scale: number): BackfitState => {
    const eta = new Float64Array(n).fill(alpha)
    for (let j = 0; j < T; j++) for (let i = 0; i < n; i++) eta[i] += f[j * n + i]
    const dev = deviance(eta)
    return {
      t,
      intercept: alpha,
      contributions: fromData(f, [T, n]),
      eta: fromData(eta, [n]),
      deviance: dev,
      change,
      converged: t > 0 && change < tolerance * (1 + scale),
      diverged: !Number.isFinite(dev),
    }
  }
  return {
    name: 'backfitting',
    init: () => {
      const mu0 = family.initialMean(problem.y, fromData(w, [n]))
      const eta0 = dense.data(lk.link(mu0) as Tensor)
      let alpha = 0
      let sw = 0
      for (let i = 0; i < n; i++) {
        alpha += w[i] * eta0[i]
        sw += w[i]
      }
      return stateOf(alpha / sw, new Float64Array(T * n), 0, Infinity, 0)
    },
    step: (state) => {
      const eta = dense.data(state.eta)
      const etaT = fromData(Float64Array.from(eta), [n])
      const mu = dense.data(lk.inverse(etaT) as Tensor)
      const dmu = dense.data(lk.derivative(etaT) as Tensor)
      const V = dense.data(family.variance(fromData(Float64Array.from(mu), [n])) as Tensor)
      // Working response and weights (for the Gaussian identity model, z = y and W = w).
      const z = Float64Array.from(eta, (e, i) => e + (y[i] - mu[i]) / dmu[i])
      const W = Float64Array.from(dmu, (g, i) => (w[i] * g * g) / V[i])
      const f = Float64Array.from(dense.data(state.contributions))
      let alpha = 0
      let sw = 0
      for (let i = 0; i < n; i++) {
        let r = z[i]
        for (let j = 0; j < T; j++) r -= f[j * n + i]
        alpha += W[i] * r
        sw += W[i]
      }
      alpha /= sw
      let change = 0
      let scale = 0
      for (let j = 0; j < T; j++) {
        const B = designs[j]
        const p = sizes[j]
        const A = Float64Array.from(penalties[j])
        const b = new Float64Array(p)
        for (let i = 0; i < n; i++) {
          let r = z[i] - alpha
          for (let q = 0; q < T; q++) if (q !== j) r -= f[q * n + i]
          for (let a = 0; a < p; a++) {
            const v = B[i * p + a] * W[i]
            if (v === 0) continue
            b[a] += v * r
            for (let c = 0; c < p; c++) A[a * p + c] += v * B[i * p + c]
          }
        }
        const beta = dense.data(choleskySolve(cholesky(fromData(A, [p, p])).L, fromData(b, [p])) as Tensor)
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let a = 0; a < p; a++) v += B[i * p + a] * beta[a]
          change = Math.max(change, Math.abs(v - f[j * n + i]))
          scale = Math.max(scale, Math.abs(v))
          f[j * n + i] = v
        }
      }
      return stateOf(alpha, f, state.t + 1, change, scale)
    },
  }
}
