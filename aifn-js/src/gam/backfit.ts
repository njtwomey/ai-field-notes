/**
 * Backfitting (Buja, Hastie and Tibshirani, 1989; Hastie and Tibshirani, 1990, "Generalized Additive Models", §4.4)
 * with local scoring for non-Gaussian families (§6.5), as a traceable algorithm: each step updates the working
 * response and weights from the current linear predictor, then smooths each term's partial residual in turn with its
 * own penalised smoother, holding the others fixed.
 */

import { gaussian, link as linkByName, type Family, type Link, type LinkName } from 'aifn/glm'
import { cholesky, choleskySolve } from 'aifn/linalg'
import { fromData, toFlat, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { buildTerms, times, type BuiltTerm, type TermSpec } from './terms'

type F64 = Float64Array
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

/** The problem a backfitting run solves. */
export type BackfitProblem = {
  terms: readonly TermSpec[]
  x: Tensor
  y: Tensor
  weights?: Tensor
  family?: Family
  link?: LinkName | Link
  /** One λ per penalty in term order (default: each term's fixed λ, else 1). */
  lambdas?: number[]
  /** Converged when the largest change of any fⱼ(xᵢ) is below tol × (1 + max |f|) (default 1e-8). */
  tol?: number
}

/** A backfitting state. */
export type BackfitState = {
  intercept: number
  /** Each term's fitted values at the data, [terms, n]. */
  contributions: Tensor
  /** Linear predictor η = α + Σ fⱼ, [n]. */
  eta: Tensor
  /** Deviance at η. */
  deviance: number
  /** Largest change of a term's fitted value in this sweep. */
  change: number
  sweep: number
  converged: boolean
}

/**
 * Backfitting as an `Algorithm`. The terms are built on the data as in `gam` (centred smooths), so its fixed point is
 * the penalised fit `gam` finds at the same λ (for the Gaussian family).
 */
export function backfitting(problem: BackfitProblem): Algorithm<Record<string, never>, BackfitState> {
  const family = problem.family ?? gaussian()
  const lk = typeof problem.link === 'object' ? problem.link : linkByName(problem.link ?? family.defaultLink)
  const [n, d] = problem.x.shape
  const X = f64(problem.x)
  const terms: BuiltTerm[] = buildTerms(problem.terms, X, n, d)
  const y = f64(problem.y)
  const w = problem.weights ? f64(problem.weights) : new Float64Array(n).fill(1)
  const tol = problem.tol ?? 1e-8
  let k = 0
  const designs = terms.map((t) => times(t.raw(X, n, d), n, t.rawSize, t.Z, t.size))
  const penalties = terms.map((t) => {
    const S = new Float64Array(t.size * t.size)
    t.penalties.forEach((Sk, j) => {
      const l = problem.lambdas?.[k] ?? (Number.isNaN(t.fixedLambda[j]) ? 1 : t.fixedLambda[j])
      k++
      for (let i = 0; i < S.length; i++) S[i] += l * Sk[i]
    })
    return S
  })
  const T = terms.length
  const deviance = (eta: F64) => {
    const mu = lk.inverse(fromData(eta, [n])) as Tensor
    const u = f64(family.unitDeviance(problem.y, mu) as Tensor)
    return u.reduce((s, v, i) => s + w[i] * v, 0)
  }
  const stateOf = (alpha: number, f: F64, sweep: number, change: number, scale: number): BackfitState => {
    const eta = new Float64Array(n).fill(alpha)
    for (let j = 0; j < T; j++) for (let i = 0; i < n; i++) eta[i] += f[j * n + i]
    return {
      intercept: alpha,
      contributions: fromData(f, [T, n]),
      eta: fromData(eta, [n]),
      deviance: deviance(eta),
      change,
      sweep,
      converged: sweep > 0 && change < tol * (1 + scale),
    }
  }
  return {
    name: 'gam-backfitting',
    init: () => {
      const mu0 = f64(family.initialMean(problem.y, fromData(w, [n])))
      const eta0 = f64(lk.link(fromData(mu0, [n])) as Tensor)
      let alpha = 0
      let sw = 0
      for (let i = 0; i < n; i++) {
        alpha += w[i] * eta0[i]
        sw += w[i]
      }
      return stateOf(alpha / sw, new Float64Array(T * n), 0, Infinity, 0)
    },
    step: (state) => {
      const eta = f64(state.eta)
      const etaT = fromData(eta, [n])
      const mu = f64(lk.inverse(etaT) as Tensor)
      const dmu = f64(lk.derivative(etaT) as Tensor)
      const V = f64(family.variance(fromData(mu, [n])) as Tensor)
      // Working response and weights (for the Gaussian identity model, z = y and W = w).
      const z = Float64Array.from(eta, (e, i) => e + (y[i] - mu[i]) / dmu[i])
      const W = Float64Array.from(dmu, (g, i) => (w[i] * g * g) / V[i])
      const f = Float64Array.from(f64(state.contributions))
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
        const p = terms[j].size
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
        const beta = f64(choleskySolve(cholesky(fromData(A, [p, p])).L, fromData(b, [p])) as Tensor)
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let a = 0; a < p; a++) v += B[i * p + a] * beta[a]
          change = Math.max(change, Math.abs(v - f[j * n + i]))
          scale = Math.max(scale, Math.abs(v))
          f[j * n + i] = v
        }
      }
      return stateOf(alpha, f, state.sweep + 1, change, scale)
    },
    done: (s) => s.converged,
  }
}
