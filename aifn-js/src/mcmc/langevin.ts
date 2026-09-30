/**
 * Langevin samplers: the unadjusted Langevin algorithm (ULA), the Metropolis-adjusted Langevin algorithm (MALA;
 * Roberts & Tweedie, 1996) and stochastic-gradient Langevin dynamics (SGLD; Welling & Teh, 2011).
 *
 * All discretise the Langevin diffusion dθ = ∇ log π(θ) dt + √2 dW, whose stationary law is π, with the Euler step
 * θ′ = θ + h ∇ log π(θ) + √(2h) ξ, ξ ~ N(0, I). ULA keeps every step and so samples a biased law π_h ≠ π whose bias
 * grows with h; MALA treats the step as a proposal and corrects it with a Metropolis–Hastings test, so π is exact.
 */

import { stream as makeStream, type Stream } from 'aifn/random'
import { fromData, type Tensor, type Vector } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { badLogDensity } from './metropolis'
import type { AcceptRejectState, ChainStart, ChainState, Target, VectorLike } from './types'
import { allFinite, data, logDensityAndGrad, standardNormals, toF64, vec, type F64 } from './util'

/** The state of `unadjustedLangevin` and `mala`. */
export type LangevinState = AcceptRejectState & {
  /** ∇ log π(x). */
  grad: Vector
  /** The mean of the last proposal, x + h∇ log π(x) (the drift step before noise). */
  drift: Vector
  /** The step size h. */
  stepSize: number
}

/** Options for the Langevin samplers. */
export type LangevinOptions = {
  /** Time step h of the Euler step θ′ = θ + h∇ log π(θ) + √(2h)ξ. Default 0.1. */
  stepSize?: number
}

function langevin(target: Target, name: string, adjusted: boolean, h: number): Algorithm<ChainStart, LangevinState> {
  const d = target.dim
  const sd = Math.sqrt(2 * h)
  // log q(to | from) up to a constant, for q(· | x) = N(x + h∇ log π(x), 2hI).
  const logQ = (to: F64, from: F64, gradFrom: F64) => {
    let s = 0
    for (let i = 0; i < d; i++) s += (to[i] - from[i] - h * gradFrom[i]) ** 2
    return -s / (4 * h)
  }
  return {
    name,
    init: ({ x0 }, s) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new Error(`${name}: x0 has ${x.length} values for dimension ${d}`)
      const { value, grad } = logDensityAndGrad(target, x)
      const X = vec(x)
      return {
        t: 0,
        x: X,
        logDensity: value,
        grad: vec(grad),
        drift: X,
        stepSize: h,
        proposal: X,
        proposalLogDensity: value,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: false,
        acceptedCount: 0,
        acceptanceRate: NaN,
        stream: s ?? makeStream(name),
        diverged: badLogDensity(value) || !allFinite(x) || !allFinite(grad),
      }
    },
    step: (s) => {
      const draws = s.stream.child(s.t)
      const xi = standardNormals(draws.child('noise'), d)
      const x = data(s.x)
      const g = data(s.grad)
      const drift = new Float64Array(d)
      const y = new Float64Array(d)
      for (let i = 0; i < d; i++) {
        drift[i] = x[i] + h * g[i]
        y[i] = drift[i] + sd * xi[i]
      }
      const next = logDensityAndGrad(target, y)
      let logRatio = 0
      let acceptance = 1
      let accepted = true
      if (adjusted) {
        logRatio = next.value - s.logDensity + logQ(x, y, next.grad) - logQ(y, x, g)
        acceptance = Number.isNaN(logRatio) ? 0 : Math.min(1, Math.exp(logRatio))
        accepted = draws.uniform() < acceptance
      }
      const acceptedCount = s.acceptedCount + (accepted ? 1 : 0)
      return {
        ...s,
        t: s.t + 1,
        x: accepted ? vec(y) : s.x,
        logDensity: accepted ? next.value : s.logDensity,
        grad: accepted ? vec(next.grad) : s.grad,
        drift: vec(drift),
        proposal: vec(y),
        proposalLogDensity: next.value,
        logAcceptanceRatio: logRatio,
        acceptance,
        accepted,
        acceptedCount,
        acceptanceRate: acceptedCount / (s.t + 1),
        diverged: accepted && (badLogDensity(next.value) || !allFinite(y) || !allFinite(next.grad)),
      }
    },
  }
}

/**
 * The unadjusted Langevin algorithm: θ′ = θ + h∇ log π(θ) + √(2h)ξ, every step kept (Roberts & Tweedie, 1996, §1.4
 * call it ULA and show it can be transient for large h). Its stationary law is biased by O(h): for a Gaussian
 * N(0, σ²) it is N(0, σ²/(1 − h/(2σ²))). `acceptance` is 1 on every step.
 */
export function unadjustedLangevin(
  target: Target,
  options: LangevinOptions = {},
): Algorithm<ChainStart, LangevinState> {
  return langevin(target, 'unadjusted-langevin', false, options.stepSize ?? 0.1)
}

/**
 * The Metropolis-adjusted Langevin algorithm (Roberts & Tweedie, 1996): propose the ULA step and accept with the
 * Metropolis–Hastings ratio, whose proposal density q(θ′ | θ) = N(θ + h∇ log π(θ), 2hI) is not symmetric. Its
 * optimal acceptance rate is about 0.574 in high dimension (Roberts & Rosenthal, 1998).
 */
export function mala(target: Target, options: LangevinOptions = {}): Algorithm<ChainStart, LangevinState> {
  return langevin(target, 'mala', true, options.stepSize ?? 0.1)
}

// ---------------------------------------------------------------------------------------------------------------------
// SGLD.

/**
 * A model for SGLD: N data points with per-datum log-likelihood gradients and a log-prior gradient, for the posterior
 * log π(θ) = log p(θ) + Σᵢ log p(yᵢ | θ) + const.
 */
export type MinibatchModel = {
  dim: number
  /** N, the number of data points. */
  size: number
  gradLogPrior: (theta: Vector) => VectorLike
  /** ∇θ log p(yᵢ | θ) for data point i. */
  gradLogLikelihood: (theta: Vector, i: number) => VectorLike
}

/** The state of `sgld`. */
export type SgldState = ChainState & {
  /** The last minibatch (int32 indices into the data). */
  batch: Tensor
  /** The minibatch estimate of ∇ log π at the previous point: ∇ log p(θ) + (N/n) Σ_{i∈batch} ∇ log p(yᵢ | θ). */
  gradEstimate: Vector
  /** The step size ε_t used on the last step. */
  stepSize: number
}

/** Options for `sgld`. */
export type SgldOptions = {
  /** Minibatch size n (drawn without replacement each step). Default 10. */
  batchSize?: number
  /**
   * Step size ε_t, or a schedule t ↦ ε_t. Welling & Teh use ε_t = a(b + t)^(−γ) with γ ∈ (0.5, 1] so that Σε = ∞ and
   * Σε² < ∞. Default 1e-3.
   */
  stepSize?: number | ((t: number) => number)
}

/**
 * Stochastic-gradient Langevin dynamics (Welling & Teh, 2011, eq. 4): θ′ = θ + (ε_t/2)(∇ log p(θ) +
 * (N/n) Σ_{i∈batch} ∇ log p(yᵢ | θ)) + η, η ~ N(0, ε_t I). There is no accept/reject step: as ε_t → 0 the injected
 * noise dominates the minibatch noise and the iterates sample the posterior. `logDensity` is NaN (never evaluated).
 */
export function sgld(model: MinibatchModel, options: SgldOptions = {}): Algorithm<ChainStart, SgldState> {
  const name = 'sgld'
  const { batchSize = 10, stepSize = 1e-3 } = options
  const d = model.dim
  const N = model.size
  const n = Math.min(batchSize, N)
  const rate = (t: number) => (typeof stepSize === 'function' ? stepSize(t) : stepSize)
  return {
    name,
    init: ({ x0 }, s) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new Error(`${name}: x0 has ${x.length} values for dimension ${d}`)
      return {
        t: 0,
        x: vec(x),
        logDensity: NaN,
        batch: fromData(new Int32Array(0), [0]),
        gradEstimate: vec(new Float64Array(d)),
        stepSize: NaN,
        stream: s ?? makeStream(name),
        diverged: !allFinite(x),
      }
    },
    step: (s) => {
      const draws: Stream = s.stream.child(s.t)
      // A partial Fisher–Yates shuffle picks n indices without replacement.
      const order = Int32Array.from({ length: N }, (_, i) => i)
      const pick = draws.child('batch')
      for (let i = 0; i < n; i++) {
        const j = i + pick.int(N - i)
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      const batch = order.slice(0, n)
      const x = data(s.x)
      const g = toF64(model.gradLogPrior(s.x), name)
      for (const i of batch) {
        const gi = toF64(model.gradLogLikelihood(s.x, i), name)
        for (let k = 0; k < d; k++) g[k] += (N / n) * gi[k]
      }
      const eps = rate(s.t)
      const eta = standardNormals(draws.child('noise'), d)
      const y = new Float64Array(d)
      for (let k = 0; k < d; k++) y[k] = x[k] + 0.5 * eps * g[k] + Math.sqrt(eps) * eta[k]
      return {
        ...s,
        t: s.t + 1,
        x: vec(y),
        batch: fromData(batch, [n]),
        gradEstimate: vec(g),
        stepSize: eps,
        diverged: !allFinite(y),
      }
    },
  }
}
