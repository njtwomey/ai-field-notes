/**
 * Online Bayesian probit regression with a factorised Gaussian belief over the weights, as in AdPredictor and the
 * email reply classifier. Features are binary and sparse: an example is the list of its active feature indices.
 */
import { Phi, vTrunc, wTrunc } from './gaussian.ts'

export type Belief = { mean: number[]; variance: number[] }

export function prior(n: number, mean: number, variance: number): Belief {
  return { mean: Array(n).fill(mean), variance: Array(n).fill(variance) }
}

/** Predictive probability of a positive label: Φ(Σμ / √(β² + Σσ²)) over the active weights. */
export function predict(b: Belief, active: number[], beta: number): number {
  let m = 0
  let s2 = beta * beta
  for (const i of active) {
    m += b.mean[i]
    s2 += b.variance[i]
  }
  return Phi(m / Math.sqrt(s2))
}

/**
 * One assumed-density-filtering step for label y ∈ {+1, −1}. Every active weight moves by its share of the total
 * variance, scaled by v of the surprise t = y·Σμ/Σ, and loses the same share of its variance scaled by w(t).
 */
export function update(b: Belief, active: number[], y: 1 | -1, beta: number): Belief {
  let m = 0
  let s2 = beta * beta
  for (const i of active) {
    m += b.mean[i]
    s2 += b.variance[i]
  }
  const s = Math.sqrt(s2)
  const t = (y * m) / s
  const v = vTrunc(t)
  const w = wTrunc(t)
  const mean = b.mean.slice()
  const variance = b.variance.slice()
  for (const i of active) {
    mean[i] = b.mean[i] + ((y * b.variance[i]) / s) * v
    variance[i] = b.variance[i] * (1 - (b.variance[i] / s2) * w)
  }
  return { mean, variance }
}
