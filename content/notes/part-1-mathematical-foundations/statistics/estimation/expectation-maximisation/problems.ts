import { normal, stream, uniform } from 'aifn-compute/foundation/random'
/**
 * One-parameter EM problems for the bound figure. Each has an exact log-likelihood ℓ(θ), the EM map θ ↦ M(θ), and the
 * KL divergence between the E-step distribution built at θ_q and the exact posterior at θ. The bound built at θ_q is
 * then 𝓛(q, θ) = ℓ(θ) − KL, by the exact decomposition, which is cheaper and more accurate than evaluating
 * E_q[log p(x, z | θ)] + H(q) directly.
 */

export type EmProblem = {
  label: string
  parameter: string
  range: [number, number]
  /** Height of the plot below the maximum of ℓ, in log-likelihood units. */
  span: number
  start: number
  ll: (theta: number) => number
  step: (theta: number) => number
  gap: (thetaQ: number, theta: number) => number
}

/** KL(Bern(a) ‖ Bern(b)), with 0 log 0 = 0. */
function klBernoulli(a: number, b: number): number {
  const term = (p: number, q: number) => (p <= 0 ? 0 : p * Math.log(p / q))
  return term(a, b) + term(1 - a, 1 - b)
}

// Dempster, Laird & Rubin (1977): 197 animals in four categories with probabilities 1/2 + θ/4, (1 − θ)/4, (1 − θ)/4, θ/4.
const Y = [125, 18, 20, 34] as const
// The latent split of the first cell: of its y₁ animals, x₂ ~ Binomial(y₁, p(θ)) came from the θ/4 part.
const split = (t: number) => t / 4 / (0.5 + t / 4)

export const linkage: EmProblem = {
  label: 'genetic linkage',
  parameter: 'θ',
  range: [0.2, 0.95],
  span: 30,
  start: 0.25,
  ll: (t) => Y[0] * Math.log(0.5 + t / 4) + (Y[1] + Y[2]) * Math.log((1 - t) / 4) + Y[3] * Math.log(t / 4),
  step: (t) => {
    const x2 = Y[0] * split(t)
    return (x2 + Y[3]) / (x2 + Y[1] + Y[2] + Y[3])
  },
  gap: (tq, t) => Y[0] * klBernoulli(split(tq), split(t)),
}

// Symmetric mixture ½N(μ, 1) + ½N(−μ, 1): the likelihood is even in μ, with maxima at ±μ̂ and a stationary point at 0.
const XS: number[] = (() => {
  const g = stream(11)
  return Array.from({ length: 60 }, () => (uniform(g) < 0.5 ? 1 : -1) * 1.2 + normal(g))
})()
const phi = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
/** Posterior probability that x came from the +μ component: φ(x − μ)/(φ(x − μ) + φ(x + μ)) = sigmoid(2μx). */
const resp = (m: number, x: number) => 1 / (1 + Math.exp(-2 * m * x))

export const mixture: EmProblem = {
  label: 'symmetric mixture',
  parameter: 'μ',
  range: [-2.5, 2.5],
  span: 25,
  start: 0.3,
  ll: (m) => XS.reduce((s, x) => s + Math.log(0.5 * phi(x - m) + 0.5 * phi(x + m)), 0),
  step: (m) => XS.reduce((s, x) => s + (2 * resp(m, x) - 1) * x, 0) / XS.length,
  gap: (mq, m) => XS.reduce((s, x) => s + klBernoulli(resp(mq, x), resp(m, x)), 0),
}

/** θ₀, θ₁, … under repeated EM steps. */
export function iterate(p: EmProblem, start: number, n: number): number[] {
  const out = [start]
  for (let i = 0; i < n; i++) out.push(p.step(out[i]))
  return out
}

/** The rate of linear convergence at a fixed point: |M′(θ̂)|, by a central difference. */
export function rate(p: EmProblem, fixed: number): number {
  const h = 1e-5
  return Math.abs((p.step(fixed + h) - p.step(fixed - h)) / (2 * h))
}
