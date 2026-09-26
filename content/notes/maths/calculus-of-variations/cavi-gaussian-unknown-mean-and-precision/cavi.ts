/**
 * Coordinate ascent variational inference for xᵢ ~ N(μ, τ⁻¹) with the conjugate prior μ | τ ~ N(μ₀, (λ₀τ)⁻¹),
 * τ ~ Gamma(a₀, b₀) (shape, rate), following Bishop (2006, §10.1.3). The exact Normal-Gamma posterior is included for
 * comparison.
 */
import { logGamma } from '@/lib/math/special'

export type Prior = { mu0: number; lambda0: number; a0: number; b0: number }
/** q(μ) = N(mu, 1/lambda) and q(τ) = Gamma(a, b). */
export type Factors = { mu: number; lambda: number; a: number; b: number }
/** Normal-Gamma: μ | τ ~ N(mu, (lambda τ)⁻¹), τ ~ Gamma(a, b). */
export type NormalGamma = { mu: number; lambda: number; a: number; b: number }

const LOG_2PI = Math.log(2 * Math.PI)

/** Digamma ψ(x) for x > 0: recurrence up to x ≥ 6, then the asymptotic series. */
export function digamma(x: number): number {
  let result = 0
  while (x < 6) {
    result -= 1 / x
    x += 1
  }
  const f = 1 / (x * x)
  return result + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
}

type Stats = { n: number; mean: number; sumSq: (c: number) => number }

export function statistics(x: number[]): Stats {
  const n = x.length
  const mean = x.reduce((s, v) => s + v, 0) / n
  return { n, mean, sumSq: (c) => x.reduce((s, v) => s + (v - c) ** 2, 0) }
}

/** q(μ) update: precision (λ₀ + N) E[τ], mean (λ₀μ₀ + N x̄)/(λ₀ + N). */
export function updateMu(q: Factors, s: Stats, p: Prior): Factors {
  const mu = (p.lambda0 * p.mu0 + s.n * s.mean) / (p.lambda0 + s.n)
  return { ...q, mu, lambda: (p.lambda0 + s.n) * (q.a / q.b) }
}

/** q(τ) update: shape a₀ + (N + 1)/2, rate b₀ + ½ E_μ[Σ(xᵢ − μ)² + λ₀(μ − μ₀)²]. */
export function updateTau(q: Factors, s: Stats, p: Prior): Factors {
  const expected = s.sumSq(q.mu) + s.n / q.lambda + p.lambda0 * ((q.mu - p.mu0) ** 2 + 1 / q.lambda)
  return { ...q, a: p.a0 + (s.n + 1) / 2, b: p.b0 + 0.5 * expected }
}

/** ELBO = E_q[log p(x, μ, τ)] + H[q(μ)] + H[q(τ)], in closed form. */
export function elbo(q: Factors, s: Stats, p: Prior): number {
  const eTau = q.a / q.b
  const eLogTau = digamma(q.a) - Math.log(q.b)
  const likelihood = (s.n / 2) * (eLogTau - LOG_2PI) - (eTau / 2) * (s.sumSq(q.mu) + s.n / q.lambda)
  const priorMu =
    0.5 * (Math.log(p.lambda0) + eLogTau - LOG_2PI) - (p.lambda0 * eTau * ((q.mu - p.mu0) ** 2 + 1 / q.lambda)) / 2
  const priorTau = p.a0 * Math.log(p.b0) - logGamma(p.a0) + (p.a0 - 1) * eLogTau - p.b0 * eTau
  const entropyMu = 0.5 * (LOG_2PI + 1 - Math.log(q.lambda))
  const entropyTau = q.a - Math.log(q.b) + logGamma(q.a) + (1 - q.a) * digamma(q.a)
  return likelihood + priorMu + priorTau + entropyMu + entropyTau
}

export function exactPosterior(s: Stats, p: Prior): NormalGamma {
  const lambda = p.lambda0 + s.n
  return {
    mu: (p.lambda0 * p.mu0 + s.n * s.mean) / lambda,
    lambda,
    a: p.a0 + s.n / 2,
    b: p.b0 + 0.5 * s.sumSq(s.mean) + (p.lambda0 * s.n * (s.mean - p.mu0) ** 2) / (2 * lambda),
  }
}

/** log p(x), the log evidence, from the normalising constants of prior and posterior. */
export function logEvidence(s: Stats, p: Prior): number {
  const post = exactPosterior(s, p)
  return (
    logGamma(post.a) -
    logGamma(p.a0) +
    p.a0 * Math.log(p.b0) -
    post.a * Math.log(post.b) +
    0.5 * Math.log(p.lambda0 / post.lambda) -
    (s.n / 2) * LOG_2PI
  )
}

/** Unnormalised log densities over (μ, τ), for contouring. */
export const logQ = (q: Factors) => (mu: number, tau: number) =>
  tau <= 0 ? -Infinity : -0.5 * q.lambda * (mu - q.mu) ** 2 + (q.a - 1) * Math.log(tau) - q.b * tau

export const logExact = (e: NormalGamma) => (mu: number, tau: number) =>
  tau <= 0 ? -Infinity : (e.a - 0.5) * Math.log(tau) - e.b * tau - 0.5 * e.lambda * tau * (mu - e.mu) ** 2

/**
 * The level set {log f = log f(mode) − k²/2} of a unimodal density, traced by bisection along rays from the mode.
 * Both densities here are unimodal with star-shaped level sets, so each ray crosses the level exactly once.
 */
export function contour(
  logf: (mu: number, tau: number) => number,
  mode: [number, number],
  scale: [number, number],
  k: number,
  rays = 96,
): { x: number[]; y: number[] } {
  const level = logf(...mode) - (k * k) / 2
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= rays; i++) {
    const t = (2 * Math.PI * i) / rays
    const d: [number, number] = [Math.cos(t) * scale[0], Math.sin(t) * scale[1]]
    const at = (r: number) => logf(mode[0] + r * d[0], mode[1] + r * d[1])
    let hi = 0.5
    for (let j = 0; j < 40 && at(hi) > level; j++) hi *= 2
    let lo = 0
    for (let j = 0; j < 40; j++) {
      const mid = (lo + hi) / 2
      if (at(mid) > level) lo = mid
      else hi = mid
    }
    x.push(mode[0] + lo * d[0])
    y.push(mode[1] + lo * d[1])
  }
  return { x, y }
}
