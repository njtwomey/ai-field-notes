/**
 * Noise schedules. A discrete schedule (Ho et al., 2020) lists β₁ … β_T and the derived αₜ = 1 − βₜ,
 * ᾱₜ = α₁⋯αₜ and signal-to-noise ratios SNR(t) = ᾱₜ/(1 − ᾱₜ), so that x_t | x₀ ~ N(√ᾱₜ x₀, (1 − ᾱₜ)I). A continuous
 * forward SDE dx = f(t)x dt + g(t) dw on t ∈ [0, 1] (Song et al., 2021, "Score-based generative modeling through
 * stochastic differential equations", §3.4 and appendix B) has the Gaussian marginal x_t | x₀ ~ N(m(t)x₀, s(t)²I).
 */

import { fromData, type Tensor } from 'aifn/foundation/tensor'

/** A discrete noise schedule of T steps. Entry t − 1 of each tensor belongs to step t = 1 … T. */
export type NoiseSchedule = {
  readonly kind: 'linear' | 'cosine' | 'custom'
  /** The number of steps T. */
  readonly steps: number
  /** β₁ … β_T, the variance added at each step. */
  readonly betas: Tensor
  /** αₜ = 1 − βₜ. */
  readonly alphas: Tensor
  /** ᾱₜ = Π_{s ≤ t} α_s, the fraction of signal variance left after t steps. */
  readonly alphaBars: Tensor
  /** SNR(t) = ᾱₜ/(1 − ᾱₜ). */
  readonly snr: Tensor
  /** Steps whose β was capped (the cosine schedule caps β at `maxBeta`); empty when none was. */
  readonly capped: readonly number[]
}

/** A schedule from its β's (each in (0, 1)). */
export function scheduleFromBetas(
  betas: ArrayLike<number>,
  kind: NoiseSchedule['kind'] = 'custom',
  capped: readonly number[] = [],
): NoiseSchedule {
  const T = betas.length
  const b = Float64Array.from(betas)
  const a = new Float64Array(T)
  const ab = new Float64Array(T)
  const snr = new Float64Array(T)
  let prod = 1
  for (let i = 0; i < T; i++) {
    if (!(b[i] > 0 && b[i] < 1)) throw new RangeError(`schedule: β_${i + 1} = ${b[i]} is not in (0, 1)`)
    a[i] = 1 - b[i]
    prod *= a[i]
    ab[i] = prod
    snr[i] = prod / (1 - prod)
  }
  const v = (d: Float64Array) => fromData(d, [T])
  return { kind, steps: T, betas: v(b), alphas: v(a), alphaBars: v(ab), snr: v(snr), capped }
}

/**
 * The linear schedule of Ho et al. (2020, §4): βₜ rises linearly from `betaStart` = 1e-4 to `betaEnd` = 0.02 over T =
 * 1000 steps (the defaults), leaving ᾱ_T ≈ 4·10⁻⁵.
 */
export function linearSchedule(
  steps = 1000,
  { betaStart = 1e-4, betaEnd = 0.02 }: { betaStart?: number; betaEnd?: number } = {},
): NoiseSchedule {
  const betas = Float64Array.from({ length: steps }, (_, i) =>
    steps === 1 ? betaStart : betaStart + ((betaEnd - betaStart) * i) / (steps - 1),
  )
  return scheduleFromBetas(betas, 'linear')
}

/**
 * The cosine schedule of Nichol & Dhariwal (2021, eq. 17): ᾱₜ = f(t)/f(0) with f(t) = cos²(((t/T + s)/(1 + s))·π/2),
 * so ᾱ falls slowly at both ends; βₜ = 1 − ᾱₜ/ᾱₜ₋₁, capped at `maxBeta` = 0.999 as in the paper (the steps capped are
 * reported in `capped`; with the defaults only the last few are).
 */
export function cosineSchedule(
  steps = 1000,
  { offset = 0.008, maxBeta = 0.999 }: { offset?: number; maxBeta?: number } = {},
): NoiseSchedule {
  const f = (t: number) => Math.cos((((t / steps + offset) / (1 + offset)) * Math.PI) / 2) ** 2
  const capped: number[] = []
  const betas = Float64Array.from({ length: steps }, (_, i) => {
    const beta = 1 - f(i + 1) / f(i)
    if (beta > maxBeta) {
      capped.push(i + 1)
      return maxBeta
    }
    return beta
  })
  return scheduleFromBetas(betas, 'cosine', capped)
}

/** ᾱ at step t (t = 0 gives 1: no noise). */
export function alphaBarAt(schedule: NoiseSchedule, t: number): number {
  if (t === 0) return 1
  if (!Number.isInteger(t) || t < 0 || t > schedule.steps)
    throw new RangeError(`step ${t} is not in 0 … ${schedule.steps}`)
  return schedule.alphaBars.data[t - 1]
}

/** β at step t = 1 … T. */
export function betaAt(schedule: NoiseSchedule, t: number): number {
  if (!Number.isInteger(t) || t < 1 || t > schedule.steps)
    throw new RangeError(`step ${t} is not in 1 … ${schedule.steps}`)
  return schedule.betas.data[t - 1]
}

// ── Continuous-time forward SDEs ─────────────────────────────────────────────────────────────────────────────────────

/**
 * A linear forward SDE dx = f(t)·x dt + g(t) dw on t ∈ [0, 1], with its Gaussian marginal x_t | x₀ ~ N(m(t)x₀, s(t)²I).
 */
export type ForwardSde = {
  readonly kind: 'vp' | 'subVp' | 've'
  /** The drift coefficient f(t) (the drift is f(t)·x). */
  drift(t: number): number
  /** The diffusion coefficient g(t). */
  diffusion(t: number): number
  /** The mean scale m(t) of the marginal. */
  meanScale(t: number): number
  /** The standard deviation s(t) of the marginal. */
  std(t: number): number
  /** The standard deviation of the prior at t = 1 that sampling starts from (N(0, σ²I)). */
  priorStd: number
}

/**
 * The variance-preserving SDE (Song et al., 2021, eq. 11), the continuous limit of DDPM: β(t) = β_min + t(β_max −
 * β_min), f = −β(t)/2, g = √β(t), m(t) = exp(−½∫₀ᵗβ), s(t) = √(1 − m(t)²). Defaults β_min = 0.1, β_max = 20.
 */
export function vpSde({ betaMin = 0.1, betaMax = 20 }: { betaMin?: number; betaMax?: number } = {}): ForwardSde {
  const beta = (t: number) => betaMin + t * (betaMax - betaMin)
  const integral = (t: number) => betaMin * t + 0.5 * (betaMax - betaMin) * t * t
  const meanScale = (t: number) => Math.exp(-0.5 * integral(t))
  return {
    kind: 'vp',
    drift: (t) => -0.5 * beta(t),
    diffusion: (t) => Math.sqrt(beta(t)),
    meanScale,
    std: (t) => Math.sqrt(-Math.expm1(-integral(t))),
    priorStd: 1,
  }
}

/**
 * The sub-VP SDE (Song et al., 2021, eq. 12): the VP drift with g(t)² = β(t)(1 − e^{−2∫₀ᵗβ}), whose marginal variance
 * (1 − m(t)²)² is below the VP one at every time.
 */
export function subVpSde({ betaMin = 0.1, betaMax = 20 }: { betaMin?: number; betaMax?: number } = {}): ForwardSde {
  const vp = vpSde({ betaMin, betaMax })
  const beta = (t: number) => betaMin + t * (betaMax - betaMin)
  const integral = (t: number) => betaMin * t + 0.5 * (betaMax - betaMin) * t * t
  return {
    ...vp,
    kind: 'subVp',
    diffusion: (t) => Math.sqrt(beta(t) * -Math.expm1(-2 * integral(t))),
    std: (t) => -Math.expm1(-integral(t)),
  }
}

/**
 * The variance-exploding SDE (Song et al., 2021, eq. 9): σ(t) = σ_min(σ_max/σ_min)ᵗ, f = 0,
 * g = σ(t)√(2 log(σ_max/σ_min)), m = 1, s(t) = σ(t). Defaults σ_min = 0.01, σ_max = 50.
 */
export function veSde({ sigmaMin = 0.01, sigmaMax = 50 }: { sigmaMin?: number; sigmaMax?: number } = {}): ForwardSde {
  const sigma = (t: number) => sigmaMin * (sigmaMax / sigmaMin) ** t
  const k = Math.sqrt(2 * Math.log(sigmaMax / sigmaMin))
  return {
    kind: 've',
    drift: () => 0,
    diffusion: (t) => sigma(t) * k,
    meanScale: () => 1,
    std: sigma,
    priorStd: sigmaMax,
  }
}
