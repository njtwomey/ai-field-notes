/**
 * Bandit environments: Bernoulli and Gaussian multi-armed bandits, and a linear contextual bandit. Every round draws the
 * full vector of arm rewards from the round's own stream, so two policies run on the same stream see the same rewards
 * for the same arm in the same round (common random numbers): differences between their curves come from the
 * policies, not from luck.
 */

import type { BanditEnvironment } from 'aifn-applied/decisions/bandits'
import { normal, uniform } from 'aifn/foundation/random'
import { isTensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import type { EnvironmentInfo } from 'aifn/foundation/contracts'
import { definer } from 'aifn/foundation/registry'
import { int, oneOf, real, space } from 'aifn/foundation/space'

const read = (v: Tensor | readonly number[]) => (isTensor(v) ? toFlat(v) : [...v])

/** A Bernoulli bandit: arm a pays 1 with probability μ_a and 0 otherwise. */
export function bernoulliBandit(means: Tensor | readonly number[]): BanditEnvironment {
  const mu = Float64Array.from(read(means))
  if (mu.some((p) => !(p >= 0 && p <= 1))) throw new RangeError('bernoulliBandit: means must be probabilities')
  return {
    name: 'Bernoulli bandit',
    arms: mu.length,
    dim: 0,
    context: () => null,
    rewards: (s) => mu.map((p) => (uniform(s) < p ? 1 : 0)),
    means: () => mu,
    bounded: true,
  }
}

/** A Gaussian bandit: arm a pays N(μ_a, σ_a²). */
export function gaussianBandit(
  means: Tensor | readonly number[],
  sds: number | readonly number[] = 1,
): BanditEnvironment {
  const mu = Float64Array.from(read(means))
  const sd = typeof sds === 'number' ? new Float64Array(mu.length).fill(sds) : Float64Array.from(sds)
  return {
    name: 'Gaussian bandit',
    arms: mu.length,
    dim: 0,
    context: () => null,
    rewards: (s) => mu.map((m, a) => m + sd[a] * normal(s)),
    means: () => mu,
    bounded: false,
  }
}

/** Options for `linearBandit`. */
export interface LinearBanditOptions {
  /** The true parameter θ*. */
  theta: readonly number[]
  /** Arms per round. Default 5. */
  arms?: number
  /**
   * `fixed`: the same unit vectors every round, evenly spread in angle (2-D) or the coordinate axes and their negatives;
   * `random`: fresh arms each round, with random directions and lengths in [0.5, 1] (the contextual setting).
   */
  mode?: 'fixed' | 'random'
  /** Standard deviation of the Gaussian reward noise. Default 0.3. */
  noise?: number
}

/**
 * A linear bandit: arm x pays xᵀθ* + ε with ε ~ N(0, noise²) (Abbasi-Yadkori, Pál and Szepesvári, 2011, NeurIPS). The
 * arms are the round's context.
 */
export function linearBandit(options: LinearBanditOptions): BanditEnvironment {
  const theta = Float64Array.from(options.theta)
  const d = theta.length
  const k = options.arms ?? 5
  const noise = options.noise ?? 0.3
  const mode = options.mode ?? 'fixed'
  const fixed = Array.from({ length: k }, (_, i) => {
    const v = new Float64Array(d)
    if (d === 2) {
      const angle = (2 * Math.PI * i) / k + 0.3
      v[0] = Math.cos(angle)
      v[1] = Math.sin(angle)
    } else v[i % d] = i < d ? 1 : -1
    return v
  })
  const dot = (x: Float64Array) => x.reduce((s, v, j) => s + v * theta[j], 0)
  return {
    name: 'linear bandit',
    arms: k,
    dim: d,
    context: (s) =>
      mode === 'fixed'
        ? fixed
        : Array.from({ length: k }, () => {
            const v = Float64Array.from({ length: d }, () => normal(s))
            const norm = Math.hypot(...v)
            const length = 0.5 + 0.5 * uniform(s)
            return v.map((x) => (length * x) / norm)
          }),
    rewards: (s, ctx) => Float64Array.from(ctx!, (x) => dot(x) + noise * normal(s)),
    means: (ctx) => Float64Array.from(ctx!, dot),
    bounded: false,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const environment = definer<EnvironmentInfo>('environment', 'data/environments')

environment(
  {
    key: 'bernoulliBandit',
    name: 'Bernoulli bandit',
    summary: 'Arms paying 1 with their mean probability, else 0; the means are a required argument.',
    family: 'bandit',
    params: space({}),
    notes: ['multi-armed-bandit', 'thompson-sampling'],
  },
  bernoulliBandit,
)

environment(
  {
    key: 'gaussianBandit',
    name: 'Gaussian bandit',
    summary: 'Arms paying Gaussian rewards around their means; the means are a required argument.',
    family: 'bandit',
    params: space({}),
    notes: ['multi-armed-bandit', 'upper-confidence-bound-algorithm'],
  },
  gaussianBandit,
)

environment(
  {
    key: 'linearBandit',
    name: 'Linear bandit',
    summary: 'Arms are feature vectors whose mean reward is linear in an unknown θ*; θ* is a required option.',
    family: 'bandit',
    params: space({
      arms: int(2, 50, { default: 5 }),
      mode: oneOf(['fixed', 'random']),
      noise: real(0, 3, { default: 0.3 }),
    }),
    notes: ['contextual-bandit', 'linucb', 'linear-thompson-sampling'],
  },
  linearBandit,
)
