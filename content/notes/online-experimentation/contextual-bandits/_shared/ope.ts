/**
 * A synthetic logged-bandit problem for the off-policy figures: a scalar context x ~ Unif(0, 1), three Bernoulli arms
 * whose means are linear in x, a logging policy that favours a poor arm with probability 1 − ε and explores uniformly
 * otherwise, and the four standard value estimators.
 *
 * The random source is passed in (`rng(seed)` from '@/lib/math' in the widgets), so this module has no imports.
 */

export type Rand = { uniform: () => number; normal: () => number }

export const K = 3

/** Mean reward of arm a at context x. The best policy plays arm 0 below x = 0.4375 and arm 1 above it. */
export function mu(a: number, x: number): number {
  if (a === 0) return 0.7 - 0.5 * x
  if (a === 1) return 0.35 + 0.3 * x
  return 0.15 + 0.4 * x
}

/** The logging policy's favourite arm: the worst arm for low x, and arm 0, which is poor for high x, above 0.5. */
export const favourite = (x: number) => (x < 0.5 ? 2 : 0)

/** Probability that the logging policy with exploration rate ε plays arm a at context x. */
export const loggingProb = (a: number, x: number, eps: number) => (1 - eps) * (a === favourite(x) ? 1 : 0) + eps / K

export type Row = { x: number; a: number; r: number; p: number }

export function logData(n: number, eps: number, r: Rand): Row[] {
  return Array.from({ length: n }, () => {
    const x = r.uniform()
    const a = r.uniform() < 1 - eps ? favourite(x) : Math.floor(r.uniform() * K)
    return { x, a, r: r.uniform() < mu(a, x) ? 1 : 0, p: loggingProb(a, x, eps) }
  })
}

export type RewardModel = 'constant' | 'linear'

/**
 * Per-arm least-squares fit of the reward on the logged rows: a constant (which ignores x and is misspecified) or a
 * line in x (which is correct here). Arms with too few rows fall back to the overall mean reward.
 */
export function fitModel(rows: Row[], model: RewardModel): (a: number, x: number) => number {
  const overall = rows.reduce((acc, row) => acc + row.r, 0) / Math.max(rows.length, 1)
  const coefs = Array.from({ length: K }, (_, a) => {
    let n = 0
    let sx = 0
    let sy = 0
    let sxx = 0
    let sxy = 0
    for (const row of rows) {
      if (row.a !== a) continue
      n++
      sx += row.x
      sy += row.r
      sxx += row.x * row.x
      sxy += row.x * row.r
    }
    if (n === 0) return [overall, 0]
    if (model === 'constant' || n < 3) return [sy / n, 0]
    const slope = (n * sxy - sx * sy) / Math.max(n * sxx - sx * sx, 1e-12)
    return [(sy - slope * sx) / n, slope]
  })
  return (a, x) => coefs[a][0] + coefs[a][1] * x
}

export type Estimates = { ips: number; snips: number; dm: number; dr: number; ess: number }

/** The four estimators of V(π) for a deterministic target policy π, and the effective sample size (Σw)² / Σw². */
export function estimate(rows: Row[], target: (x: number) => number, model: RewardModel): Estimates {
  const rhat = fitModel(rows, model)
  let sw = 0
  let sww = 0
  let swr = 0
  let dm = 0
  let correction = 0
  for (const row of rows) {
    const w = row.a === target(row.x) ? 1 / row.p : 0
    sw += w
    sww += w * w
    swr += w * row.r
    dm += rhat(target(row.x), row.x)
    correction += w * (row.r - rhat(row.a, row.x))
  }
  const n = rows.length
  return {
    ips: swr / n,
    snips: sw > 0 ? swr / sw : 0,
    dm: dm / n,
    dr: (dm + correction) / n,
    ess: sww > 0 ? (sw * sw) / sww : 0,
  }
}

/** The best policy for this problem, and its true value E[max_a μ(x, a)] = 0.5765625. */
export const bestPolicy = (x: number) => (x < 0.4375 ? 0 : 1)
export const BEST_VALUE = 0.5765625

/** True value of the threshold policy "arm 0 below θ, arm 1 above", in closed form. */
export function thresholdValue(theta: number): number {
  const t = Math.min(1, Math.max(0, theta))
  return 0.7 * t - 0.25 * t * t + 0.35 * (1 - t) + 0.15 * (1 - t * t)
}
