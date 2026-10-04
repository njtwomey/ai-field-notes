/**
 * Stochastic Bernoulli bandits for the figures in this category: samplers, the classical index policies and a seeded
 * simulator that averages cumulative pseudo-regret over independent runs.
 *
 * The random source is passed in (`seededRand(seed)` from './rand' in the widgets), so this module has no imports and runs
 * under plain Node for checking.
 */

export type Rand = { uniform: () => number; normal: () => number }

/** Gamma(shape, 1) by Marsaglia and Tsang's squeeze method; shapes below 1 use the boost Gamma(a) = Gamma(a+1) U^(1/a). */
export function sampleGamma(shape: number, r: Rand): number {
  if (shape < 1) return sampleGamma(shape + 1, r) * Math.pow(Math.max(r.uniform(), 1e-300), 1 / shape)
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    let x: number
    let v: number
    do {
      x = r.normal()
      v = 1 + c * x
    } while (v <= 0)
    v = v * v * v
    const u = r.uniform()
    if (u < 1 - 0.0331 * x ** 4) return d * v
    if (Math.log(Math.max(u, 1e-300)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

export function sampleBeta(a: number, b: number, r: Rand): number {
  const x = sampleGamma(a, r)
  const y = sampleGamma(b, r)
  return x / (x + y)
}

/** KL divergence between Bernoulli(p) and Bernoulli(q), in nats. */
export function klBernoulli(p: number, q: number): number {
  const eps = 1e-12
  const pp = Math.min(Math.max(p, eps), 1 - eps)
  const qq = Math.min(Math.max(q, eps), 1 - eps)
  return pp * Math.log(pp / qq) + (1 - pp) * Math.log((1 - pp) / (1 - qq))
}

/**
 * The largest q in [p, 1] with n kl(p, q) at most `level`: the KL-UCB index. kl(p, .) is convex and increasing on
 * [p, 1], so Newton's method started to the right of the root, at Pinsker's bound p + sqrt(level / 2n), decreases
 * monotonically to it.
 */
export function klUpper(p: number, n: number, level: number): number {
  const top = 1 - 1e-12
  let q = Math.min(top, p + Math.sqrt(level / (2 * n)))
  if (n * klBernoulli(p, q) <= level) return q
  for (let i = 0; i < 30; i++) {
    const f = n * klBernoulli(p, q) - level
    if (f < 1e-9) break
    const slope = (n * (q - p)) / (q * (1 - q))
    q = Math.max(p, q - f / slope)
  }
  return q
}

export type AlgorithmId = 'uniform' | 'etc' | 'egreedy' | 'edecay' | 'ucb1' | 'klucb' | 'ts'

/** Each algorithm keeps one colour slot in every figure of the category. */
export const ALGORITHMS: { id: AlgorithmId; label: string; slot: number }[] = [
  { id: 'uniform', label: 'uniform (A/B)', slot: 7 },
  { id: 'etc', label: 'explore-then-commit', slot: 6 },
  { id: 'egreedy', label: 'ε-greedy', slot: 3 },
  { id: 'edecay', label: 'decaying ε-greedy', slot: 4 },
  { id: 'ucb1', label: 'UCB1', slot: 0 },
  { id: 'klucb', label: 'KL-UCB', slot: 2 },
  { id: 'ts', label: 'Thompson sampling', slot: 1 },
]

export type Tuning = {
  /** Exploration probability of constant ε-greedy. */
  epsilon: number
  /** Pulls per arm in the exploration phase of explore-then-commit. */
  etcM: number
  /** Decaying ε-greedy explores with probability min(1, c K / t). */
  decayC: number
}

export const DEFAULT_TUNING: Tuning = { epsilon: 0.1, etcM: 50, decayC: 5 }

export type Policy = {
  /** The arm to pull in round t = 1, 2, ... */
  choose: (t: number) => number
  update: (arm: number, reward: number) => void
  /** Pull counts and reward sums, read by figures that display the policy's state. */
  n: number[]
  s: number[]
}

const argmax = (xs: number[]) => {
  let best = 0
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i
  return best
}

export function makePolicy(id: AlgorithmId, k: number, tuning: Tuning, r: Rand): Policy {
  const n = new Array<number>(k).fill(0)
  const s = new Array<number>(k).fill(0)
  const means = () => n.map((c, i) => (c > 0 ? s[i] / c : 0))
  // Index policies pull every arm once first, so every empirical mean is defined.
  const unpulled = () => n.findIndex((c) => c === 0)
  const choose = (t: number): number => {
    switch (id) {
      case 'uniform':
        return (t - 1) % k
      case 'etc': {
        if (t <= tuning.etcM * k) return (t - 1) % k
        return argmax(means())
      }
      case 'egreedy':
      case 'edecay': {
        const first = unpulled()
        if (first >= 0) return first
        const eps = id === 'egreedy' ? tuning.epsilon : Math.min(1, (tuning.decayC * k) / t)
        if (r.uniform() < eps) return Math.floor(r.uniform() * k)
        return argmax(means())
      }
      case 'ucb1': {
        const first = unpulled()
        if (first >= 0) return first
        const logT = Math.log(t)
        return argmax(n.map((c, i) => s[i] / c + Math.sqrt((2 * logT) / c)))
      }
      case 'klucb': {
        const first = unpulled()
        if (first >= 0) return first
        const logT = Math.log(t)
        // Pinsker's inequality, kl(p, q) >= 2 (q - p)^2, caps each KL index cheaply; arms whose cap is below the best
        // index found so far cannot win, so their bisection is skipped.
        const caps = n.map((c, i) => s[i] / c + Math.sqrt(logT / (2 * c)))
        const order = caps.map((_, i) => i).sort((a, b) => caps[b] - caps[a])
        let best = order[0]
        let bestIndex = -1
        for (const i of order) {
          if (caps[i] <= bestIndex) break
          const index = klUpper(s[i] / n[i], n[i], logT)
          if (index > bestIndex) [best, bestIndex] = [i, index]
        }
        return best
      }
      case 'ts':
        return argmax(n.map((c, i) => sampleBeta(1 + s[i], 1 + c - s[i], r)))
    }
  }
  const update = (arm: number, reward: number) => {
    n[arm] += 1
    s[arm] += reward
  }
  return { choose, update, n, s }
}

export type SimulationResult = {
  /** Rounds at which the curve is recorded. */
  t: number[]
  /** Mean cumulative pseudo-regret at each recorded round, over the runs. */
  regret: number[]
  /** Mean number of pulls of each arm after the last round. */
  pulls: number[]
}

/** Recorded rounds: about `points` of them, always including the last. */
export function checkpoints(horizon: number, points = 200): number[] {
  const every = Math.max(1, Math.floor(horizon / points))
  const ts: number[] = []
  for (let t = every; t <= horizon; t += every) ts.push(t)
  if (ts[ts.length - 1] !== horizon) ts.push(horizon)
  return ts
}

/**
 * Runs one algorithm on Bernoulli arms with the given means, `runs` times with independent seeds, and averages the
 * cumulative pseudo-regret sum_t (mu* - mu_{A_t}). The reward draws of run j are the same for every algorithm, so
 * differences between curves come from the algorithms, not from luck.
 */
export function simulate(
  id: AlgorithmId,
  means: number[],
  horizon: number,
  runs: number,
  seed: number,
  tuning: Tuning,
  makeRand: (seed: number) => Rand,
): SimulationResult {
  const k = means.length
  const best = Math.max(...means)
  const gaps = means.map((m) => best - m)
  const ts = checkpoints(horizon)
  const regret = new Array<number>(ts.length).fill(0)
  const pulls = new Array<number>(k).fill(0)
  for (let run = 0; run < runs; run++) {
    const env = makeRand(seed * 7919 + run)
    const agent = makeRand(seed * 104729 + run + 1)
    // Reward table: each arm's j-th pull returns the j-th entry of its own stream, independent of the algorithm.
    const policy = makePolicy(id, k, tuning, agent)
    let cum = 0
    let next = 0
    const streams = means.map(() => [] as number[])
    for (let t = 1; t <= horizon; t++) {
      const a = policy.choose(t)
      const stream = streams[a]
      while (stream.length <= policy.n[a]) {
        for (let i = 0; i < k; i++) streams[i].push(env.uniform() < means[i] ? 1 : 0)
      }
      policy.update(a, stream[policy.n[a]])
      cum += gaps[a]
      if (t === ts[next]) regret[next++] += cum
    }
    policy.n.forEach((c, i) => (pulls[i] += c))
  }
  return { t: ts, regret: regret.map((v) => v / runs), pulls: pulls.map((v) => v / runs) }
}

/**
 * The Lai-Robbins asymptotic lower bound for Bernoulli arms: every consistent algorithm has
 * E[R_T] >= (1 + o(1)) ln T * sum over suboptimal arms of gap / kl(mu_i, mu*).
 */
export function laiRobbinsConstant(means: number[]): number {
  const best = Math.max(...means)
  return means.reduce((acc, m) => (m < best ? acc + (best - m) / klBernoulli(m, best) : acc), 0)
}
