/**
 * Bandit policies as pure objects: `init` builds the policy's state, `choose` picks an arm from it (and reports the
 * index or score of every arm), `update` returns the next state after a reward. States are plain data.
 */

import { cholesky } from 'aifn/linalg'
import { beta as betaDraw, normal, type Stream } from 'aifn/random'
import { fromData, toFlat } from 'aifn/tensor'

/** What a policy chose, with the per-arm quantity it maximised and, for randomised policies, its probabilities. */
export interface Choice {
  arm: number
  /** The index, sample or estimate of each arm the choice maximised. */
  scores: Float64Array
  /** P(arm) under the policy this round, when the policy randomises over arms (ε-greedy, EXP3). */
  probabilities?: Float64Array
}

/** A bandit policy. `P` is its state: plain data, never mutated. */
export interface BanditPolicy<P = unknown> {
  name: string
  init(arms: number, dim: number): P
  /** Choose in round t = 1, 2, …; randomness comes only from `s`. */
  choose(state: P, t: number, s: Stream, context: Float64Array[] | null): Choice
  update(state: P, choice: Choice, reward: number, context: Float64Array[] | null): P
}

/** Pull counts and reward sums per arm: the state of every context-free index policy. */
export interface ArmStatistics {
  counts: Float64Array
  sums: Float64Array
}

function argmaxRandom(scores: Float64Array, s: Stream): number {
  let best = -Infinity
  let ties: number[] = []
  scores.forEach((v, i) => {
    if (v > best) {
      best = v
      ties = [i]
    } else if (v === best) ties.push(i)
  })
  return ties.length === 1 ? ties[0] : ties[s.int(ties.length)]
}

const stats = (arms: number): ArmStatistics => ({ counts: new Float64Array(arms), sums: new Float64Array(arms) })

function record(st: ArmStatistics, arm: number, reward: number): ArmStatistics {
  const counts = st.counts.slice()
  const sums = st.sums.slice()
  counts[arm] += 1
  sums[arm] += reward
  return { ...st, counts, sums }
}

const means = (st: ArmStatistics) => st.counts.map((c, i) => (c > 0 ? st.sums[i] / c : 0))
const firstUnpulled = (st: ArmStatistics) => st.counts.findIndex((c) => c === 0)

/** An index policy on arm statistics: pulls every arm once, then the arm with the largest index. */
function indexPolicy(name: string, index: (st: ArmStatistics, t: number) => Float64Array): BanditPolicy<ArmStatistics> {
  return {
    name,
    init: (arms) => stats(arms),
    choose(st, t, s) {
      const first = firstUnpulled(st)
      const scores = first >= 0 ? st.counts.map((c) => (c === 0 ? Infinity : -Infinity)) : index(st, t)
      return { arm: first >= 0 ? first : argmaxRandom(scores, s), scores }
    },
    update: (st, c, r) => record(st, c.arm, r),
  }
}

/** Round robin over the arms (an A/B/n test): arm (t − 1) mod K. */
export function uniformPolicy(): BanditPolicy<ArmStatistics> {
  return {
    name: 'uniform',
    init: (arms) => stats(arms),
    choose: (st, t) => ({ arm: (t - 1) % st.counts.length, scores: means(st) }),
    update: (st, c, r) => record(st, c.arm, r),
  }
}

/** Explore then commit: m pulls of every arm in turn, then the best empirical mean for ever. */
export function exploreThenCommit({ m = 50 }: { m?: number } = {}): BanditPolicy<ArmStatistics> {
  return {
    name: `explore-then-commit (m = ${m})`,
    init: (arms) => stats(arms),
    choose(st, t, s) {
      const k = st.counts.length
      const scores = means(st)
      return { arm: t <= m * k ? (t - 1) % k : argmaxRandom(scores, s), scores }
    },
    update: (st, c, r) => record(st, c.arm, r),
  }
}

/**
 * ε-greedy: with probability ε a uniformly random arm, otherwise the best empirical mean. With `decay: c` the
 * exploration rate is min(1, cK/t) (Auer, Cesa-Bianchi and Fischer, 2002, Machine Learning 47, §3).
 */
export function epsilonGreedy({
  epsilon = 0.1,
  decay,
}: { epsilon?: number; decay?: number } = {}): BanditPolicy<ArmStatistics> {
  return {
    name: decay === undefined ? `ε-greedy (ε = ${epsilon})` : `decaying ε-greedy (c = ${decay})`,
    init: (arms) => stats(arms),
    choose(st, t, s) {
      const k = st.counts.length
      const first = firstUnpulled(st)
      const scores = means(st)
      if (first >= 0) return { arm: first, scores }
      const eps = decay === undefined ? epsilon : Math.min(1, (decay * k) / t)
      const greedy = argmaxRandom(scores, s)
      const probabilities = new Float64Array(k).fill(eps / k)
      probabilities[greedy] += 1 - eps
      return { arm: s.uniform() < eps ? s.int(k) : greedy, scores, probabilities }
    },
    update: (st, c, r) => record(st, c.arm, r),
  }
}

/**
 * UCB1 (Auer, Cesa-Bianchi and Fischer, 2002): the index μ̂_a + √(c ln t / n_a), c = 2 by default, for rewards in
 * [0, 1].
 */
export function ucb1({ c = 2 }: { c?: number } = {}): BanditPolicy<ArmStatistics> {
  return indexPolicy(`UCB1`, (st, t) => st.counts.map((n, i) => st.sums[i] / n + Math.sqrt((c * Math.log(t)) / n)))
}

/** KL(Bernoulli(p) ‖ Bernoulli(q)) in nats, with 0 log 0 = 0; infinite when q is 0 or 1 and p is not. */
export function klBernoulli(p: number, q: number): number {
  const term = (x: number, y: number) => (x === 0 ? 0 : y === 0 ? Infinity : x * Math.log(x / y))
  return term(p, q) + term(1 - p, 1 - q)
}

/**
 * The KL-UCB index: the largest q ∈ [p, 1] with n · kl(p, q) ≤ level (Garivier and Cappé, 2011, COLT), by bisection to
 * 1e-10.
 */
export function klUcbIndex(p: number, n: number, level: number): number {
  let lo = p
  let hi = 1
  if (n * klBernoulli(p, hi) <= level) return 1
  for (let it = 0; it < 60 && hi - lo > 1e-10; it++) {
    const mid = (lo + hi) / 2
    if (n * klBernoulli(p, mid) <= level) lo = mid
    else hi = mid
  }
  return lo
}

/** KL-UCB for Bernoulli rewards: the index `klUcbIndex(μ̂_a, n_a, ln t + c ln ln t)`, c = 0 by default. */
export function klUcb({ c = 0 }: { c?: number } = {}): BanditPolicy<ArmStatistics> {
  return indexPolicy('KL-UCB', (st, t) => {
    const level = Math.log(t) + (c > 0 && t > 1 ? c * Math.log(Math.log(t)) : 0)
    return st.counts.map((n, i) => klUcbIndex(st.sums[i] / n, n, level))
  })
}

/**
 * Thompson sampling for Bernoulli rewards (Thompson, 1933, Biometrika 25; Agrawal and Goyal, 2012, COLT): draw
 * θ_a ~ Beta(α + s_a, β + n_a − s_a) and pull the largest. Rewards outside {0, 1} are used as fractional successes.
 */
export function thompsonBernoulli({
  alpha = 1,
  beta = 1,
}: { alpha?: number; beta?: number } = {}): BanditPolicy<ArmStatistics> {
  return {
    name: 'Thompson sampling',
    init: (arms) => stats(arms),
    choose(st, _t, s) {
      const scores = st.counts.map((n, i) => betaDraw(s, alpha + st.sums[i], beta + n - st.sums[i]))
      return { arm: argmaxRandom(scores, s), scores }
    },
    update: (st, c, r) => record(st, c.arm, r),
  }
}

/**
 * Thompson sampling for Gaussian rewards with known noise sd σ and a N(μ₀, τ₀²) prior on each mean: the posterior of
 * arm a is N(m_a, v_a) with 1/v_a = 1/τ₀² + n_a/σ², m_a = v_a (μ₀/τ₀² + s_a/σ²).
 */
export function thompsonGaussian({ priorMean = 0, priorSd = 1, noiseSd = 1 } = {}): BanditPolicy<ArmStatistics> {
  return {
    name: 'Gaussian Thompson sampling',
    init: (arms) => stats(arms),
    choose(st, _t, s) {
      const scores = st.counts.map((n, i) => {
        const precision = 1 / priorSd ** 2 + n / noiseSd ** 2
        const mean = (priorMean / priorSd ** 2 + st.sums[i] / noiseSd ** 2) / precision
        return mean + normal(s) / Math.sqrt(precision)
      })
      return { arm: argmaxRandom(scores, s), scores }
    },
    update: (st, c, r) => record(st, c.arm, r),
  }
}

/** The state of EXP3: log-weights per arm, with counts and sums for display. */
export interface Exp3State extends ArmStatistics {
  logWeights: Float64Array
}

/**
 * EXP3 for adversarial rewards in [0, 1] (Auer, Cesa-Bianchi, Freund and Schapire, 2002, SIAM J. Comput. 32(1)):
 * p_a = (1 − γ) w_a / Σ w + γ/K, and the pulled arm's weight grows by exp(γ r̂ / K) with the importance-weighted reward
 * r̂ = r / p_a.
 */
export function exp3({ gamma = 0.1 }: { gamma?: number } = {}): BanditPolicy<Exp3State> {
  const probs = (lw: Float64Array) => {
    const k = lw.length
    const mx = Math.max(...lw)
    const w = lw.map((v) => Math.exp(v - mx))
    const total = w.reduce((a, b) => a + b, 0)
    return w.map((v) => (1 - gamma) * (v / total) + gamma / k)
  }
  return {
    name: `EXP3 (γ = ${gamma})`,
    init: (arms) => ({ ...stats(arms), logWeights: new Float64Array(arms) }),
    choose(st, _t, s) {
      const p = probs(st.logWeights)
      let u = s.uniform()
      let arm = p.length - 1
      for (let a = 0; a < p.length; a++) {
        if (u < p[a]) {
          arm = a
          break
        }
        u -= p[a]
      }
      return { arm, scores: p, probabilities: p }
    },
    update(st, c, r) {
      const k = st.logWeights.length
      const p = c.probabilities![c.arm]
      const logWeights = st.logWeights.slice()
      logWeights[c.arm] += (gamma * (r / p)) / k
      return { ...record(st, c.arm, r), logWeights }
    },
  }
}

/** The ridge statistics of a linear policy: V = λI + Σ x xᵀ (and its inverse) and b = Σ r x. */
export interface RidgeState extends ArmStatistics {
  /** V⁻¹, d × d row-major, kept by Sherman–Morrison updates. */
  vInverse: Float64Array
  b: Float64Array
  /** θ̂ = V⁻¹ b. */
  theta: Float64Array
  dim: number
}

function ridgeInit(arms: number, d: number, lambda: number): RidgeState {
  const vInverse = new Float64Array(d * d)
  for (let i = 0; i < d; i++) vInverse[i * d + i] = 1 / lambda
  return { ...stats(arms), vInverse, b: new Float64Array(d), theta: new Float64Array(d), dim: d }
}

function ridgeUpdate(st: RidgeState, x: Float64Array, r: number, arm: number): RidgeState {
  const d = st.dim
  const A = st.vInverse
  // Sherman–Morrison: (V + x xᵀ)⁻¹ = V⁻¹ − V⁻¹x xᵀV⁻¹ / (1 + xᵀV⁻¹x).
  const Ax = new Float64Array(d)
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Ax[i] += A[i * d + j] * x[j]
  const denom = 1 + x.reduce((s, v, i) => s + v * Ax[i], 0)
  const next = A.slice()
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) next[i * d + j] -= (Ax[i] * Ax[j]) / denom
  const b = st.b.map((v, i) => v + r * x[i])
  const theta = new Float64Array(d)
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) theta[i] += next[i * d + j] * b[j]
  return { ...record(st, arm, r), vInverse: next, b, theta, dim: d }
}

function width(A: Float64Array, x: Float64Array, d: number): number {
  let s = 0
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) s += x[i] * A[i * d + j] * x[j]
  return Math.sqrt(Math.max(0, s))
}

/**
 * LinUCB (Li, Chu, Langford and Schapire, 2010, WWW; the disjoint-free form of Abbasi-Yadkori et al., 2011): pull the
 * arm maximising xᵀθ̂ + α ‖x‖_{V⁻¹}, with the ridge estimate θ̂ = V⁻¹ b. α = 0 is the greedy ridge policy.
 */
export function linUcb({ alpha = 1, lambda = 1 }: { alpha?: number; lambda?: number } = {}): BanditPolicy<RidgeState> {
  return {
    name: alpha === 0 ? 'greedy ridge' : `LinUCB (α = ${alpha})`,
    init: (arms, d) => ridgeInit(arms, d, lambda),
    choose(st, _t, s, ctx) {
      if (!ctx) throw new Error('linUcb needs a contextual (linear) bandit')
      const scores = Float64Array.from(
        ctx,
        (x) => x.reduce((a, v, i) => a + v * st.theta[i], 0) + alpha * width(st.vInverse, x, st.dim),
      )
      return { arm: argmaxRandom(scores, s), scores }
    },
    update: (st, c, r, ctx) => ridgeUpdate(st, ctx![c.arm], r, c.arm),
  }
}

/**
 * Linear Thompson sampling (Agrawal and Goyal, 2013, ICML): draw θ̃ ~ N(θ̂, v² V⁻¹) and pull the arm maximising xᵀθ̃.
 */
export function linearThompson({
  v = 0.5,
  lambda = 1,
}: { v?: number; lambda?: number } = {}): BanditPolicy<RidgeState> {
  return {
    name: `linear Thompson sampling (v = ${v})`,
    init: (arms, d) => ridgeInit(arms, d, lambda),
    choose(st, _t, s, ctx) {
      if (!ctx) throw new Error('linearThompson needs a contextual (linear) bandit')
      const d = st.dim
      const L = toFlat(
        cholesky(
          fromData(
            st.vInverse.map((x) => x * v * v),
            [d, d],
          ),
        ).L,
      )
      const z = Float64Array.from({ length: d }, () => normal(s))
      const draw = st.theta.map((m, i) => {
        let acc = m
        for (let j = 0; j <= i; j++) acc += L[i * d + j] * z[j]
        return acc
      })
      const scores = Float64Array.from(ctx, (x) => x.reduce((a, val, i) => a + val * draw[i], 0))
      return { arm: argmaxRandom(scores, s), scores }
    },
    update: (st, c, r, ctx) => ridgeUpdate(st, ctx![c.arm], r, c.arm),
  }
}
