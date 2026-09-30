/**
 * Running a policy on an environment: `banditRun` is the traceable algorithm (one round per step), `regretCurves`
 * averages cumulative pseudo-regret over replicates with `replicate`, and `laiRobbinsBound` gives the asymptotic lower
 * bound for Bernoulli arms (Lattimore and Szepesvári, 2020, "Bandit Algorithms", ch. 4 and 16).
 */

import type { Status } from 'aifn/foundation/contracts'
import { replicate, stream, type Stream, child } from 'aifn/foundation/random'
import { fromData, type Tensor } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import type { BanditEnvironment } from './environment'
import { klBernoulli, type BanditPolicy, type Choice } from './policies'

/** One round's state of a bandit run. */
export interface BanditState<P = unknown> extends Status {
  /** Rounds played so far (0 initially). */
  t: number
  /** The arm pulled and reward received in the last round (−1 and 0 initially). */
  arm: number
  reward: number
  /** The best expected reward minus the pulled arm's in the last round (pseudo-regret). */
  regret: number
  cumulativeRegret: number
  cumulativeReward: number
  /** Pulls per arm. */
  counts: Tensor
  /** The per-arm scores the last choice maximised (indices, samples or estimates). */
  scores: Tensor
  /** The arms' expected rewards in the last round (they change each round in a contextual bandit). */
  means: Tensor
  /** The policy's internal state. */
  policy: P
  /** The last round's context (arms × dim), or null. */
  context: Tensor | null
}

/**
 * A policy on an environment as a traceable algorithm (no start). Round t draws the context and every arm's reward
 * from `child(ctx.stream, 'env')` (its `context` and `reward` substreams for a contextual bandit) and the policy's
 * randomness from `child(ctx.stream, 'agent')`, so different policies traced on the same root stream face identical
 * rewards (common random numbers).
 */
export function banditRun<P>(env: BanditEnvironment, policy: BanditPolicy<P>): Algorithm<void, BanditState<P>> {
  return {
    name: `${policy.name} on ${env.name}`,
    init() {
      const state = policy.init(env.arms, env.dim)
      return {
        t: 0,
        arm: -1,
        reward: 0,
        regret: 0,
        cumulativeRegret: 0,
        cumulativeReward: 0,
        counts: fromData(new Float64Array(env.arms)),
        scores: fromData(new Float64Array(env.arms)),
        means: fromData(new Float64Array(env.arms)),
        policy: state,
        context: null,
      }
    },
    step(s, ctx) {
      const t = s.t + 1
      const round = playRound(env, policy, s.policy, t, ctx.stream)
      const counts = (s.counts.data as Float64Array).slice()
      counts[round.choice.arm] += 1
      return {
        ...s,
        t,
        arm: round.choice.arm,
        reward: round.reward,
        regret: round.regret,
        cumulativeRegret: s.cumulativeRegret + round.regret,
        cumulativeReward: s.cumulativeReward + round.reward,
        counts: fromData(counts),
        scores: fromData(Float64Array.from(round.choice.scores)),
        means: fromData(Float64Array.from(round.means)),
        policy: round.next,
        context: round.context
          ? fromData(Float64Array.from(round.context.flatMap((x) => [...x])), [env.arms, env.dim])
          : null,
      }
    },
  }
}

/** Round t on its own stream (the runner's step stream): the world draws from `env`, the policy from `agent`. */
function playRound<P>(env: BanditEnvironment, policy: BanditPolicy<P>, state: P, t: number, s: Stream) {
  const world = child(s, 'env')
  // A context-free bandit draws its rewards from the round's stream directly (one substream fewer per round).
  const context = env.dim > 0 ? env.context(child(world, 'context')) : null
  const rewards = env.rewards(env.dim > 0 ? child(world, 'reward') : world, context)
  const means = env.means(context)
  const choice: Choice = policy.choose(state, t, child(s, 'agent'), context)
  const reward = rewards[choice.arm]
  let best = -Infinity
  for (const m of means) best = Math.max(best, m)
  return {
    choice,
    reward,
    regret: best - means[choice.arm],
    means,
    context,
    next: policy.update(state, choice, reward, context),
  }
}

/**
 * Cumulative pseudo-regret of one run at every round 1…horizon, and the final pull counts. Round t uses
 * `child(root, 'step', t − 1)`, the stream the runner gives `banditRun`'s step t, so a run here equals a trace of
 * `banditRun` on the same root.
 */
function runOnce<P>(env: BanditEnvironment, policy: BanditPolicy<P>, horizon: number, root: Stream) {
  let state = policy.init(env.arms, env.dim)
  const regret = new Float64Array(horizon)
  const counts = new Float64Array(env.arms)
  let cum = 0
  for (let t = 1; t <= horizon; t++) {
    const r = playRound(env, policy, state, t, child(root, 'step', t - 1))
    cum += r.regret
    regret[t - 1] = cum
    counts[r.choice.arm] += 1
    state = r.next
  }
  return { regret, counts }
}

/** Regret curves of several policies averaged over replicates. */
export interface RegretCurves {
  /** Recorded rounds (about `points` of them, always including the horizon). */
  t: Tensor
  /** Mean cumulative pseudo-regret, [policies, points]. */
  mean: Tensor
  /** Standard deviation across replicates, [policies, points]. */
  sd: Tensor
  /** The 10% and 90% quantiles across replicates, [policies, points]. */
  lower: Tensor
  upper: Tensor
  /** Mean pulls per arm at the horizon, [policies, arms]. */
  pulls: Tensor
  /** Every replicate's final cumulative regret, [policies, runs]. */
  final: Tensor
  names: string[]
}

/**
 * Run each policy `runs` times for `horizon` rounds and summarise the cumulative pseudo-regret Σ_t (μ* − μ_{A_t}).
 * Replicate k of every policy uses `child(s, k)` (through `replicate`), so all policies face the same rewards, and
 * raising `runs` reuses the replicates already computed when `cache` is kept.
 */
export function regretCurves(
  env: BanditEnvironment,
  policies: readonly BanditPolicy<unknown>[],
  options: { horizon: number; runs: number; stream?: Stream; points?: number },
): RegretCurves {
  const { horizon, runs, points = 200 } = options
  const s = options.stream ?? stream(0)
  const every = Math.max(1, Math.floor(horizon / points))
  const ts: number[] = []
  for (let t = every; t <= horizon; t += every) ts.push(t)
  if (ts[ts.length - 1] !== horizon) ts.push(horizon)
  const P = policies.length
  const mean = new Float64Array(P * ts.length)
  const sd = new Float64Array(P * ts.length)
  const lower = new Float64Array(P * ts.length)
  const upper = new Float64Array(P * ts.length)
  const pulls = new Float64Array(P * env.arms)
  const final = new Float64Array(P * runs)
  policies.forEach((policy, p) => {
    const results = replicate(runs, s, (r) => runOnce(env, policy, horizon, r), { cache: false })
    ts.forEach((t, k) => {
      const col = results.map((r) => r.regret[t - 1]).sort((a, b) => a - b)
      const m = col.reduce((a, b) => a + b, 0) / runs
      mean[p * ts.length + k] = m
      sd[p * ts.length + k] = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, runs - 1))
      lower[p * ts.length + k] = col[Math.floor(0.1 * (runs - 1))]
      upper[p * ts.length + k] = col[Math.ceil(0.9 * (runs - 1))]
    })
    results.forEach((r, k) => {
      r.counts.forEach((c, a) => (pulls[p * env.arms + a] += c / runs))
      final[p * runs + k] = r.regret[horizon - 1]
    })
  })
  const shape = [P, ts.length]
  return {
    t: fromData(Float64Array.from(ts)),
    mean: fromData(mean, shape),
    sd: fromData(sd, shape),
    lower: fromData(lower, shape),
    upper: fromData(upper, shape),
    pulls: fromData(pulls, [P, env.arms]),
    final: fromData(final, [P, runs]),
    names: policies.map((p) => p.name),
  }
}

/**
 * The Lai–Robbins lower bound for Bernoulli arms (Lai and Robbins, 1985, Adv. Appl. Math. 6): every consistent policy
 * has E[R_T] ≥ (1 + o(1)) ln T · Σ_{a: μ_a < μ*} (μ* − μ_a) / kl(μ_a, μ*). Returns the constant and the bound at each t.
 */
export function laiRobbinsBound(
  means: readonly number[],
  t: Tensor | readonly number[],
): { constant: number; bound: Tensor } {
  const best = Math.max(...means)
  const constant = means.reduce((acc, m) => (m < best ? acc + (best - m) / klBernoulli(m, best) : acc), 0)
  const ts = 'shape' in t ? Array.from(t.data) : t
  return { constant, bound: fromData(Float64Array.from(ts, (v) => constant * Math.log(v))) }
}
