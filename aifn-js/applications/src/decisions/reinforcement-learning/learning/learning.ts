/**
 * Learning from sampled experience (Sutton and Barto, 2018, chs. 5–7 and 13): Monte Carlo control, TD(0) prediction,
 * SARSA, Q-learning and expected SARSA, n-step SARSA, and REINFORCE with a tabular softmax policy. Each is a traceable
 * algorithm whose step is one episode from the start state; the state holds the value tables, the greedy policy and the
 * episode's path and return. Episode e draws only from the runner's step stream (`ctx.stream`), so the state is plain
 * data and a run resumed from any stored state reproduces the rest.
 */

import type { Status } from 'aifn/foundation/contracts'
import { type Stream, integers, uniform } from 'aifn/foundation/random'
import { fromData, type Tensor } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import { isActive, type Outcome, type TabularMdp } from '../mdp'
import { greedyActions, policyMatrix, type PolicyInput } from '../mdp'

/** Sample an outcome of action a in state s from the stream r (one uniform draw). */
export function sampleOutcome(r: Stream, mdp: TabularMdp, s: number, a: number): Outcome {
  const outs = mdp.outcomes[s * mdp.actions + a]
  let u = uniform(r)
  for (const o of outs) {
    if (u < o.p) return o
    u -= o.p
  }
  return outs[outs.length - 1]
}

/** ε-greedy action in s under Q, breaking ties uniformly at random (so an untrained agent does not always go one way). */
function epsilonGreedyAction(Q: Float64Array, s: number, A: number, eps: number, r: Stream): number {
  if (uniform(r) < eps) return integers(r, A)
  let best = -Infinity
  let ties: number[] = []
  for (let a = 0; a < A; a++) {
    const q = Q[s * A + a]
    if (q > best) {
      best = q
      ties = [a]
    } else if (q === best) ties.push(a)
  }
  return ties.length === 1 ? ties[0] : ties[integers(r, ties.length)]
}

/** E_π[Q(s, ·)] under the ε-greedy policy: ε times the mean plus (1 − ε) times the maximum. */
function expectedQ(Q: Float64Array, s: number, A: number, eps: number): number {
  let best = -Infinity
  let mean = 0
  for (let a = 0; a < A; a++) {
    const q = Q[s * A + a]
    mean += q / A
    best = Math.max(best, q)
  }
  return eps * mean + (1 - eps) * best
}

/** The state of an episodic learner after an episode. */
export interface LearnerState extends Status {
  /** Action values, states × actions (zeros for REINFORCE and TD prediction). */
  Q: Tensor
  /** State values: max_a Q, the TD(0) estimate, or REINFORCE's baseline. */
  V: Tensor
  /** The greedy policy (int32, −1 at terminals). */
  policy: Tensor
  /** Episodes completed. */
  t: number
  /** The last episode's return Σ γᵗ rₜ (+ γ^T times the terminal value on arrival); 0 before the first episode. */
  return: number
  /** Undiscounted sum of the last episode's rewards (the usual learning-curve quantity). */
  rewardSum: number
  /** States visited in the last episode, starting at the start state (int32). */
  path: Tensor
  /** Actions taken (int32), one fewer than the path unless the episode hit `maxSteps`. */
  actions: Tensor
  /** The last episode ended at a terminal state (rather than at `maxSteps`). */
  reachedTerminal: boolean
  /** Softmax policy parameters θ (REINFORCE only), states × actions. */
  theta?: Tensor
  /** Visit counts (Monte Carlo with sample averages). */
  counts?: Tensor
}

/** Options shared by the episodic learners. */
export interface EpisodeOptions {
  /** The learning rate α (the step size of every value update). */
  learningRate?: number
  /** Exploration rate ε of the ε-greedy behaviour policy. */
  epsilon?: number
  /** Longest episode, a guard against an early policy wandering for ever. Default 1000. */
  maxSteps?: number
  /** Initial action value (optimistic values encourage exploration). Default 0. */
  initialQ?: number
}

function values(mdp: TabularMdp, Q: Float64Array): Float64Array {
  const V = Float64Array.from(mdp.terminalValue)
  for (let s = 0; s < mdp.states; s++) {
    if (!isActive(mdp, s)) continue
    let best = -Infinity
    for (let a = 0; a < mdp.actions; a++) best = Math.max(best, Q[s * mdp.actions + a])
    V[s] = best
  }
  return V
}

function initialState(mdp: TabularMdp, initialQ: number): LearnerState {
  const Q = new Float64Array(mdp.states * mdp.actions)
  for (let st = 0; st < mdp.states; st++)
    if (isActive(mdp, st)) Q.fill(initialQ, st * mdp.actions, (st + 1) * mdp.actions)
  return {
    Q: fromData(Q, [mdp.states, mdp.actions]),
    V: fromData(values(mdp, Q)),
    policy: fromData(greedyActions(mdp, Q)),
    t: 0,
    return: 0,
    rewardSum: 0,
    path: fromData(Int32Array.from([mdp.start])),
    actions: fromData(new Int32Array(0)),
    reachedTerminal: false,
  }
}

function finish(
  mdp: TabularMdp,
  s: LearnerState,
  Q: Float64Array,
  ep: Episode,
  extra: Partial<LearnerState> = {},
): LearnerState {
  return {
    ...s,
    Q: fromData(Q, [mdp.states, mdp.actions]),
    V: fromData(values(mdp, Q)),
    policy: fromData(greedyActions(mdp, Q)),
    t: s.t + 1,
    return: ep.G,
    rewardSum: ep.rewards.reduce((a, b) => a + b, 0),
    path: fromData(Int32Array.from(ep.states)),
    actions: fromData(Int32Array.from(ep.actions)),
    reachedTerminal: ep.terminated,
    ...extra,
  }
}

interface Episode {
  states: number[]
  actions: number[]
  rewards: number[]
  terminated: boolean
  G: number
}

function discountedReturn(mdp: TabularMdp, rewards: number[], final: number, terminated: boolean): number {
  let G = terminated ? mdp.terminalValue[final] : 0
  for (let t = rewards.length - 1; t >= 0; t--) G = rewards[t] + mdp.gamma * G
  return G
}

/** Generate an episode with a behaviour policy `act`. */
function rollout(mdp: TabularMdp, act: (s: number) => number, r: Stream, maxSteps: number): Episode {
  const states = [mdp.start]
  const actions: number[] = []
  const rewards: number[] = []
  let s = mdp.start
  let terminated = !isActive(mdp, s)
  for (let t = 0; t < maxSteps && !terminated; t++) {
    const a = act(s)
    const o = sampleOutcome(r, mdp, s, a)
    actions.push(a)
    rewards.push(o.reward)
    states.push(o.next)
    s = o.next
    terminated = !isActive(mdp, s)
  }
  return { states, actions, rewards, terminated, G: discountedReturn(mdp, rewards, s, terminated) }
}

/** The one-step TD control methods. */
export type TdMethod = 'sarsa' | 'q-learning' | 'expected-sarsa'

/**
 * One-step TD control with an ε-greedy behaviour policy: SARSA bootstraps from the next action actually taken
 * (on-policy), Q-learning from the greedy action (off-policy; Watkins, 1989), expected SARSA from the expectation under
 * the ε-greedy policy (van Seijen et al., 2009). Q(s, a) ← Q(s, a) + α (r + γ · bootstrap − Q(s, a)).
 */
export function tdControl(
  mdp: TabularMdp,
  {
    method = 'q-learning',
    learningRate: alpha = 0.5,
    epsilon = 0.1,
    maxSteps = 1000,
    initialQ = 0,
  }: EpisodeOptions & { method?: TdMethod } = {},
): Algorithm<void, LearnerState> {
  const A = mdp.actions
  return {
    name: method,
    init: () => initialState(mdp, initialQ),
    step(state, ctx) {
      const r = ctx.stream
      const Q = Float64Array.from(state.Q.data)
      const act = (s: number) => epsilonGreedyAction(Q, s, A, epsilon, r)
      const ep: Episode = { states: [mdp.start], actions: [], rewards: [], terminated: false, G: 0 }
      let s = mdp.start
      let a = act(s)
      for (let t = 0; t < maxSteps; t++) {
        const o = sampleOutcome(r, mdp, s, a)
        ep.actions.push(a)
        ep.rewards.push(o.reward)
        ep.states.push(o.next)
        const k = s * A + a
        if (!isActive(mdp, o.next)) {
          Q[k] += alpha * (o.reward + mdp.gamma * mdp.terminalValue[o.next] - Q[k])
          ep.terminated = true
          s = o.next
          break
        }
        const next = act(o.next)
        let boot: number
        if (method === 'sarsa') boot = Q[o.next * A + next]
        else if (method === 'q-learning') {
          boot = -Infinity
          for (let b = 0; b < A; b++) boot = Math.max(boot, Q[o.next * A + b])
        } else boot = expectedQ(Q, o.next, A, epsilon)
        Q[k] += alpha * (o.reward + mdp.gamma * boot - Q[k])
        s = o.next
        a = next
      }
      ep.G = discountedReturn(mdp, ep.rewards, s, ep.terminated)
      return finish(mdp, state, Q, ep)
    },
  }
}

/** Q-learning (see `tdControl`). */
export const qLearning = (mdp: TabularMdp, options: EpisodeOptions = {}) =>
  tdControl(mdp, { ...options, method: 'q-learning' })
/** SARSA (see `tdControl`). */
export const sarsa = (mdp: TabularMdp, options: EpisodeOptions = {}) => tdControl(mdp, { ...options, method: 'sarsa' })
/** Expected SARSA (see `tdControl`). */
export const expectedSarsa = (mdp: TabularMdp, options: EpisodeOptions = {}) =>
  tdControl(mdp, { ...options, method: 'expected-sarsa' })

/**
 * n-step SARSA (Sutton and Barto, 2018, §7.2): the target is the n-step return
 * G_{t:t+n} = Σ_{i<n} γⁱ r_{t+i+1} + γⁿ Q(s_{t+n}, a_{t+n}), truncated at the end of the episode.
 */
export function nStepSarsa(
  mdp: TabularMdp,
  {
    n = 4,
    learningRate: alpha = 0.5,
    epsilon = 0.1,
    maxSteps = 1000,
    initialQ = 0,
  }: EpisodeOptions & { n?: number } = {},
): Algorithm<void, LearnerState> {
  const A = mdp.actions
  const g = mdp.gamma
  return {
    name: `${n}-step SARSA`,
    init: () => initialState(mdp, initialQ),
    step(state, ctx) {
      const r = ctx.stream
      const Q = Float64Array.from(state.Q.data)
      const act = (s: number) => epsilonGreedyAction(Q, s, A, epsilon, r)
      const S = [mdp.start]
      const Acts = [act(mdp.start)]
      const R = [0]
      let T = Infinity
      for (let t = 0; ; t++) {
        if (t < T) {
          const o = sampleOutcome(r, mdp, S[t], Acts[t])
          R.push(o.reward)
          S.push(o.next)
          if (!isActive(mdp, o.next) || t + 1 >= maxSteps) T = t + 1
          else Acts.push(act(o.next))
        }
        const tau = t - n + 1
        if (tau >= 0) {
          let G = 0
          for (let i = tau + 1; i <= Math.min(tau + n, T); i++) G += g ** (i - tau - 1) * R[i]
          if (tau + n < T) G += g ** n * Q[S[tau + n] * A + Acts[tau + n]]
          else if (T !== Infinity && !isActive(mdp, S[T])) G += g ** (T - tau) * mdp.terminalValue[S[T]]
          const k = S[tau] * A + Acts[tau]
          Q[k] += alpha * (G - Q[k])
        }
        if (tau === T - 1) break
      }
      const terminated = !isActive(mdp, S[S.length - 1])
      const rewards = R.slice(1)
      const ep: Episode = {
        states: S,
        actions: Acts.slice(0, rewards.length),
        rewards,
        terminated,
        G: discountedReturn(mdp, rewards, S[S.length - 1], terminated),
      }
      return finish(mdp, state, Q, ep)
    },
  }
}

/**
 * On-policy first-visit Monte Carlo control with ε-soft policies (Sutton and Barto, 2018, §5.4): after each episode,
 * Q(s, a) moves towards the return that followed the first visit of (s, a), by sample averaging (default) or a
 * constant learning rate α.
 */
export function monteCarloControl(
  mdp: TabularMdp,
  { learningRate: alpha, epsilon = 0.1, maxSteps = 1000, initialQ = 0 }: EpisodeOptions = {},
): Algorithm<void, LearnerState> {
  const A = mdp.actions
  return {
    name: 'Monte Carlo control',
    init: () => ({
      ...initialState(mdp, initialQ),
      counts: fromData(new Float64Array(mdp.states * A), [mdp.states, A]),
    }),
    step(state, ctx) {
      const r = ctx.stream
      const Q = Float64Array.from(state.Q.data)
      const N = Float64Array.from(state.counts!.data)
      const ep = rollout(mdp, (s) => epsilonGreedyAction(Q, s, A, epsilon, r), r, maxSteps)
      const T = ep.actions.length
      const first = new Map<number, number>()
      for (let t = 0; t < T; t++) {
        const k = ep.states[t] * A + ep.actions[t]
        if (!first.has(k)) first.set(k, t)
      }
      let G = ep.terminated ? mdp.terminalValue[ep.states[T]] : 0
      for (let t = T - 1; t >= 0; t--) {
        G = ep.rewards[t] + mdp.gamma * G
        const k = ep.states[t] * A + ep.actions[t]
        if (first.get(k) !== t) continue
        N[k] += 1
        Q[k] += (alpha ?? 1 / N[k]) * (G - Q[k])
      }
      return finish(mdp, state, Q, ep, { counts: fromData(N, [mdp.states, A]) })
    },
  }
}

/**
 * TD(0) prediction of a fixed policy's value (Sutton, 1988, Machine Learning 3): V(s) ← V(s) + α (r + γ V(s′) − V(s))
 * after every transition. `policy` is deterministic (an action per state) or stochastic (states × actions).
 */
export function tdPrediction(
  mdp: TabularMdp,
  policy: PolicyInput,
  { learningRate: alpha = 0.1, maxSteps = 1000 }: Pick<EpisodeOptions, 'learningRate' | 'maxSteps'> = {},
): Algorithm<void, LearnerState> {
  const A = mdp.actions
  const pi = policyMatrix(mdp, policy)
  return {
    name: 'TD(0) prediction',
    init: () => {
      const base = initialState(mdp, 0)
      return { ...base, V: fromData(Float64Array.from(mdp.terminalValue)) }
    },
    step(state, ctx) {
      const r = ctx.stream
      const V = Float64Array.from(state.V.data)
      const act = (s: number) => {
        let u = uniform(r)
        for (let a = 0; a < A; a++) {
          if (u < pi[s * A + a]) return a
          u -= pi[s * A + a]
        }
        return A - 1
      }
      const ep = rollout(mdp, act, r, maxSteps)
      for (let t = 0; t < ep.actions.length; t++) {
        const s = ep.states[t]
        const next = ep.states[t + 1]
        V[s] += alpha * (ep.rewards[t] + mdp.gamma * V[next] - V[s])
      }
      const Q = state.Q.data as Float64Array
      return { ...finish(mdp, state, Q, ep), V: fromData(V) }
    },
  }
}

/**
 * REINFORCE (Williams, 1992, Machine Learning 8) with a tabular softmax policy π(a|s) ∝ exp θ(s, a): after each
 * episode, θ(s_t, ·) += α γᵗ (G_t − b(s_t)) (e_{a_t} − π(·|s_t)), the score-function gradient. With `baseline`, b is a
 * state-value estimate learned alongside at learning rate `baselineLearningRate` (Sutton and Barto, 2018, §13.4); without, b = 0.
 */
export function reinforce(
  mdp: TabularMdp,
  {
    learningRate: alpha = 0.1,
    baseline = true,
    baselineLearningRate: baselineAlpha = 0.1,
    maxSteps = 1000,
  }: { learningRate?: number; baseline?: boolean; baselineLearningRate?: number; maxSteps?: number } = {},
): Algorithm<void, LearnerState & { probabilities: Tensor }> {
  const A = mdp.actions
  const probs = (theta: Float64Array, s: number) => {
    let mx = -Infinity
    for (let a = 0; a < A; a++) mx = Math.max(mx, theta[s * A + a])
    const e = Array.from({ length: A }, (_, a) => Math.exp(theta[s * A + a] - mx))
    const z = e.reduce((x, y) => x + y, 0)
    return e.map((v) => v / z)
  }
  const allProbs = (theta: Float64Array) => {
    const P = new Float64Array(mdp.states * A)
    for (let s = 0; s < mdp.states; s++) if (isActive(mdp, s)) P.set(probs(theta, s), s * A)
    return fromData(P, [mdp.states, A])
  }
  return {
    name: baseline ? 'REINFORCE with baseline' : 'REINFORCE',
    init: () => {
      const theta = new Float64Array(mdp.states * A)
      return {
        ...initialState(mdp, 0),
        V: fromData(Float64Array.from(mdp.terminalValue)),
        theta: fromData(theta, [mdp.states, A]),
        probabilities: allProbs(theta),
      }
    },
    step(state, ctx) {
      const r = ctx.stream
      const theta = Float64Array.from(state.theta!.data)
      const V = Float64Array.from(state.V.data)
      const ep = rollout(
        mdp,
        (s) => {
          const p = probs(theta, s)
          let u = uniform(r)
          for (let a = 0; a < A; a++) {
            if (u < p[a]) return a
            u -= p[a]
          }
          return A - 1
        },
        r,
        maxSteps,
      )
      const T = ep.actions.length
      // Returns from each step, then the updates (computed with the policy that generated the episode).
      const G = new Float64Array(T + 1)
      G[T] = ep.terminated ? mdp.terminalValue[ep.states[T]] : 0
      for (let t = T - 1; t >= 0; t--) G[t] = ep.rewards[t] + mdp.gamma * G[t + 1]
      const before = Float64Array.from(theta)
      for (let t = 0; t < T; t++) {
        const s = ep.states[t]
        const delta = G[t] - (baseline ? V[s] : 0)
        if (baseline) V[s] += baselineAlpha * delta
        const p = probs(before, s)
        const scale = alpha * mdp.gamma ** t * delta
        for (let a = 0; a < A; a++) theta[s * A + a] += scale * ((a === ep.actions[t] ? 1 : 0) - p[a])
      }
      const Qprefs = Float64Array.from(theta)
      return {
        ...finish(mdp, state, Qprefs, ep),
        Q: state.Q,
        V: fromData(V),
        theta: fromData(theta, [mdp.states, A]),
        probabilities: allProbs(theta),
      }
    },
  }
}

/**
 * The path a policy follows from the start when every move takes its most likely outcome: stops at a terminal state,
 * a repeated state, or after `maxLength` states. For drawing the greedy route on a grid.
 */
export function greedyPath(mdp: TabularMdp, policy: Tensor | ArrayLike<number>, maxLength = 4 * mdp.states): Tensor {
  const pol = 'shape' in policy ? policy.data : policy
  const path = [mdp.start]
  const seen = new Set(path)
  let s = mdp.start
  while (isActive(mdp, s) && path.length < maxLength) {
    const a = pol[s]
    if (a < 0) break
    const outs = mdp.outcomes[s * mdp.actions + a]
    const o = outs.reduce((best, x) => (x.p > best.p ? x : best), outs[0])
    if (seen.has(o.next)) break
    path.push(o.next)
    seen.add(o.next)
    s = o.next
  }
  return fromData(Int32Array.from(path))
}
