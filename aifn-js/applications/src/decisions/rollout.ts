/**
 * Running an agent in an environment (docs/aifn-environments.md §4): `rollout` takes one environment step per algorithm
 * step, `episodes` one whole episode per step, and `randomAgent` acts uniformly over the action domain. Both loops are
 * traceable `Algorithm`s. Step t's environment draws come from `child(ctx.stream, 'env')` and the agent's from
 * `child(ctx.stream, 'agent')`, as in `banditRun`, so two agents traced on one root stream face the same environment
 * randomness at every step. The rollout truncates an episode at the environment's `horizon` (Gymnasium's `TimeLimit`).
 */

import type { Agent, AgentInfo, Domain, Environment, Status, StepContext, Transition } from 'aifn/foundation/contracts'
import { child, type Stream } from 'aifn/foundation/random'
import { definer } from 'aifn/foundation/registry'
import { sampleDomain, space } from 'aifn/foundation/space'
import type { Algorithm } from 'aifn/foundation/trace'

// ── One step ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The transition of one step, with the agent's scores and action probabilities when it reports them. */
export interface RolloutTransition<O, A> extends Transition<O, A> {
  scores?: Float64Array
  probabilities?: Float64Array
}

/** The rollout's view of a running episode. */
interface Position<S, O, G> {
  envState: S
  observation: O
  agent: G
  /** Steps taken in the current episode. */
  length: number
}

/** One environment step: act, step, truncate at the horizon, learn. */
function advance<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  at: Position<S, O, G>,
  stream: Stream,
): { at: Position<S, O, G>; transition: RolloutTransition<O, A> } {
  const choice = agent.act(at.agent, at.observation, child(stream, 'agent'))
  const step = env.step(at.envState, choice.action, child(stream, 'env'))
  const length = at.length + 1
  const transition: RolloutTransition<O, A> = {
    observation: at.observation,
    action: choice.action,
    reward: step.reward,
    next: step.observation,
    terminated: step.terminated,
    truncated: !step.terminated && (step.truncated || length >= env.horizon),
    ...(choice.scores && { scores: choice.scores }),
    ...(choice.probabilities && { probabilities: choice.probabilities }),
  }
  return {
    at: { envState: step.state, observation: step.observation, agent: agent.learn(at.agent, transition), length },
    transition,
  }
}

/** A fresh episode from a reset drawn from `stream`. */
function begin<S, O, A, G>(env: Environment<S, O, A>, agent: G, stream: Stream): Position<S, O, G> {
  const { state, observation } = env.reset(stream)
  return { envState: state, observation, agent, length: 0 }
}

// ── rollout ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A rollout's state after step t: the environment, the agent, and the episode so far. */
export interface RolloutState<S, O, A, G> extends Status {
  /** Environment steps taken. */
  t: number
  /** Episodes completed. */
  episode: number
  /** The environment's state and the agent's observation after the last step (the reset state at t = 0). */
  envState: S
  observation: O
  /** The agent's state (what it has learnt). */
  agent: G
  /** The last step's transition (null at t = 0 and before an episode's first step). */
  last: RolloutTransition<O, A> | null
  /** Steps taken in the current episode. */
  length: number
  /** The undiscounted sum of the current episode's rewards. */
  episodeReturn: number
  /** The last step ended its episode: the next step resets the environment first. */
  ended: boolean
}

/** Options for `rollout`. */
export interface RolloutOptions {
  /** Stop (`done`) once this many episodes have ended. Default: no limit. */
  episodes?: number
}

/**
 * One environment step per algorithm step (no start). When a step ends its episode (`terminated`, or `truncated` by the
 * environment or at its `horizon`), the state shows the arrival and the next step resets the environment, from
 * `child(ctx.stream, 'reset')`, before acting. Step 0 resets from the init stream.
 */
export function rollout<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
  { episodes }: RolloutOptions = {},
): Algorithm<void, RolloutState<S, O, A, G>> {
  return {
    name: `${agent.name} on ${env.name}`,
    init(_, stream) {
      const at = begin(env, agent.init(env, child(stream, 'agent')), child(stream, 'env'))
      return { t: 0, episode: 0, ...at, last: null, episodeReturn: 0, ended: false }
    },
    step(s, ctx: StepContext) {
      const from = s.ended ? begin(env, s.agent, child(ctx.stream, 'reset')) : s
      const { at, transition } = advance(env, agent, from, ctx.stream)
      const ended = transition.terminated || transition.truncated
      return {
        t: s.t + 1,
        episode: s.episode + (ended ? 1 : 0),
        ...at,
        last: transition,
        episodeReturn: (s.ended ? 0 : s.episodeReturn) + transition.reward,
        ended,
      }
    },
    done: (s) => episodes !== undefined && s.episode >= episodes,
  }
}

// ── episodes ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** An `episodes` state after e episodes: the agent and the last episode in full. */
export interface EpisodeState<O, A, G> extends Status {
  /** Episodes completed. */
  t: number
  agent: G
  /** The last episode's observations, from the reset to the arrival (one more than its actions). */
  observations: O[]
  actions: A[]
  rewards: number[]
  /** The undiscounted sum of the last episode's rewards (0 before the first). */
  episodeReturn: number
  /** The last episode ended at a terminal state, not by truncation. */
  reachedTerminal: boolean
}

/**
 * One whole episode per algorithm step (no start). Episode e resets from `child(ctx.stream, 'reset')` and takes its
 * k-th environment step on `child(ctx.stream, 'step', k)`, split into `env` and `agent` as in `rollout`. An episode
 * ends at a terminal state or at the environment's `horizon` (which must then be finite).
 */
export function episodes<S, O, A, G>(
  env: Environment<S, O, A>,
  agent: Agent<G, O, A>,
): Algorithm<void, EpisodeState<O, A, G>> {
  return {
    name: `${agent.name} on ${env.name}, by episode`,
    init(_, stream) {
      const { observation } = env.reset(child(stream, 'env'))
      return {
        t: 0,
        agent: agent.init(env, child(stream, 'agent')),
        observations: [observation],
        actions: [],
        rewards: [],
        episodeReturn: 0,
        reachedTerminal: false,
      }
    },
    step(s, ctx) {
      let at = begin(env, s.agent, child(ctx.stream, 'reset'))
      const observations = [at.observation]
      const actions: A[] = []
      const rewards: number[] = []
      let terminated = false
      for (let k = 0; ; k++) {
        const { at: next, transition } = advance(env, agent, at, child(ctx.stream, 'step', k))
        at = next
        observations.push(transition.next)
        actions.push(transition.action)
        rewards.push(transition.reward)
        terminated = transition.terminated
        if (transition.terminated || transition.truncated) break
      }
      return {
        t: s.t + 1,
        agent: at.agent,
        observations,
        actions,
        rewards,
        episodeReturn: rewards.reduce((a, b) => a + b, 0),
        reachedTerminal: terminated,
      }
    },
  }
}

// ── The random agent ─────────────────────────────────────────────────────────────────────────────────────────────────

/** The random agent's state: the action domain it draws from. */
export interface RandomAgentState {
  action: Domain
}

/** An agent acting uniformly at random over the action domain; it learns nothing. */
export function randomAgent(): Agent<RandomAgentState, unknown, number> {
  return {
    name: 'random',
    init: (env) => ({ action: env.action }),
    act: (g, _, stream) => {
      const n = g.action.n
      return { action: sampleDomain(stream, g.action), probabilities: new Float64Array(n).fill(1 / n) }
    },
    learn: (g) => g,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<AgentInfo>('agent', 'decisions')(
  {
    key: 'randomAgent',
    name: 'Random agent',
    summary:
      'Acts uniformly at random over a discrete action domain and learns nothing: the baseline every agent beats.',
    params: space({}),
    requires: { action: 'discrete' },
    random: true,
  },
  randomAgent,
)
