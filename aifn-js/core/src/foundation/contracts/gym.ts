/**
 * Sequential decisions: the `Environment` an agent acts in, the `Agent` that acts and learns, and the `Domain` of their
 * observations and actions (docs/aifn-gym.md §3). Gymnasium's semantics in aifn's style: environment state is
 * plain data, randomness is an explicit `Stream`, and `terminated` (a true end: values do not bootstrap past it) is kept
 * apart from `truncated` (cut short by a time limit: values still bootstrap). Implementations live in applications
 * (`aifn-applied/gym`).
 */

import type { Index, Scalar, Shape, Size, Tensor, Value } from './numbers'
import type { Stream } from './random'

// ── Domains ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The integers 0 … n − 1 (Gymnasium's `Discrete(n)`), with optional display names, one per value. */
export interface DiscreteDomain {
  readonly kind: 'discrete'
  readonly n: Size
  readonly names?: readonly string[]
}

/**
 * A box of real arrays of shape `shape` (Gymnasium's `Box(low, high, shape)`): every element lies in
 * [low[i], high[i]], the bounds flat in row-major order (bounds may be infinite). Optional display names, one per
 * element.
 */
export interface BoxDomain {
  readonly kind: 'box'
  readonly low: readonly number[]
  readonly high: readonly number[]
  readonly shape: Shape
  readonly names?: readonly string[]
}

/** A set of observations or actions: discrete values or a continuous box. */
export type Domain = DiscreteDomain | BoxDomain

/** The kinds of domain, as the registry pairs agents with environments. */
export type DomainKind = Domain['kind']

// ── Environment ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** What `step` returns: the next state, what the agent sees of it, the reward, and whether the episode ended. */
export interface Step<S, O> {
  state: S
  observation: O
  reward: Scalar
  /** A terminal state: the return ends here and values do not bootstrap past it. */
  terminated: boolean
  /** Cut short by the environment itself (the rollout also truncates at `horizon`): values still bootstrap. */
  truncated: boolean
}

/** One possible result of an action in a finite MDP: probability `p` of reaching state `next` with `reward`. */
export interface Outcome {
  p: number
  next: Index
  reward: Scalar
}

/**
 * An explicit finite MDP, for planners: `outcomes[s · actions + a]` is the distribution over (next state, reward), empty
 * at terminal states. A terminal state has the fixed value `terminalValue[s]`. `encode` and `decode` map environment
 * states to table indices.
 */
export interface TabularModel<S> {
  readonly kind: 'tabular'
  readonly states: Size
  readonly actions: Size
  readonly outcomes: readonly (readonly Outcome[])[]
  /** 1 for states where no action is taken. */
  readonly terminal: Uint8Array
  readonly terminalValue: Float64Array
  readonly gamma: Scalar
  encode(state: S): Index
  decode(index: Index): S
}

/**
 * Deterministic, differentiable dynamics, for model-based control (LQR about an equilibrium, iLQR, MPC): the state as
 * a vector x of length `stateSize` and the action as a vector u of length `actionSize`. `transition` gives the next
 * state and `reward` the reward of taking u in x; both are written with tensor primitives, so x and u may be traced
 * (`jacobian`, `grad`). `encode` and `decode` map environment states to and from x.
 */
export interface DynamicsModel<S> {
  readonly kind: 'dynamics'
  readonly stateSize: Size
  readonly actionSize: Size
  transition(x: Value, u: Value): Value
  reward(x: Value, u: Value): Value
  encode(state: S): Tensor
  decode(x: Value): S
}

/** The explicit dynamics of an environment: a finite MDP's tables, or differentiable deterministic dynamics. */
export type EnvironmentModel<S> = TabularModel<S> | DynamicsModel<S>

/** Ground truth for evaluation, never shown to agents. */
export interface EnvironmentOracle<S, A> {
  /** E[reward | state, action] (a bandit's arm means). */
  expectedReward?(state: S, action: A): Scalar
  /** The best achievable expected reward in a state (for pseudo-regret). */
  bestExpectedReward?(state: S): Scalar
  /** Optimal state values, when known exactly. */
  optimalValues?(): Tensor
}

/**
 * How the lab draws a state. `grid`: a `width` × `height` grid of cells, cell (x, y) at index y · width + x with y = 0 at
 * the bottom row; `cells` names each cell's kind (`wall`, `goal`, …), `cell` gives a state's cell index, and
 * `actionVectors` the (dx, dy) of each action.
 */
export interface GridRender<S> {
  readonly kind: 'grid'
  readonly width: Size
  readonly height: Size
  readonly cells: readonly string[]
  cell(state: S): Index
  readonly actionVectors: readonly (readonly [number, number])[]
}

/** `pendulum`: a rod of `length` pivoted at the origin, at `angle(state)` radians from upright (anticlockwise). */
export interface PendulumRender<S> {
  readonly kind: 'pendulum'
  readonly length: number
  angle(state: S): number
}

/**
 * `cartpole`: a cart at `cart(state)` on a track [−`trackLimit`, `trackLimit`], carrying a pole of `poleLength` at
 * `angle(state)` radians from upright (positive clockwise, towards +x).
 */
export interface CartPoleRender<S> {
  readonly kind: 'cartpole'
  readonly poleLength: number
  readonly trackLimit: number
  cart(state: S): number
  angle(state: S): number
}

/** How to draw an environment's state. */
export type RenderSpec<S> = GridRender<S> | PendulumRender<S> | CartPoleRender<S>

/**
 * What an agent may know of an environment before acting: its domains, discount and episode cap, and its explicit
 * model when it has one (planning agents require it).
 */
export interface EnvironmentShape {
  readonly name: string
  readonly observation: Domain
  readonly action: Domain
  /** The discount the problem is posed with (agents may use their own). */
  readonly gamma: Scalar
  /** The longest episode: the rollout truncates there (`Infinity` for none). */
  readonly horizon: number
  readonly model?: EnvironmentModel<unknown>
}

/** How an episode ended, for display: a success (a goal reached, survived the time limit) or a failure, and why. */
export interface EpisodeEnd {
  success: boolean
  /** A short reason, e.g. "pole fell: θ = 13.1°", "reached the goal", "survived 500 steps". */
  reason: string
}

/**
 * An environment. `S` is its state (plain data), `O` what the agent observes, `A` an action. `reset` and `step` are
 * pure and draw only from their stream. The optional capabilities serve planners (`model`), evaluation (`oracle`) and
 * the lab (`render`).
 */
export interface Environment<S, O, A> extends EnvironmentShape {
  reset(stream: Stream): { state: S; observation: O }
  step(state: S, action: A, stream: Stream): Step<S, O>
  /** The actions allowed in a state, when not every action of the domain is. */
  legal?(state: S): readonly A[]
  /**
   * How an episode that ended at `state` went: `terminated` there, or `truncated` after `steps` steps. Null when the
   * environment has no notion of success (a bandit's round).
   */
  ending?(state: S, how: 'terminated' | 'truncated', steps: number): EpisodeEnd | null
  readonly model?: EnvironmentModel<S>
  readonly oracle?: EnvironmentOracle<S, A>
  readonly render?: RenderSpec<S>
}

// ── Agent ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One environment step as the agent sees it. */
export interface Transition<O, A> {
  observation: O
  action: A
  reward: Scalar
  next: O
  terminated: boolean
  truncated: boolean
  /** The actions legal at `next`, when the environment masks actions (`legal`). */
  nextLegal?: readonly A[]
}

/** An agent's choice, with what the lab may show: the scores it ranked actions by and its action probabilities. */
export interface Decision<A> {
  action: A
  scores?: Float64Array
  probabilities?: Float64Array
}

/**
 * A learning agent. Its state `G` is plain data; `init`, `act` and `learn` are pure, and only `init` and `act` draw
 * randomness. Episode-level learners buffer transitions in `learn` and update at the episode's end.
 */
export interface Agent<G, O, A> {
  readonly name: string
  init(env: EnvironmentShape, stream: Stream): G
  /** Choose an action; `legal` lists the allowed actions when the environment masks some. */
  act(agent: G, observation: O, stream: Stream, legal?: readonly A[]): Decision<A>
  learn(agent: G, transition: Transition<O, A>): G
  /** The deterministic action of the learnt policy (no exploration), for evaluating it; `act` is used when absent. */
  greedy?(agent: G, observation: O, legal?: readonly A[]): A
  /** Scalars that track learning (a value estimate, an exploration rate, a loss), for training curves. */
  scalars?(agent: G): Readonly<Record<string, number>>
}
