/**
 * Sequential decisions: the `Environment` an agent acts in, the `Agent` that acts and learns, and the `Domain` of their
 * observations and actions (docs/aifn-environments.md §3). Gymnasium's semantics in aifn's style: environment state is
 * plain data, randomness is an explicit `Stream`, and `terminated` (a true end: values do not bootstrap past it) is kept
 * apart from `truncated` (cut short by a time limit: values still bootstrap). Implementations live in applications
 * (`aifn-applied/decisions`).
 */

import type { Index, Scalar, Size, Tensor } from './numbers'
import type { Stream } from './random'

// ── Domains ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A set of observations or actions. `discrete` is the integers 0 … n − 1 (Gymnasium's `Discrete(n)`), with optional
 * display names, one per value. Continuous boxes (for classic control) will join this union as `{ kind: 'box', … }`.
 */
export type Domain = { readonly kind: 'discrete'; readonly n: Size; readonly names?: readonly string[] }

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

/** The explicit dynamics of an environment. Deterministic, differentiable dynamics (for MPC, iLQR) will join this. */
export type EnvironmentModel<S> = TabularModel<S>

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

/** How to draw an environment's state. */
export type RenderSpec<S> = GridRender<S>

/** What an agent may know of an environment before acting: its domains, discount and episode cap. */
export interface EnvironmentShape {
  readonly name: string
  readonly observation: Domain
  readonly action: Domain
  /** The discount the problem is posed with (agents may use their own). */
  readonly gamma: Scalar
  /** The longest episode: the rollout truncates there (`Infinity` for none). */
  readonly horizon: number
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
  act(agent: G, observation: O, stream: Stream): Decision<A>
  learn(agent: G, transition: Transition<O, A>): G
}
