/**
 * Finite Markov decision processes as plain data, and the grid environments built on them: the stochastic gridworld,
 * cliff walking, text mazes and FrozenLake.
 *
 * Conventions. States are integers; a grid cell (x, y) is state y · width + x with y = 0 at the bottom row. Actions on
 * grids are up, right, down, left (0–3). Terminal states take no action and have a fixed value `terminalValue` (0
 * unless an environment pays an exit reward there), so every backup is r + γ V(s′) with V(s′) = terminalValue(s′) at a
 * terminal s′, and an episode's return adds γ^T · terminalValue on arrival.
 */

import type { Environment, GridRender, Outcome, TabularModel } from 'aifn/foundation/contracts'
import { uniform, type Stream } from 'aifn/foundation/random'
import { discreteDomain } from 'aifn/foundation/space'
import type { Tensor } from 'aifn/foundation/tensor'

export type { Outcome } from 'aifn/foundation/contracts'

/** What a grid cell is, for drawing and for the rules. */
export type CellKind = 'open' | 'wall' | 'start' | 'goal' | 'trap' | 'hole' | 'cliff' | 'terminal'

/** A finite MDP. */
export interface TabularMdp {
  name: string
  states: number
  actions: number
  /** `outcomes[s * actions + a]`: the distribution over (next state, reward). Empty for terminal states and walls. */
  outcomes: Outcome[][]
  /** The start state of every episode. */
  start: number
  /** 1 for states where no action is taken (terminals and walls). */
  terminal: Uint8Array
  /** The fixed value of each terminal state (0 elsewhere). */
  terminalValue: Float64Array
  gamma: number
  actionNames: string[]
  /** Grid layout, when the MDP is a grid. */
  grid?: { width: number; height: number; kinds: CellKind[]; actionVectors: [number, number][] }
}

/** Up, right, down, left as (dx, dy) with y pointing up. */
export const GRID_ACTIONS: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]
export const GRID_ACTION_NAMES = ['up', 'right', 'down', 'left']

/** The state of grid cell (x, y). */
export const cellState = (width: number, x: number, y: number): number => y * width + x
/** The (x, y) cell of a grid state. */
export const stateCell = (width: number, s: number): [number, number] => [s % width, Math.floor(s / width)]

/** The tables planners read: what a `TabularMdp` and an environment's tabular model have in common. */
export type MdpTables = Pick<
  TabularModel<unknown>,
  'states' | 'actions' | 'outcomes' | 'terminal' | 'terminalValue' | 'gamma'
>

/** States where an action is taken. */
export const isActive = (mdp: MdpTables, s: number): boolean => !mdp.terminal[s]

/** Sample an outcome of action a in state s from the stream r (one uniform draw). */
export function sampleOutcome(r: Stream, mdp: MdpTables, s: number, a: number): Outcome {
  const outs = mdp.outcomes[s * mdp.actions + a]
  let u = uniform(r)
  for (const o of outs) {
    if (u < o.p) return o
    u -= o.p
  }
  return outs[outs.length - 1]
}

/**
 * A general finite MDP from dense arrays: P[s][a][s′] transition probabilities and R[s][a][s′] rewards (or R[s][a]
 * expected rewards), with optional terminal states.
 */
export function tabularMdp(options: {
  transitions: readonly (readonly (readonly number[])[])[]
  rewards: readonly (readonly (readonly number[] | number)[])[]
  gamma: number
  start?: number
  terminal?: readonly number[]
  terminalValue?: readonly number[]
  name?: string
  actionNames?: string[]
}): TabularMdp {
  const P = options.transitions
  const S = P.length
  const A = P[0].length
  const terminal = new Uint8Array(S)
  for (const s of options.terminal ?? []) terminal[s] = 1
  const outcomes: Outcome[][] = []
  for (let s = 0; s < S; s++)
    for (let a = 0; a < A; a++) {
      if (terminal[s]) {
        outcomes.push([])
        continue
      }
      const r = options.rewards[s][a]
      outcomes.push(
        P[s][a].flatMap((p, t) => (p > 0 ? [{ p, next: t, reward: typeof r === 'number' ? r : r[t] }] : [])),
      )
    }
  return {
    name: options.name ?? 'MDP',
    states: S,
    actions: A,
    outcomes,
    start: options.start ?? 0,
    terminal,
    terminalValue: Float64Array.from({ length: S }, (_, s) => (terminal[s] ? (options.terminalValue?.[s] ?? 0) : 0)),
    gamma: options.gamma,
    actionNames: options.actionNames ?? Array.from({ length: A }, (_, a) => `a${a}`),
  }
}

// ── Policies (shared by planning and learning) ─────────────────────────────────────────────────────────────────────

/** A policy: a deterministic action per state (int32, −1 at terminals) or a stochastic one, states × actions. */
export type PolicyInput = Tensor | readonly number[]

/** The greedy action per state (lowest index on ties within 1e-9, or `prefer[s]` if it is tied), −1 at terminals. */
export function greedyActions(mdp: MdpTables, Q: ArrayLike<number>, prefer?: ArrayLike<number>): Int32Array {
  const { states: S, actions: A } = mdp
  const out = new Int32Array(S).fill(-1)
  for (let s = 0; s < S; s++) {
    if (!isActive(mdp, s)) continue
    let best = -Infinity
    for (let a = 0; a < A; a++) best = Math.max(best, Q[s * A + a])
    const keep = prefer?.[s]
    if (keep !== undefined && keep >= 0 && Q[s * A + keep] >= best - 1e-9) out[s] = keep
    else
      for (let a = 0; a < A; a++)
        if (Q[s * A + a] >= best - 1e-9) {
          out[s] = a
          break
        }
  }
  return out
}

/** π(a | s) as a dense states × actions array from a deterministic or stochastic policy. */
export function policyMatrix(mdp: MdpTables, policy: PolicyInput): Float64Array {
  const { states: S, actions: A } = mdp
  const v = 'shape' in policy ? Array.from(policy.data) : [...policy]
  if (v.length === S * A) return Float64Array.from(v)
  if (v.length !== S) throw new Error(`policy: expected ${S} actions or ${S} × ${A} probabilities`)
  const pi = new Float64Array(S * A)
  for (let s = 0; s < S; s++) if (isActive(mdp, s) && v[s] >= 0) pi[s * A + v[s]] = 1
  return pi
}

// ── As an environment ──────────────────────────────────────────────────────────────────────────────────────────────

/** A tabular MDP as an `Environment` whose state and observation are the state index and whose action is an index. */
export type MdpEnvironment = Environment<number, number, number> & {
  readonly model: TabularModel<number>
  readonly render?: GridRender<number>
}

/**
 * A finite MDP as an `Environment` (docs/aifn-environments.md §6): episodes start at `mdp.start`; a step samples one
 * outcome with one uniform draw; arriving at a terminal state ends the episode (`terminated`) and pays the outcome's
 * reward plus γ times the state's terminal value, so returns agree with the MDP's values. The tables are the `model`
 * (for planners), and a grid MDP gains a grid `render`. `horizon` (default 4 × states) caps an episode.
 */
export function mdpEnvironment(
  mdp: TabularMdp,
  { horizon = 4 * mdp.states }: { horizon?: number } = {},
): MdpEnvironment {
  const id = (s: number) => s
  const model: TabularModel<number> = {
    kind: 'tabular',
    states: mdp.states,
    actions: mdp.actions,
    outcomes: mdp.outcomes,
    terminal: mdp.terminal,
    terminalValue: mdp.terminalValue,
    gamma: mdp.gamma,
    encode: id,
    decode: id,
  }
  const g = mdp.grid
  return {
    name: mdp.name,
    observation: discreteDomain(mdp.states),
    action: discreteDomain(mdp.actions, mdp.actionNames),
    gamma: mdp.gamma,
    horizon,
    reset: () => ({ state: mdp.start, observation: mdp.start }),
    step(s, a, stream) {
      if (!isActive(mdp, s)) throw new RangeError(`${mdp.name}: step from terminal state ${s}`)
      const o = sampleOutcome(stream, mdp, s, a)
      const terminated = !isActive(mdp, o.next)
      const reward = o.reward + (terminated ? mdp.gamma * mdp.terminalValue[o.next] : 0)
      return { state: o.next, observation: o.next, reward, terminated, truncated: false }
    },
    model,
    ...(g && {
      render: {
        kind: 'grid' as const,
        width: g.width,
        height: g.height,
        cells: g.kinds,
        cell: id,
        actionVectors: g.actionVectors,
      },
    }),
  }
}
