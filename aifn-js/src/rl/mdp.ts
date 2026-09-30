/**
 * Finite Markov decision processes as plain data, and the grid environments built on them: the stochastic gridworld,
 * cliff walking, text mazes and FrozenLake.
 *
 * Conventions. States are integers; a grid cell (x, y) is state y · width + x with y = 0 at the bottom row. Actions on
 * grids are up, right, down, left (0–3). Terminal states take no action and have a fixed value `terminalValue` (0
 * unless an environment pays an exit reward there), so every backup is r + γ V(s′) with V(s′) = terminalValue(s′) at a
 * terminal s′, and an episode's return adds γ^T · terminalValue on arrival.
 */

/** One possible result of taking an action. */
export interface Outcome {
  p: number
  next: number
  reward: number
}

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

/** States where an action is taken. */
export const isActive = (mdp: TabularMdp, s: number): boolean => !mdp.terminal[s]

interface GridSpec {
  name: string
  width: number
  height: number
  kinds: CellKind[]
  start: number
  gamma: number
  /** Probabilities of the intended direction and of each perpendicular one. */
  slip: number
  /** The result of landing on a cell after a move from `from`. */
  land: (cell: number) => { next: number; reward: number }
  terminalValue?: (s: number) => number
  /** Kinds that end an episode. */
  terminalKinds: CellKind[]
  /** A move off the grid or into a wall leaves the agent in place; `clamp` does so without calling `land`. */
}

function buildGrid(spec: GridSpec): TabularMdp {
  const { width, height, kinds, slip } = spec
  const S = width * height
  const terminal = Uint8Array.from(kinds, (k) => (k === 'wall' || spec.terminalKinds.includes(k) ? 1 : 0))
  const terminalValue = Float64Array.from({ length: S }, (_, s) =>
    terminal[s] && spec.terminalValue ? spec.terminalValue(s) : 0,
  )
  const move = (s: number, a: number) => {
    const [x, y] = stateCell(width, s)
    const nx = x + GRID_ACTIONS[a][0]
    const ny = y + GRID_ACTIONS[a][1]
    if (nx < 0 || ny < 0 || nx >= width || ny >= height) return s
    const t = cellState(width, nx, ny)
    return kinds[t] === 'wall' ? s : t
  }
  const outcomes: Outcome[][] = []
  for (let s = 0; s < S; s++)
    for (let a = 0; a < 4; a++) {
      if (terminal[s]) {
        outcomes.push([])
        continue
      }
      const dirs: [number, number][] =
        slip > 0
          ? [
              [a, 1 - slip],
              [(a + 1) % 4, slip / 2],
              [(a + 3) % 4, slip / 2],
            ]
          : [[a, 1]]
      // Merge outcomes that land in the same place with the same reward.
      const merged = new Map<string, Outcome>()
      for (const [dir, p] of dirs) {
        if (p <= 0) continue
        const { next, reward } = spec.land(move(s, dir))
        const key = `${next}:${reward}`
        const o = merged.get(key)
        if (o) o.p += p
        else merged.set(key, { p, next, reward })
      }
      outcomes.push([...merged.values()])
    }
  return {
    name: spec.name,
    states: S,
    actions: 4,
    outcomes,
    start: spec.start,
    terminal,
    terminalValue,
    gamma: spec.gamma,
    actionNames: GRID_ACTION_NAMES,
    grid: { width, height, kinds, actionVectors: GRID_ACTIONS },
  }
}

/** Options for `gridworld`. */
export interface GridworldOptions {
  width?: number
  height?: number
  /** Wall cells as [x, y]. */
  walls?: readonly (readonly [number, number])[]
  /** Terminal cells with their exit values (the reward for reaching them). */
  terminals?: readonly { x: number; y: number; value: number }[]
  /** Probability that a move goes to one of the two perpendicular directions instead (split equally). Default 0.2. */
  noise?: number
  /** Reward for every move. Default 0. */
  stepReward?: number
  gamma?: number
  start?: readonly [number, number]
}

/**
 * The stochastic gridworld of Russell and Norvig (2021, "Artificial Intelligence: A Modern Approach", §17.1) as the
 * site uses it: a 4 × 3 grid with a wall at (1, 1), exits worth +1 at (3, 2) and −1 at (3, 1), moves that slip
 * sideways with probability `noise`, and a step reward. Terminal cells hold their exit value.
 */
export function gridworld(options: GridworldOptions = {}): TabularMdp {
  const { width = 4, height = 3, noise = 0.2, stepReward = 0, gamma = 0.9 } = options
  const walls = options.walls ?? [[1, 1]]
  const terminals = options.terminals ?? [
    { x: 3, y: 2, value: 1 },
    { x: 3, y: 1, value: -1 },
  ]
  const kinds: CellKind[] = Array(width * height).fill('open')
  for (const [x, y] of walls) kinds[cellState(width, x, y)] = 'wall'
  const values = new Map<number, number>()
  for (const t of terminals) {
    const s = cellState(width, t.x, t.y)
    kinds[s] = t.value > 0 ? 'goal' : t.value < 0 ? 'trap' : 'terminal'
    values.set(s, t.value)
  }
  const [sx, sy] = options.start ?? [0, 0]
  return buildGrid({
    name: 'gridworld',
    width,
    height,
    kinds,
    start: cellState(width, sx, sy),
    gamma,
    slip: noise,
    land: (cell) => ({ next: cell, reward: stepReward }),
    terminalValue: (s) => values.get(s) ?? 0,
    terminalKinds: ['goal', 'trap', 'terminal'],
  })
}

/**
 * Cliff walking (Sutton and Barto, 2018, "Reinforcement Learning: An Introduction", Example 6.6): a 12 × 4 grid, start
 * at the bottom-left, goal at the bottom-right, and the cells between them a cliff. Every step pays −1; stepping into
 * the cliff pays −100 and returns the agent to the start. Moves are deterministic.
 */
export function cliffWalking({
  width = 12,
  height = 4,
  gamma = 1,
}: { width?: number; height?: number; gamma?: number } = {}): TabularMdp {
  const kinds: CellKind[] = Array(width * height).fill('open')
  const start = 0
  const goal = width - 1
  for (let x = 1; x < width - 1; x++) kinds[x] = 'cliff'
  kinds[start] = 'start'
  kinds[goal] = 'goal'
  return buildGrid({
    name: 'cliff walking',
    width,
    height,
    kinds,
    start,
    gamma,
    slip: 0,
    land: (cell) => (kinds[cell] === 'cliff' ? { next: start, reward: -100 } : { next: cell, reward: -1 }),
    terminalKinds: ['goal'],
  })
}

/** Options for `maze`. */
export interface MazeOptions {
  /** Probability of slipping to a perpendicular direction (split equally). Default 0. */
  slip?: number
  /** Rewards for a step, reaching the goal and falling into a trap. Defaults −1, 10 and −20. */
  rewards?: { step?: number; goal?: number; trap?: number }
  gamma?: number
}

/**
 * A maze from rows of text, top row first: `#` wall, `.` open, `S` start, `G` goal, `T` trap. Reaching the goal pays
 * the goal reward and ends the episode; a trap pays the trap reward and sends the agent back to the start; every other
 * move pays the step reward.
 */
export function maze(rows: readonly string[], options: MazeOptions = {}): TabularMdp {
  const { slip = 0, gamma = 0.95 } = options
  const rewards = { step: -1, goal: 10, trap: -20, ...options.rewards }
  const height = rows.length
  const width = rows[0].length
  const kinds: CellKind[] = Array(width * height).fill('open')
  let start = 0
  rows.forEach((row, i) => {
    const y = height - 1 - i
    ;[...row].forEach((ch, x) => {
      const s = cellState(width, x, y)
      if (ch === '#') kinds[s] = 'wall'
      else if (ch === 'T') kinds[s] = 'trap'
      else if (ch === 'G') kinds[s] = 'goal'
      else if (ch === 'S') {
        kinds[s] = 'start'
        start = s
      }
    })
  })
  return buildGrid({
    name: 'maze',
    width,
    height,
    kinds,
    start,
    gamma,
    slip,
    land: (cell) =>
      kinds[cell] === 'goal'
        ? { next: cell, reward: rewards.goal }
        : kinds[cell] === 'trap'
          ? { next: start, reward: rewards.trap }
          : { next: cell, reward: rewards.step },
    terminalKinds: ['goal'],
  })
}

/** The FrozenLake maps of Gymnasium. */
export const FROZEN_LAKE_MAPS: Record<'4x4' | '8x8', readonly string[]> = {
  '4x4': ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  '8x8': ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF', 'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG'],
}

/**
 * FrozenLake (Gymnasium's `FrozenLake-v1`): cross a frozen lake from S to G without falling into a hole H. On slippery
 * ice the agent moves in the intended direction or either perpendicular one with probability 1/3 each. Reaching G pays
 * 1; holes end the episode with nothing. Map rows are given top row first; actions follow this module's order (up,
 * right, down, left), not Gymnasium's.
 */
export function frozenLake({
  map = '4x4',
  slippery = true,
  gamma = 0.99,
}: { map?: '4x4' | '8x8' | readonly string[]; slippery?: boolean; gamma?: number } = {}): TabularMdp {
  const rows = typeof map === 'string' ? FROZEN_LAKE_MAPS[map] : map
  const height = rows.length
  const width = rows[0].length
  const kinds: CellKind[] = Array(width * height).fill('open')
  let start = 0
  rows.forEach((row, i) =>
    [...row].forEach((ch, x) => {
      const s = cellState(width, x, height - 1 - i)
      if (ch === 'H') kinds[s] = 'hole'
      else if (ch === 'G') kinds[s] = 'goal'
      else if (ch === 'S') {
        kinds[s] = 'start'
        start = s
      }
    }),
  )
  return buildGrid({
    name: 'FrozenLake',
    width,
    height,
    kinds,
    start,
    gamma,
    slip: slippery ? 2 / 3 : 0,
    land: (cell) => ({ next: cell, reward: kinds[cell] === 'goal' ? 1 : 0 }),
    terminalKinds: ['goal', 'hole'],
  })
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
