/**
 * A maze as a Markov decision process, solved two ways: by value iteration on the known model (planning) and by
 * Q-learning or SARSA from sampled steps (learning).
 *
 * Cells are indexed s = y · width + x with y = 0 at the bottom row. The agent acts in every open cell. A move goes in
 * its intended direction with probability 1 − slip and to each perpendicular direction with probability slip / 2; a
 * move into a wall or off the grid leaves the agent where it is. Entering the goal pays REWARD.goal and ends the
 * episode. Entering a trap pays REWARD.trap and sends the agent back to the start. Every other move pays REWARD.step.
 *
 * The module has no imports, so it runs unchanged under Node for checking the numbers quoted in the note. Randomness
 * comes from the caller as a uniform sampler on [0, 1).
 */

export type Maze = {
  width: number
  height: number
  walls: Set<number>
  traps: Set<number>
  start: number
  goal: number
}

/** Up, right, down, left, as (dx, dy) with y pointing up. */
export const MOVES: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]
export const N_ACTIONS = 4

export const REWARD = { step: -1, goal: 10, trap: -20 }

export const cellOf = (m: Pick<Maze, 'width'>, x: number, y: number) => y * m.width + x
export const xyOf = (m: Pick<Maze, 'width'>, s: number): [number, number] => [s % m.width, Math.floor(s / m.width)]

/** A maze from rows of text, top row first: `#` wall, `.` open, `S` start, `G` goal, `T` trap. */
export function parseMaze(rows: string[]): Maze {
  const height = rows.length
  const width = rows[0].length
  const m: Maze = { width, height, walls: new Set(), traps: new Set(), start: 0, goal: 0 }
  rows.forEach((row, i) => {
    const y = height - 1 - i
    ;[...row].forEach((ch, x) => {
      const s = cellOf(m, x, y)
      if (ch === '#') m.walls.add(s)
      else if (ch === 'T') m.traps.add(s)
      else if (ch === 'S') m.start = s
      else if (ch === 'G') m.goal = s
    })
  })
  return m
}

/** Cells where the agent chooses an action: not a wall, not a trap, not the goal. */
export const isOpen = (m: Maze, s: number) => !m.walls.has(s) && !m.traps.has(s) && s !== m.goal

/** Cell reached by moving from s in direction a, or s itself if that cell is a wall or off the grid. */
function move(m: Maze, s: number, a: number): number {
  const [x, y] = xyOf(m, s)
  const nx = x + MOVES[a][0]
  const ny = y + MOVES[a][1]
  if (nx < 0 || ny < 0 || nx >= m.width || ny >= m.height) return s
  const t = cellOf(m, nx, ny)
  return m.walls.has(t) ? s : t
}

export type Outcome = { p: number; next: number; reward: number; done: boolean }

/** What landing in a cell means: the goal ends the episode, a trap returns the agent to the start. */
function land(m: Maze, t: number): Omit<Outcome, 'p'> {
  if (t === m.goal) return { next: t, reward: REWARD.goal, done: true }
  if (m.traps.has(t)) return { next: m.start, reward: REWARD.trap, done: false }
  return { next: t, reward: REWARD.step, done: false }
}

/** The outcomes of action a in cell s, with their probabilities. */
export function outcomes(m: Maze, s: number, a: number, slip: number): Outcome[] {
  const main = { p: 1 - slip, ...land(m, move(m, s, a)) }
  if (slip === 0) return [main]
  return [
    main,
    { p: slip / 2, ...land(m, move(m, s, (a + 1) % 4)) },
    { p: slip / 2, ...land(m, move(m, s, (a + 3) % 4)) },
  ]
}

/** Samples one step: the intended move, or a perpendicular slip. Also returns the cell landed in (a trap, say). */
export function sampleStep(m: Maze, s: number, a: number, slip: number, uniform: () => number) {
  const u = uniform()
  const dir = u < 1 - slip ? a : u < 1 - slip / 2 ? (a + 1) % 4 : (a + 3) % 4
  const cell = move(m, s, dir)
  return { cell, ...land(m, cell) }
}

export const maxQ = (Q: ArrayLike<number>, s: number) => Math.max(Q[4 * s], Q[4 * s + 1], Q[4 * s + 2], Q[4 * s + 3])

/** Greedy action in s; ties go to the lowest action index. */
export function greedyAction(Q: ArrayLike<number>, s: number): number {
  let best = 0
  for (let a = 1; a < N_ACTIONS; a++) if (Q[4 * s + a] > Q[4 * s + best]) best = a
  return best
}

/** The state values V(s) = max_a Q(s, a) of every cell (0 at walls, traps and the goal). */
export function stateValues(m: Maze, Q: ArrayLike<number>): number[] {
  return Array.from({ length: m.width * m.height }, (_, s) => (isOpen(m, s) ? maxQ(Q, s) : 0))
}

/** One synchronous sweep of the Bellman optimality backup on action values. */
export function backup(m: Maze, Q: Float32Array, gamma: number, slip: number): Float32Array {
  const next = new Float32Array(Q.length)
  for (let s = 0; s < m.width * m.height; s++) {
    if (!isOpen(m, s)) continue
    for (let a = 0; a < N_ACTIONS; a++) {
      let q = 0
      for (const o of outcomes(m, s, a, slip)) q += o.p * (o.reward + (o.done ? 0 : gamma * maxQ(Q, o.next)))
      next[4 * s + a] = q
    }
  }
  return next
}

export type Sweep = { Q: Float32Array; residual: number }

/** Value iteration on action values from Q₀ = 0, until the largest change is below `tol` or `maxSweeps` is reached. */
export function valueIteration(m: Maze, gamma: number, slip: number, maxSweeps: number, tol = 1e-4): Sweep[] {
  const out: Sweep[] = [{ Q: new Float32Array(4 * m.width * m.height), residual: NaN }]
  for (let k = 1; k <= maxSweeps; k++) {
    const prev = out[k - 1].Q
    const Q = backup(m, prev, gamma, slip)
    let residual = 0
    for (let s = 0; s < m.width * m.height; s++)
      if (isOpen(m, s)) residual = Math.max(residual, Math.abs(maxQ(Q, s) - maxQ(prev, s)))
    out.push({ Q, residual })
    if (residual < tol) break
  }
  return out
}

/**
 * The route the greedy policy intends from the start: follow the greedy action's intended move until the goal, a
 * trap, a repeated cell or `maxLength` cells.
 */
export function greedyPath(m: Maze, Q: ArrayLike<number>, maxLength = 4 * m.width * m.height): number[] {
  const path = [m.start]
  const seen = new Set(path)
  let s = m.start
  while (isOpen(m, s) && path.length < maxLength) {
    const t = move(m, s, greedyAction(Q, s))
    if (seen.has(t)) break
    path.push(t)
    seen.add(t)
    s = t
  }
  return path
}

export type Method = 'q-learning' | 'sarsa'

export type LearnSettings = {
  method: Method
  /** Step size α. */
  alpha: number
  gamma: number
  /** Exploration rate ε at the first episode. */
  epsilon: number
  /** Decay ε as ε / (1 + e / 10) at episode e, so exploration fades out. */
  decay: boolean
  slip: number
}

export type Episode = {
  /** Undiscounted sum of rewards. */
  ret: number
  steps: number
  epsilon: number
  /** Cells visited in order; −1 marks a return to the start from a trap. */
  path: Int16Array
}

/**
 * A learning agent that advances one environment step at a time, so a widget can animate it. After each episode it
 * stores a copy of Q, so `snapshots[e]` is Q after e episodes (`snapshots[0]` is the all-zero start).
 */
export type Agent = {
  Q: Float32Array
  state: number
  episodes: Episode[]
  snapshots: Float32Array[]
  /** The episode in progress. */
  ret: number
  steps: number
  path: number[]
  epsilon: () => number
  /** One step of acting and learning. Returns true when the step ended an episode. */
  step: () => boolean
  finished: () => boolean
}

export function createAgent(
  m: Maze,
  settings: LearnSettings,
  uniform: () => number,
  maxEpisodes: number,
  maxSteps: number,
): Agent {
  const { method, alpha, gamma, slip } = settings
  const Q = new Float32Array(4 * m.width * m.height)
  const episodes: Episode[] = []
  const epsilon = () => (settings.decay ? settings.epsilon / (1 + episodes.length / 10) : settings.epsilon)

  // ε-greedy with random tie-breaking, so an untrained agent does not always walk the same way.
  const act = (s: number) => {
    if (uniform() < epsilon()) return Math.floor(uniform() * N_ACTIONS)
    const best = maxQ(Q, s)
    const ties = [0, 1, 2, 3].filter((a) => Q[4 * s + a] === best)
    return ties[Math.floor(uniform() * ties.length)]
  }

  let action = act(m.start)
  const agent: Agent = {
    Q,
    state: m.start,
    episodes,
    snapshots: [Q.slice()],
    ret: 0,
    steps: 0,
    path: [m.start],
    epsilon,
    finished: () => agent.episodes.length >= maxEpisodes,
    step: () => {
      if (agent.finished()) return false
      const s = agent.state
      const a = action
      const { cell, next, reward, done } = sampleStep(m, s, a, slip, uniform)
      const nextAction = done ? 0 : act(next)
      // Q-learning bootstraps from the greedy action in the next cell; SARSA from the action it will actually take.
      const bootstrap = done ? 0 : method === 'sarsa' ? Q[4 * next + nextAction] : maxQ(Q, next)
      Q[4 * s + a] += alpha * (reward + gamma * bootstrap - Q[4 * s + a])
      agent.ret += reward
      agent.steps += 1
      if (next !== cell) agent.path.push(cell, -1)
      agent.path.push(next)
      agent.state = next
      action = nextAction
      if (!done && agent.steps < maxSteps) return false
      agent.episodes.push({ ret: agent.ret, steps: agent.steps, epsilon: epsilon(), path: Int16Array.from(agent.path) })
      agent.snapshots.push(Q.slice())
      agent.state = m.start
      agent.ret = 0
      agent.steps = 0
      agent.path = [m.start]
      action = act(m.start)
      return true
    },
  }
  return agent
}
