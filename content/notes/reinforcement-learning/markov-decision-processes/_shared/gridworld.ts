/**
 * A small gridworld MDP and exact dynamic programming on it: value iteration and policy iteration.
 *
 * Cells are indexed s = y · width + x with y = 0 at the bottom row. Terminal cells pay their reward and end the
 * episode, so their value is fixed at that reward. Every other move pays `stepReward`. An action moves in its
 * intended direction with probability 1 − noise and to each perpendicular direction with probability noise / 2;
 * a move into a wall or off the grid leaves the agent where it is.
 */

export type Vec2 = [number, number]

export type GridSpec = {
  width: number
  height: number
  walls: Set<number>
  /** Terminal cells and their rewards. */
  terminals: Map<number, number>
  gamma: number
  noise: number
  stepReward: number
}

/** Up, right, down, left, as (dx, dy) with y pointing up. */
export const ACTIONS: Vec2[] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]

export const cellIndex = (spec: Pick<GridSpec, 'width'>, x: number, y: number) => y * spec.width + x
export const cellXY = (spec: Pick<GridSpec, 'width'>, s: number): Vec2 => [s % spec.width, Math.floor(s / spec.width)]

/** Cell reached by moving from s in direction a, or s itself if that cell is a wall or off the grid. */
function move(spec: GridSpec, s: number, a: number): number {
  const [x, y] = cellXY(spec, s)
  const nx = x + ACTIONS[a][0]
  const ny = y + ACTIONS[a][1]
  if (nx < 0 || ny < 0 || nx >= spec.width || ny >= spec.height) return s
  const t = cellIndex(spec, nx, ny)
  return spec.walls.has(t) ? s : t
}

/** Outcomes of taking action a in cell s: (probability, next cell) pairs. */
export function transitions(spec: GridSpec, s: number, a: number): [number, number][] {
  const side = spec.noise / 2
  return [
    [1 - spec.noise, move(spec, s, a)],
    [side, move(spec, s, (a + 1) % 4)],
    [side, move(spec, s, (a + 3) % 4)],
  ]
}

export const isActive = (spec: GridSpec, s: number) => !spec.walls.has(s) && !spec.terminals.has(s)

/** Action value Q(s, a) = Σ p(s' | s, a) [r + γ V(s')] under a value table V. */
export function qValue(spec: GridSpec, V: number[], s: number, a: number): number {
  return transitions(spec, s, a).reduce((acc, [p, t]) => acc + p * (spec.stepReward + spec.gamma * V[t]), 0)
}

/** The value table everything starts from: 0 everywhere except terminal cells, which hold their reward. */
export function initialValues(spec: GridSpec): number[] {
  return Array.from({ length: spec.width * spec.height }, (_, s) => spec.terminals.get(s) ?? 0)
}

/** Greedy action in every active cell (−1 elsewhere). Ties go to the lowest action index, or to `prefer` if tied. */
export function greedy(spec: GridSpec, V: number[], prefer?: number[]): number[] {
  return V.map((_, s) => {
    if (!isActive(spec, s)) return -1
    const q = ACTIONS.map((_, a) => qValue(spec, V, s, a))
    const best = Math.max(...q)
    const keep = prefer?.[s]
    if (keep !== undefined && keep >= 0 && q[keep] >= best - 1e-9) return keep
    return q.findIndex((v) => v >= best - 1e-9)
  })
}

export type Sweep = { V: number[]; policy: number[]; residual: number }

/** Value iteration from V₀: each sweep applies the Bellman optimality operator to every active cell at once. */
export function valueIteration(spec: GridSpec, maxSweeps: number): Sweep[] {
  let V = initialValues(spec)
  const out: Sweep[] = [{ V, policy: greedy(spec, V), residual: NaN }]
  for (let k = 1; k <= maxSweeps; k++) {
    const next = V.map((v, s) => (isActive(spec, s) ? Math.max(...ACTIONS.map((_, a) => qValue(spec, V, s, a))) : v))
    const residual = Math.max(...next.map((v, s) => Math.abs(v - V[s])))
    V = next
    out.push({ V, policy: greedy(spec, V), residual })
    if (residual < 1e-10) break
  }
  return out
}

/** Exact value of a deterministic policy: solve (I − γ P_π) v = r_π over the active cells by Gaussian elimination. */
export function evaluatePolicy(spec: GridSpec, policy: number[]): number[] {
  const V = initialValues(spec)
  const active = V.map((_, s) => s).filter((s) => isActive(spec, s))
  const row = new Map(active.map((s, i) => [s, i]))
  const n = active.length
  const A = active.map(() => new Array<number>(n + 1).fill(0))
  active.forEach((s, i) => {
    A[i][i] += 1
    for (const [p, t] of transitions(spec, s, policy[s])) {
      A[i][n] += p * spec.stepReward
      const j = row.get(t)
      if (j === undefined) A[i][n] += p * spec.gamma * V[t]
      else A[i][j] -= p * spec.gamma
    }
  })
  for (let c = 0; c < n; c++) {
    let pivot = c
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[pivot][c])) pivot = r
    ;[A[c], A[pivot]] = [A[pivot], A[c]]
    for (let r = 0; r < n; r++) {
      if (r === c || A[r][c] === 0) continue
      const f = A[r][c] / A[c][c]
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]
    }
  }
  active.forEach((s, i) => (V[s] = A[i][n] / A[i][i]))
  return V
}

/** Policy iteration from "always up": evaluate exactly, improve greedily (keeping the old action on ties), repeat. */
export function policyIteration(spec: GridSpec, maxIterations: number): Sweep[] {
  let policy: number[] = initialValues(spec).map((_, s) => (isActive(spec, s) ? 0 : -1))
  let V = evaluatePolicy(spec, policy)
  const out: Sweep[] = [{ V, policy, residual: NaN }]
  for (let k = 1; k <= maxIterations; k++) {
    const next = greedy(spec, V, policy)
    const changed = next.filter((a, s) => a !== policy[s]).length
    if (changed === 0) break
    const nextV = evaluatePolicy(spec, next)
    const residual = Math.max(...nextV.map((v, s) => Math.abs(v - V[s])))
    policy = next
    V = nextV
    out.push({ V, policy, residual })
  }
  return out
}
