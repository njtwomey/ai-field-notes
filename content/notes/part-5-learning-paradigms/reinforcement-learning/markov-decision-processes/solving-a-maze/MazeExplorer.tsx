import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  setting,
  slider,
  useAxis,
  useFigureState,
  Vectors,
  type SeriesSpec,
} from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'
import {
  MOVES,
  REWARD,
  createAgent,
  greedyAction,
  greedyPath,
  isOpen,
  maxQ,
  parseMaze,
  valueIteration,
  xyOf,
  type Episode,
  type Maze,
  type Method,
} from '../_shared/maze'
import { MAZES, type MazeName } from './mazes'

const MAX_EPISODES = 300
const MAX_STEPS = 1000
const MAX_SWEEPS = 300
const SMOOTH = 10

type Mode = Method | 'planner'
const MODES: { value: Mode; label: string }[] = [
  { value: 'q-learning', label: 'Q-learning' },
  { value: 'sarsa', label: 'SARSA' },
  { value: 'planner', label: 'value iteration' },
]
const MAZE_OPTIONS = (Object.keys(MAZES) as MazeName[]).map((value) => ({ value, label: MAZES[value].label }))
const presetWalls = (name: MazeName) => [...parseMaze([...MAZES[name].rows]).walls]

/**
 * The whole run, computed up front: learning keeps Q after every episode (snapshot e is Q after e episodes) and the
 * episodes themselves; planning keeps every value-iteration sweep.
 */
type Trace =
  | { kind: 'learn'; snapshots: Float32Array[]; episodes: Episode[] }
  | { kind: 'plan'; snapshots: Float32Array[]; residuals: number[] }

const learning = (v: Readonly<Record<string, unknown>>) => v.method !== 'planner'

/** Is the goal reachable from the start through open cells and traps (a trap returns the agent, but can be crossed)? */
function connected(m: Maze, walls: Set<number>): boolean {
  const seen = new Set([m.start])
  const queue = [m.start]
  while (queue.length) {
    const s = queue.pop()!
    const [x, y] = xyOf(m, s)
    for (const [dx, dy] of MOVES) {
      const nx = x + dx
      const ny = y + dy
      const t = ny * m.width + nx
      if (nx < 0 || ny < 0 || nx >= m.width || ny >= m.height || walls.has(t) || seen.has(t)) continue
      if (t === m.goal) return true
      seen.add(t)
      if (!m.traps.has(t)) queue.push(t)
    }
  }
  return false
}

/** Hatching that fills each cell, drawn as one line series with NaN breaks. */
function hatch(m: Maze, cells: number[], shape: 'fill' | 'cross') {
  const xs: number[] = []
  const ys: number[] = []
  for (const s of cells) {
    const [x, y] = xyOf(m, s)
    if (shape === 'cross') {
      xs.push(x - 0.3, x + 0.3, NaN, x - 0.3, x + 0.3, NaN)
      ys.push(y - 0.3, y + 0.3, NaN, y + 0.3, y - 0.3, NaN)
    } else {
      for (let k = -4; k <= 4; k++) {
        xs.push(x - 0.46, x + 0.46, NaN)
        ys.push(y + k * 0.1, y + k * 0.1, NaN)
      }
    }
  }
  return { x: xs, y: ys }
}

/** Cells of a path as a line, broken where a trap sends the agent back to the start (marked −1). */
function pathLine(m: Maze, path: ArrayLike<number>) {
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < path.length; i++) {
    if (path[i] < 0) {
      xs.push(NaN)
      ys.push(NaN)
      continue
    }
    const [x, y] = xyOf(m, path[i])
    xs.push(x)
    ys.push(y)
  }
  return { x: xs, y: ys }
}

const movingAverage = (xs: number[]) => {
  let sum = 0
  return xs.map((v, i) => {
    sum += v - (i >= SMOOTH ? xs[i - SMOOTH] : 0)
    return sum / Math.min(i + 1, SMOOTH)
  })
}

/**
 * A maze solved by Q-learning, SARSA or value iteration. The raster is max_a Q(s, a) with the greedy policy as arrows;
 * the chart below is the return (or length) of every episode. The player walks through the episodes (or sweeps) and
 * shows the values, policy and path as they were after each one; the chart's handle moves the same position.
 */
export function MazeExplorer() {
  const state = useFigureState({
    maze: choice<MazeName>(MAZE_OPTIONS, 'routes', { label: 'maze' }),
    method: choice<Mode>(MODES, 'q-learning', { label: 'method' }),
    gamma: slider(0.8, 0.99, 0.95, { step: 0.01, label: 'discount γ' }),
    slip: slider(0, 0.3, 0, { step: 0.01, label: 'slip probability' }),
    alpha: float(0.5, {
      gt: 0,
      max: 1,
      scale: 'log10',
      suggestions: [0.1, 0.25, 0.5, 1],
      label: 'step size α',
      when: learning,
    }),
    epsilon: slider(0, 0.5, 0.1, { step: 0.01, label: 'exploration ε', when: learning }),
    decay: setting(false, { label: 'decay ε as ε / (1 + e/10)', when: learning }),
    seed: int(1, { ge: 0, label: 'seed', when: learning }),
    metric: choice<'return' | 'steps'>(
      [
        { value: 'return', label: 'return' },
        { value: 'steps', label: 'steps' },
      ],
      'return',
      { label: 'chart', when: learning },
    ),
  })
  const mode = state.method
  const isLearning = mode !== 'planner'

  // Clicked walls belong to their maze; another maze opens with its preset walls.
  const [layout, setLayout] = useState(() => ({ name: state.maze, walls: presetWalls(state.maze) }))
  const walls = layout.name === state.maze ? layout.walls : presetWalls(state.maze)
  const wallKey = walls.join(',')
  const maze = useMemo((): Maze => {
    const m = parseMaze([...MAZES[state.maze].rows])
    return { ...m, walls: new Set(wallKey ? wallKey.split(',').map(Number) : []) }
  }, [state.maze, wallKey])

  // The optimal action values, for the colour range, the readout and the reference line.
  const optimal = useMemo(
    () => valueIteration(maze, state.gamma, state.slip, 3000, 1e-9).at(-1)!.Q,
    [maze, state.gamma, state.slip],
  )
  const vStar = maxQ(optimal, maze.start)
  const optimalPath = useMemo(() => greedyPath(maze, optimal), [maze, optimal])
  const range = useMemo((): [number, number] => {
    let lo = REWARD.goal
    for (let s = 0; s < maze.width * maze.height; s++) if (isOpen(maze, s)) lo = Math.min(lo, maxQ(optimal, s))
    return [Math.floor(lo), REWARD.goal]
  }, [maze, optimal])

  // Any change of maze or setting computes a new run from Q = 0.
  const { alpha, epsilon, decay, seed, gamma, slip } = state
  const trace = useMemo((): Trace => {
    if (mode === 'planner') {
      const sweeps = valueIteration(maze, gamma, slip, MAX_SWEEPS)
      return { kind: 'plan', snapshots: sweeps.map((w) => w.Q), residuals: sweeps.map((w) => w.residual) }
    }
    const g = stream(seed)
    const agent = createAgent(
      maze,
      { method: mode, alpha, gamma, epsilon, decay, slip },
      () => uniform(g),
      MAX_EPISODES,
      MAX_STEPS,
    )
    while (!agent.finished()) agent.step()
    return { kind: 'learn', snapshots: agent.snapshots, episodes: agent.episodes }
  }, [maze, mode, alpha, gamma, epsilon, decay, slip, seed])
  const total = trace.snapshots.length - 1

  // The walk-through restarts at 0 for a new run.
  const [position, setPosition] = useState({ trace, k: 0 })
  const shown = position.trace === trace ? Math.min(position.k, total) : 0
  const go = (v: number) => setPosition({ trace, k: Math.min(total, Math.max(0, Math.round(v))) })

  const toggleWall = ([x, y]: [number, number]) => {
    const cx = Math.round(x)
    const cy = Math.round(y)
    if (cx < 0 || cy < 0 || cx >= maze.width || cy >= maze.height) return
    const s = cy * maze.width + cx
    if (s === maze.start || s === maze.goal || maze.traps.has(s)) return
    const next = new Set(maze.walls)
    if (next.has(s)) next.delete(s)
    else next.add(s)
    // A wall that cuts the start off from the goal is refused.
    if (!connected(maze, next)) return
    setLayout({ name: state.maze, walls: [...next].sort((a, b) => a - b) })
  }

  const Q = trace.snapshots[shown]
  const xs = useMemo(() => Array.from({ length: maze.width }, (_, i) => i), [maze.width])
  const ys = useMemo(() => Array.from({ length: maze.height }, (_, i) => i), [maze.height])
  const [lo, hi] = range
  const z = useMemo(
    () =>
      ys.map((y) =>
        xs.map((x) => {
          const s = y * maze.width + x
          if (s === maze.goal) return hi
          if (!isOpen(maze, s)) return lo
          return Math.min(hi, Math.max(lo, maxQ(Q, s)))
        }),
      ),
    [xs, ys, maze, Q, lo, hi],
  )
  const vectors = useMemo(
    () =>
      xs.flatMap((x) =>
        ys.flatMap((y) => {
          const s = y * maze.width + x
          if (!isOpen(maze, s)) return []
          const q = [Q[4 * s], Q[4 * s + 1], Q[4 * s + 2], Q[4 * s + 3]]
          // No arrow until the cell's actions differ: an untouched cell has no preference.
          if (q.every((v) => v === q[0])) return []
          const [dx, dy] = MOVES[greedyAction(Q, s)]
          return [
            {
              from: [x - 0.25 * dx, y - 0.25 * dy] as [number, number],
              to: [x + 0.3 * dx, y + 0.3 * dy] as [number, number],
            },
          ]
        }),
      ),
    [xs, ys, maze, Q],
  )

  const path = useMemo(() => {
    if (trace.kind === 'plan') return greedyPath(maze, Q)
    return shown > 0 ? trace.episodes[shown - 1].path : []
  }, [trace, maze, Q, shown])

  const wallHatch = useMemo(() => hatch(maze, [...maze.walls], 'fill'), [maze])
  const traps = useMemo(() => hatch(maze, [...maze.traps], 'cross'), [maze])
  const line = useMemo(() => pathLine(maze, path), [maze, path])
  const [sx, sy] = xyOf(maze, maze.start)
  const [gx, gy] = xyOf(maze, maze.goal)
  const overlay = useMemo(
    (): SeriesSpec[] => [
      { name: 'wall', type: 'line', ...wallHatch, emphasis: true },
      { name: trace.kind === 'plan' ? 'greedy route' : 'episode path', type: 'line', ...line, slot: 1 },
      { name: 'trap −20', type: 'line', ...traps, slot: 7 },
      { name: 'start', type: 'scatter', x: [sx], y: [sy], slot: 2 },
      { name: 'goal +10', type: 'scatter', x: [gx], y: [gy], slot: 3 },
    ],
    [wallHatch, traps, line, sx, sy, gx, gy, trace.kind],
  )

  // The chart: one point per episode (or sweep) up to the one shown.
  const metric = state.metric
  const series = useMemo((): SeriesSpec[] => {
    const units = Array.from({ length: shown }, (_, i) => i + 1)
    if (trace.kind === 'plan') {
      const ks = [0, ...units]
      return [
        { name: 'V_k(start)', type: 'line', x: ks, y: ks.map((k) => maxQ(trace.snapshots[k], maze.start)), slot: 0 },
        { name: 'v*(start)', type: 'line', x: [0, Math.max(10, total)], y: [vStar, vStar], dashed: true, slot: 1 },
      ]
    }
    const values = trace.episodes.slice(0, shown).map((e) => (metric === 'return' ? e.ret : e.steps))
    const out: SeriesSpec[] = [
      {
        name: `${metric === 'return' ? 'return' : 'steps'} per episode`,
        type: 'line',
        x: units,
        y: values,
        muted: true,
      },
      { name: `moving average (${SMOOTH} episodes)`, type: 'line', x: units, y: movingAverage(values), slot: 0 },
    ]
    // With deterministic moves the shortest route has a known return and length.
    if (slip === 0 && optimalPath.at(-1) === maze.goal) {
      const n = optimalPath.length - 1
      const best = metric === 'return' ? REWARD.goal + (n - 1) * REWARD.step : n
      out.push({
        name: 'shortest route',
        type: 'line',
        x: [1, Math.max(20, total)],
        y: [best, best],
        dashed: true,
        slot: 1,
      })
    }
    return out
  }, [trace, shown, total, metric, maze, vStar, slip, optimalPath])

  const unit = trace.kind === 'plan' ? 'sweep' : 'episode'
  const episode = trace.kind === 'learn' && shown > 0 ? trace.episodes[shown - 1] : undefined
  const residual = trace.kind === 'plan' && shown > 0 ? trace.residuals[shown] : undefined
  const heatHeight = Math.round(Math.min(480, Math.max(260, (maze.height / maze.width) * 560 + 60)))
  const logSteps = isLearning && metric === 'steps'

  const gridX = useAxis({ label: 'x', range: [-0.5, maze.width - 0.5], nice: false, key: state.maze })
  const gridY = useAxis({ label: 'y', range: [-0.5, maze.height - 0.5], nice: false, equal: gridX, key: state.maze })
  const chartX = useAxis({ label: unit, range: [0, Math.max(trace.kind === 'plan' ? 10 : 20, total)], integer: true })
  const chartY = useAxis({
    label: trace.kind === 'plan' ? 'value of the start' : metric === 'return' ? 'return' : 'steps',
    hold: 'union',
    log: logSteps,
    key: `${mode}:${metric}:${state.maze}`,
  })

  return (
    <Figure
      title="Solving a maze"
      state={state}
      caption={
        <>
          Colour is the value max<sub>a</sub> Q(s, a) of each cell and arrows are the greedy policy; cells without an
          arrow have not been updated yet. Each move pays −1, entering the goal pays +10 and ends the episode, and
          entering a trap (red cross) pays −20 and returns the agent to the start. With slip, a move goes sideways with
          that probability. Press play to walk through the episodes; the orange line is the path of the episode shown.
          The chart shows the return (or number of steps) of every episode so far; drag its line, or step the player, to
          go back and see the values, policy and path as they were after that episode. In value-iteration mode the model
          is known and each position is one sweep of the Bellman optimality backup. Click a cell to add or remove a
          wall; changing the maze or any setting computes a new run from Q = 0.
        </>
      }
      controls={
        <>
          <Player value={shown} onChange={go} count={total + 1} label={unit} format={(v) => `${unit} ${v}`} />
          <Button
            variant="outline"
            size="sm"
            onClick={() => setLayout({ name: state.maze, walls: presetWalls(state.maze) })}
          >
            Reset walls
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label={`${unit}s run`} value={String(total)} />
          <Readout label="shown" value={`${unit} ${shown}`} />
          {isLearning && <Readout label="return" value={episode ? formatNumber(episode.ret) : '–'} />}
          {isLearning && <Readout label="steps" value={episode ? String(episode.steps) : '–'} />}
          {isLearning && <Readout label="ε" value={episode ? formatNumber(episode.epsilon) : formatNumber(epsilon)} />}
          {!isLearning && (
            <Readout label="largest change" value={residual === undefined ? '–' : formatNumber(residual)} />
          )}
          <Readout label="max Q(start, a)" value={formatNumber(maxQ(Q, maze.start))} />
          <Readout label="v*(start) by value iteration" value={formatNumber(vStar)} />
        </>
      }
    >
      <Plot
        x={gridX}
        y={gridY}
        height={heatHeight}
        onPlotClick={toggleWall}
        ariaLabel="Maze with the learned value of each cell, the greedy policy as arrows and the agent's path"
      >
        <Raster x={xs} y={ys} z={z} range={range} valueLabel="max Q" />
        {seriesLayers(overlay, { live: true })}
        <Vectors vectors={vectors} />
      </Plot>
      <Plot x={chartX} y={chartY} height={240} ariaLabel={`Return of every ${unit}, with a handle to go back in time`}>
        {seriesLayers(series)}
        <Handle kind="x" at={shown} label={String(shown)} onDrag={go} />
      </Plot>
    </Figure>
  )
}
