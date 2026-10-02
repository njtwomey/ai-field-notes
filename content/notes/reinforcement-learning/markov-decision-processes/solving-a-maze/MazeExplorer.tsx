import { Pause, Play, RotateCcw, SkipForward, StepForward } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type HeatmapOverlay,
  type Param,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
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
  type Agent,
  type Maze,
  type Method,
  type Sweep,
} from '../_shared/maze'
import { MAZES, type MazeName } from './mazes'

const MAX_EPISODES = 300
const MAX_STEPS = 1000
const MAX_SWEEPS = 300
const SMOOTH = 10
/** Environment steps per second (learning) and sweeps per second (planning), indexed by the speed slider. */
const STEP_RATES = [2, 8, 30, 120, 500, 2000, 10000]
const SWEEP_RATES = [0.5, 1, 2, 4, 8, 15, 30]
/** While an episode runs, the value table and arrows refresh at most this often; the agent moves every frame. */
const REFRESH_MS = 200

type Mode = Method | 'planner'
const MODES: { value: Mode; label: string }[] = [
  { value: 'q-learning', label: 'Q-learning' },
  { value: 'sarsa', label: 'SARSA' },
  { value: 'planner', label: 'value iteration' },
]
const MAZE_OPTIONS = (Object.keys(MAZES) as MazeName[]).map((value) => ({ value, label: MAZES[value].label }))

type Layout = { name: MazeName; walls: number[] }
const presetLayout = (name: MazeName): Layout => ({ name, walls: [...parseMaze([...MAZES[name].rows]).walls] })

/** Learning runs an agent step by step; planning reveals precomputed value-iteration sweeps one at a time. */
/** `generation` counts resets, so a reset with unchanged settings still makes a new run. */
type Run = { generation: number; maze: Maze } & (
  { kind: 'learn'; agent: Agent } | { kind: 'plan'; sweeps: Sweep[]; shown: number }
)

const count = (run: Run) => (run.kind === 'learn' ? run.agent.episodes.length : run.shown)
const finished = (run: Run) => (run.kind === 'learn' ? run.agent.finished() : run.shown >= run.sweeps.length - 1)
const snapshot = (run: Run, e: number) => (run.kind === 'learn' ? run.agent.snapshots[e] : run.sweeps[e].Q)
const liveQ = (run: Run) => (run.kind === 'learn' ? run.agent.Q : run.sweeps[run.shown].Q)

/** Advance by n units: environment steps when learning, sweeps when planning. Returns true if an episode ended. */
function advance(run: Run, n: number): boolean {
  if (run.kind === 'plan') {
    run.shown = Math.min(run.sweeps.length - 1, run.shown + n)
    return n > 0
  }
  let ended = false
  for (let i = 0; i < n && !run.agent.finished(); i++) ended = run.agent.step() || ended
  return ended
}

/** What the page shows between animation frames: the agent's cell and a copy of Q, refreshed at most every REFRESH_MS. */
type Frame = { run: Run; cell: number; done: number; Q: Float32Array }

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
 * A maze solved by Q-learning, SARSA or value iteration, animated. The heatmap is max_a Q(s, a) with the greedy policy
 * as arrows; the chart below is the return (or length) of every episode. Its handle travels back to any earlier
 * episode and shows the values, policy and path as they were then.
 */
export function MazeExplorer() {
  const [layout, setLayout] = useState<Layout>(() => presetLayout('routes'))
  const [generation, setGeneration] = useState(0)
  const [mode, setMode] = useState<Mode>('q-learning')
  const [decay, setDecay] = useState(false)
  const [metric, setMetric] = useState<'return' | 'steps'>('return')
  const [playing, setPlaying] = useState(false)
  const alpha = useParam(0.5, { min: 0.05, max: 1, step: 0.05 })
  const gamma = useParam(0.95, { min: 0.8, max: 0.99, step: 0.01 })
  const epsilon = useParam(0.1, { min: 0, max: 0.5, step: 0.01 })
  const slip = useParam(0, { min: 0, max: 0.3, step: 0.01 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const speed = useParam(3, { min: 0, max: STEP_RATES.length - 1, step: 1 })

  const wallKey = layout.walls.join(',')
  const maze = useMemo(() => {
    const m = parseMaze([...MAZES[layout.name].rows])
    return { ...m, walls: new Set(wallKey ? wallKey.split(',').map(Number) : []) }
  }, [layout.name, wallKey])

  // The optimal action values, for the colour range, the readout and the reference line.
  const optimal = useMemo(
    () => valueIteration(maze, gamma.value, slip.value, 3000, 1e-9).at(-1)!.Q,
    [maze, gamma.value, slip.value],
  )
  const vStar = maxQ(optimal, maze.start)
  const optimalPath = useMemo(() => greedyPath(maze, optimal), [maze, optimal])
  const range = useMemo((): [number, number] => {
    let lo = REWARD.goal
    for (let s = 0; s < maze.width * maze.height; s++) if (isOpen(maze, s)) lo = Math.min(lo, maxQ(optimal, s))
    return [Math.floor(lo), REWARD.goal]
  }, [maze, optimal])

  // Any change of maze or setting starts a new run from Q = 0.
  const run = useMemo((): Run => {
    if (mode === 'planner')
      return {
        generation,
        kind: 'plan',
        maze,
        sweeps: valueIteration(maze, gamma.value, slip.value, MAX_SWEEPS),
        shown: 0,
      }
    const settings = {
      method: mode,
      alpha: alpha.value,
      gamma: gamma.value,
      epsilon: epsilon.value,
      decay,
      slip: slip.value,
    }
    return {
      generation,
      kind: 'learn',
      maze,
      agent: createAgent(maze, settings, rng(seed.value).uniform, MAX_EPISODES, MAX_STEPS),
    }
  }, [maze, mode, alpha.value, gamma.value, epsilon.value, decay, slip.value, seed.value, generation])

  const fresh = useMemo(
    (): Frame => ({
      run,
      cell: run.kind === 'learn' ? run.agent.state : run.maze.start,
      done: count(run),
      Q: liveQ(run).slice(),
    }),
    [run],
  )
  const [frame, setFrame] = useState<Frame>(fresh)
  const [travel, setTravel] = useState<{ run: Run; e: number } | null>(null)
  const current = frame.run === run ? frame : fresh
  const done = current.done
  const past = travel?.run === run ? travel.e : null
  const shown = past ?? done

  const sync = (force: boolean) =>
    setFrame((prev) => ({
      run,
      cell: run.kind === 'learn' ? run.agent.state : run.maze.start,
      done: count(run),
      Q: prev.run !== run || force || count(run) !== prev.done ? liveQ(run).slice() : prev.Q,
    }))

  // The animation loop: a budget of steps (or sweeps) per second, spent once per frame.
  const rate = (mode === 'planner' ? SWEEP_RATES : STEP_RATES)[speed.value]
  useEffect(() => {
    if (!playing) return
    let handle = 0
    let last = performance.now()
    let lastRefresh = last
    let budget = 0
    const loop = (now: number) => {
      budget += (Math.min(100, now - last) / 1000) * rate
      last = now
      const n = Math.floor(budget)
      budget -= n
      const ended = advance(run, n)
      const refresh = ended || now - lastRefresh > REFRESH_MS
      if (refresh) lastRefresh = now
      setFrame((prev) => ({
        run,
        cell: run.kind === 'learn' ? run.agent.state : run.maze.start,
        done: count(run),
        Q: refresh || prev.run !== run ? liveQ(run).slice() : prev.Q,
      }))
      if (finished(run)) setPlaying(false)
      else handle = requestAnimationFrame(loop)
    }
    handle = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(handle)
  }, [playing, run, rate])

  const resume = () => setTravel(null)
  const play = () => {
    resume()
    setPlaying(!playing && !finished(run))
  }
  const stepOnce = () => {
    resume()
    setPlaying(false)
    advance(run, 1)
    sync(true)
  }
  const finishEpisode = () => {
    resume()
    setPlaying(false)
    if (run.kind === 'learn') {
      const target = run.agent.episodes.length + 1
      while (!run.agent.finished() && run.agent.episodes.length < target) run.agent.step()
    }
    sync(true)
  }
  // Reset restores the preset's walls and starts learning again from Q = 0.
  const reset = () => {
    setPlaying(false)
    setTravel(null)
    setLayout(presetLayout(layout.name))
    setGeneration((g) => g + 1)
  }

  const travelParam: Param = {
    value: shown,
    min: 0,
    max: Math.max(1, done),
    step: 1,
    set: (v) => {
      if (!Number.isFinite(v)) return
      setPlaying(false)
      const e = Math.min(done, Math.max(0, Math.round(v)))
      setTravel(e >= done ? null : { run, e })
    },
  }

  const toggleWall = (x: number, y: number) => {
    const cx = Math.round(x)
    const cy = Math.round(y)
    if (cx < 0 || cy < 0 || cx >= maze.width || cy >= maze.height) return
    const s = cy * maze.width + cx
    if (s === maze.start || s === maze.goal || maze.traps.has(s)) return
    const walls = new Set(maze.walls)
    if (walls.has(s)) walls.delete(s)
    else walls.add(s)
    // A wall that cuts the start off from the goal is refused.
    if (!connected(maze, walls)) return
    setTravel(null)
    setLayout({ name: layout.name, walls: [...walls].sort((a, b) => a - b) })
  }
  const chooseMaze = (name: MazeName) => {
    setPlaying(false)
    setLayout(presetLayout(name))
  }

  // What is shown: the live values, or a snapshot from the past.
  const Q = past === null ? current.Q : snapshot(run, past)
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
    if (run.kind === 'plan') return greedyPath(maze, Q)
    const e = past ?? done
    return e > 0 ? run.agent.episodes[e - 1].path : []
  }, [run, maze, Q, past, done])

  const walls = useMemo(() => hatch(maze, [...maze.walls], 'fill'), [maze])
  const traps = useMemo(() => hatch(maze, [...maze.traps], 'cross'), [maze])
  const line = useMemo(() => pathLine(maze, path), [maze, path])
  const [sx, sy] = xyOf(maze, maze.start)
  const [gx, gy] = xyOf(maze, maze.goal)
  const overlay: HeatmapOverlay[] = useMemo(
    () => [
      { name: 'wall', type: 'line', ...walls, emphasis: true },
      { name: run.kind === 'plan' ? 'greedy route' : 'episode path', type: 'line', ...line, slot: 1 },
      { name: 'trap −20', type: 'line', ...traps, slot: 7 },
      { name: 'start', type: 'scatter', x: [sx], y: [sy], slot: 2 },
      { name: 'goal +10', type: 'scatter', x: [gx], y: [gy], slot: 3 },
    ],
    [walls, traps, line, sx, sy, gx, gy, run.kind],
  )
  const marker: [number, number] | undefined =
    run.kind === 'learn' && past === null ? xyOf(maze, current.cell) : undefined

  // The chart: one point per episode (or sweep), redrawn only when an episode ends.
  const series: XYSeries[] = useMemo(() => {
    const units = Array.from({ length: done }, (_, i) => i + 1)
    if (run.kind === 'plan') {
      const ks = [0, ...units]
      return [
        { name: 'V_k(start)', type: 'line', x: ks, y: ks.map((k) => maxQ(run.sweeps[k].Q, maze.start)), slot: 0 },
        { name: 'v*(start)', type: 'line', x: [0, Math.max(10, done)], y: [vStar, vStar], dashed: true, slot: 1 },
      ]
    }
    const episodes = run.agent.episodes.slice(0, done)
    const values = episodes.map((e) => (metric === 'return' ? e.ret : e.steps))
    const out: XYSeries[] = [
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
    if (slip.value === 0 && optimalPath.at(-1) === maze.goal) {
      const n = optimalPath.length - 1
      const best = metric === 'return' ? REWARD.goal + (n - 1) * REWARD.step : n
      out.push({
        name: 'shortest route',
        type: 'line',
        x: [1, Math.max(20, done)],
        y: [best, best],
        dashed: true,
        slot: 1,
      })
    }
    return out
  }, [run, done, metric, maze, vStar, slip.value, optimalPath])

  const unit = run.kind === 'plan' ? 'sweep' : 'episode'
  const episode = run.kind === 'learn' && shown > 0 ? run.agent.episodes[shown - 1] : undefined
  const residual = run.kind === 'plan' && shown > 0 ? run.sweeps[shown].residual : undefined
  const learning = run.kind === 'learn'
  const heatHeight = Math.round(Math.min(480, Math.max(260, (maze.height / maze.width) * 560 + 60)))

  return (
    <Interactive
      title="Solving a maze"
      caption={
        <>
          Colour is the value max<sub>a</sub> Q(s, a) of each cell and arrows are the greedy policy; cells without an
          arrow have not been updated yet. Each move pays −1, entering the goal pays +10 and ends the episode, and
          entering a trap (red cross) pays −20 and returns the agent to the start. With slip, a move goes sideways with
          that probability. Press play to watch the agent (black dot) learn; the orange line is the path of the last
          finished episode. The chart shows the return (or number of steps) of every episode. Drag along the chart, or
          use the episode slider, to travel back: the maze then shows the values, policy and path as they were after
          that episode. Play resumes from the latest episode. In value-iteration mode the model is known and each tick
          is one sweep of the Bellman optimality backup. Click a cell to add or remove a wall; changing any setting
          restarts learning.
        </>
      }
      controls={
        <>
          <ParamChoice label="maze" value={layout.name} onChange={chooseMaze} options={MAZE_OPTIONS} />
          <ParamChoice label="method" value={mode} onChange={setMode} options={MODES} />
          <ParamSlider
            label="speed"
            param={speed}
            format={(i) =>
              mode === 'planner' ? `${SWEEP_RATES[i]} sweeps/s` : `${formatNumber(STEP_RATES[i])} steps/s`
            }
          />
          <ParamSlider label="discount γ" param={gamma} />
          <ParamSlider label="slip probability" param={slip} />
          {learning && <ParamSlider label="step size α" param={alpha} />}
          {learning && <ParamSlider label="exploration ε" param={epsilon} />}
          {learning && <ParamSlider label="seed" param={seed} format={(v) => String(v)} />}
          {learning && <ParamSwitch label="decay ε as ε / (1 + e/10)" checked={decay} onChange={setDecay} />}
          <ParamSlider label={`${unit} shown`} param={travelParam} format={(v) => String(v)} withArrows />
          {learning && (
            <ParamChoice
              label="chart"
              value={metric}
              onChange={setMetric}
              options={[
                { value: 'return', label: 'return' },
                { value: 'steps', label: 'steps' },
              ]}
            />
          )}
          <div className="flex flex-wrap items-end gap-2">
            <ParamButton onClick={play} disabled={finished(run) && past === null}>
              {playing ? <Pause /> : <Play />} {playing ? 'Pause' : 'Play'}
            </ParamButton>
            <ParamButton onClick={stepOnce} disabled={finished(run)}>
              <StepForward /> Step
            </ParamButton>
            {learning && (
              <ParamButton onClick={finishEpisode} disabled={finished(run)}>
                <SkipForward /> Episode
              </ParamButton>
            )}
            <ParamButton onClick={reset}>
              <RotateCcw /> Reset
            </ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label={`${unit}s done`} value={`${done}${finished(run) ? ' (end)' : ''}`} />
          <Readout label="shown" value={past === null ? 'latest' : `${unit} ${past}`} />
          {learning && <Readout label="return" value={episode ? formatNumber(episode.ret) : '–'} />}
          {learning && <Readout label="steps" value={episode ? String(episode.steps) : '–'} />}
          {learning && (
            <Readout label="ε" value={episode ? formatNumber(episode.epsilon) : formatNumber(epsilon.value)} />
          )}
          {!learning && (
            <Readout label="largest change" value={residual === undefined ? '–' : formatNumber(residual)} />
          )}
          <Readout label="max Q(start, a)" value={formatNumber(maxQ(Q, maze.start))} />
          <Readout label="v*(start) by value iteration" value={formatNumber(vStar)} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Heatmap
          x={xs}
          y={ys}
          z={z}
          range={range}
          overlay={overlay}
          marker={marker}
          vectors={vectors}
          onCellClick={toggleWall}
          valueLabel="max Q"
          height={heatHeight}
          ariaLabel="Maze with the learned value of each cell, the greedy policy as arrows and the agent's path"
        />
        <XYChart
          series={series}
          xLabel={unit}
          yLabel={run.kind === 'plan' ? 'value of the start' : metric === 'return' ? 'return' : 'steps'}
          xRange={[0, Math.max(run.kind === 'plan' ? 10 : 20, done)]}
          yLog={learning && metric === 'steps'}
          handles={done > 0 ? [{ kind: 'x', at: shown, label: String(shown), onDrag: travelParam.set }] : undefined}
          height={240}
          ariaLabel={`Return of every ${unit}, with a handle to travel back in time`}
        />
      </div>
    </Interactive>
  )
}
