import { useEffect, useMemo, useRef } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { SolverDatasets } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { stream, uniform } from 'aifn/foundation/random'
import { sigmoid } from 'aifn/numerics/special'

type Vec = [number, number]
type Problem = { x1: number[]; x2: number[]; y: number[]; lambda: number }

/** Passes over the data. Each is one gradient step, one Newton step, or n / B SGD steps. */
const PASSES = 20
const BOX = 6
/** Floor for the log-scale gap once a solver has converged to machine precision. */
const FLOOR = 1e-12
const LIMIT = 1e3

const softplus = (z: number) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))

function loss({ x1, x2, y, lambda }: Problem, [a, b]: Vec): number {
  let total = 0
  for (let i = 0; i < y.length; i++) {
    const z = a * x1[i] + b * x2[i]
    total += softplus(z) - y[i] * z
  }
  return total / y.length + 0.5 * lambda * (a * a + b * b)
}

/** Gradient over the rows in `rows` (all rows when omitted): Xᵀ(p − y)/|rows| + λw. */
function gradient({ x1, x2, y, lambda }: Problem, [a, b]: Vec, rows?: number[]): Vec {
  const idx = rows ?? y.map((_, i) => i)
  let g1 = 0
  let g2 = 0
  for (const i of idx) {
    const r = sigmoid(a * x1[i] + b * x2[i]) - y[i]
    g1 += r * x1[i]
    g2 += r * x2[i]
  }
  return [g1 / idx.length + lambda * a, g2 / idx.length + lambda * b]
}

/** Hessian XᵀSX/n + λI as its three distinct entries [h11, h12, h22]. */
function hessian({ x1, x2, y, lambda }: Problem, [a, b]: Vec): [number, number, number] {
  let h11 = 0
  let h12 = 0
  let h22 = 0
  for (let i = 0; i < y.length; i++) {
    const p = sigmoid(a * x1[i] + b * x2[i])
    const s = p * (1 - p)
    h11 += s * x1[i] * x1[i]
    h12 += s * x1[i] * x2[i]
    h22 += s * x2[i] * x2[i]
  }
  const n = y.length
  return [h11 / n + lambda, h12 / n, h22 / n + lambda]
}

const finite = ([a, b]: Vec) => Number.isFinite(a) && Number.isFinite(b) && Math.hypot(a, b) < LIMIT

function gradientDescent(problem: Problem, start: Vec, eta: number): Vec[] {
  const path: Vec[] = [start]
  for (let k = 0; k < PASSES; k++) {
    const w = path[k]
    const g = gradient(problem, w)
    const next: Vec = [w[0] - eta * g[0], w[1] - eta * g[1]]
    if (!finite(next)) break
    path.push(next)
  }
  return path
}

/** Newton steps −H⁻¹g; with `damped`, halve each step until the loss decreases enough (Armijo). */
function newton(problem: Problem, start: Vec, damped: boolean, steps = PASSES): Vec[] {
  const path: Vec[] = [start]
  for (let k = 0; k < steps; k++) {
    const w = path[k]
    const [g1, g2] = gradient(problem, w)
    const [h11, h12, h22] = hessian(problem, w)
    const det = h11 * h22 - h12 * h12
    if (!(det > 0)) break
    const d: Vec = [-(h22 * g1 - h12 * g2) / det, -(h11 * g2 - h12 * g1) / det]
    let t = 1
    if (damped) {
      const f0 = loss(problem, w)
      const slope = g1 * d[0] + g2 * d[1]
      while (loss(problem, [w[0] + t * d[0], w[1] + t * d[1]]) > f0 + 1e-4 * t * slope && t > 1e-10) t /= 2
    }
    const next: Vec = [w[0] + t * d[0], w[1] + t * d[1]]
    if (!finite(next)) break
    path.push(next)
  }
  return path
}

/** Mini-batch SGD. Returns every step and the index in `steps` where each pass ends. */
function sgd(problem: Problem, start: Vec, eta: number, batch: number, decay: boolean, seed: number) {
  const n = problem.y.length
  const random = stream(seed)
  const order = problem.y.map((_, i) => i)
  const steps: Vec[] = [start]
  const ends = [0]
  let w = start
  for (let pass = 0; pass < PASSES; pass++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(uniform(random) * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    const rate = decay ? eta / (1 + pass) : eta
    for (let s = 0; s < n; s += batch) {
      const g = gradient(problem, w, order.slice(s, s + batch))
      const next: Vec = [w[0] - rate * g[0], w[1] - rate * g[1]]
      if (!finite(next)) return { steps, ends }
      w = next
      steps.push(w)
    }
    ends.push(steps.length - 1)
  }
  return { steps, ends }
}

/** The path up to its first point outside the plotted box, so a diverging run does not stretch the heatmap. */
function clip(path: Vec[]): { x: number[]; y: number[] } {
  const inside: Vec[] = []
  for (const p of path) {
    if (Math.abs(p[0]) > BOX || Math.abs(p[1]) > BOX) break
    inside.push(p)
  }
  return { x: inside.map((p) => p[0]), y: inside.map((p) => p[1]) }
}

/** Palette slots by solver. Slots 0 and 1 are the classes in the data panel, and slot 0 is the heatmap's own hue. */
const SLOTS = { gd: 2, newton: 3, sgd: 4 } as const
/** Independent SGD runs from the same start, differing only in the order the examples are visited. */
const SGD_RUNS = 10
const LOSS_RANGE: [number, number] = [0, 3]
const PASS_RANGE: [number, number] = [0, PASSES]
const GAP_RANGE: [number, number] = [FLOOR, 10]
/** Starting weights per dataset, chosen away from the minimum so the paths differ. */
const DATASETS = ['overlapping', 'correlated', 'noisy labels', 'separable'] as const
type DatasetName = (typeof DATASETS)[number]
const STARTS: Record<DatasetName, Vec> = {
  overlapping: [-3, 1],
  correlated: [-4, -2],
  'noisy labels': [-3, 1],
  separable: [-3, 1],
}

/** The decision boundary w·x = 0 (bias fixed at 0) across a square of half-width r, or nothing when w = 0. */
function boundary([a, b]: Vec, r: number): { x: number[]; y: number[] } {
  const norm = Math.hypot(a, b)
  if (norm < 1e-9) return { x: [], y: [] }
  const [u, v] = [(-b / norm) * 2 * r, (a / norm) * 2 * r]
  return { x: [-u, u], y: [-v, v] }
}

/**
 * Gradient descent, Newton's method and 10 runs of mini-batch SGD on four datasets with two features and the bias
 * fixed at 0. The data and the unpenalised loss surfaces come from python/mlc/figures/logistic_regression.py; the
 * solvers run here. With two parameters and 200 examples, a full recompute per control change is cheap.
 */
export function SolverRace() {
  const { data, error } = useFigure<SolverDatasets>('logistic-regression/solver-datasets')
  const state = useFigureState({
    dataset: choice(DATASETS, 'overlapping', { label: 'dataset' }),
    w1: float(STARTS.overlapping[0], { min: -BOX, max: BOX, step: 0.1, label: 'start w₁' }),
    w2: float(STARTS.overlapping[1], { min: -BOX, max: BOX, step: 0.1, label: 'start w₂' }),
    pass: slider(0, PASSES, PASSES, { step: 1, label: 'pass', format: (v) => v.toFixed(0) }),
    eta: float(1, { gt: 0, le: 3, scale: 'log10', suggestions: [0.1, 0.3, 1, 3], label: 'step size η (GD, SGD)' }),
    lambda: float(0, { min: 0, max: 0.5, step: 0.01, label: 'L2 penalty λ' }),
    batch: int(10, { ge: 1, le: 200, suggestions: [1, 10, 50], label: 'SGD batch B' }),
    decay: setting(true, 'SGD step decays as η/(1 + pass)'),
    damped: setting(true, 'Newton with backtracking'),
  })

  const name = state.dataset
  // A new dataset starts every solver from that dataset's start point (not on first render: the URL may set one).
  const shown = useRef(name)
  useEffect(() => {
    if (shown.current === name) return
    shown.current = name
    const [a, b] = STARTS[name]
    state.set('w1', a)
    state.set('w2', b)
  }, [name, state])

  const dataset = data?.datasets.find((d) => d.name === name)
  const problem = useMemo((): Problem | undefined => {
    if (!dataset) return undefined
    return { x1: dataset.data.x, x2: dataset.data.y, y: dataset.data.group ?? [], lambda: state.lambda }
  }, [dataset, state.lambda])
  // Without a penalty, separable data has no minimum: the loss falls towards 0 as ‖w‖ grows.
  const hasMinimum = !!dataset && (!dataset.separable || state.lambda > 0)

  const start: Vec = [state.w1, state.w2]
  const runs = useMemo(() => {
    if (!problem) return undefined
    const s: Vec = [state.w1, state.w2]
    const optimum = hasMinimum ? newton(problem, [0, 0], true, 50).at(-1)! : undefined
    const best = optimum ? loss(problem, optimum) : 0
    const gd = gradientDescent(problem, s, state.eta)
    const nt = newton(problem, s, state.damped)
    const sg = Array.from({ length: SGD_RUNS }, (_, r) =>
      sgd(problem, s, state.eta, state.batch, state.decay, 7 + 101 * r),
    )
    // L = λmax(XᵀX)/(4n) + λ bounds the curvature everywhere, since p(1 − p) ≤ 1/4.
    const n = problem.y.length
    const [a11, a12, a22] = problem.y.reduce(
      ([p, q, r], _, i) => [p + problem.x1[i] ** 2, q + problem.x1[i] * problem.x2[i], r + problem.x2[i] ** 2],
      [0, 0, 0],
    )
    const lmax = (a11 + a22) / 2 + Math.hypot((a11 - a22) / 2, a12)
    const smoothness = lmax / (4 * n) + problem.lambda
    let condition: number | undefined
    if (optimum) {
      const [h11, h12, h22] = hessian(problem, optimum)
      const mid = (h11 + h22) / 2
      const rad = Math.hypot((h11 - h22) / 2, h12)
      condition = (mid + rad) / (mid - rad)
    }
    return { optimum, best, gd, nt, sg, smoothness, condition }
  }, [problem, hasMinimum, state.w1, state.w2, state.eta, state.batch, state.decay, state.damped])

  const surface = useMemo(() => {
    if (!dataset) return undefined
    const { x, y, z } = dataset.surface
    return z.map((row, i) => row.map((v, j) => v + 0.5 * state.lambda * (x[j] ** 2 + y[i] ** 2)))
  }, [dataset, state.lambda])

  // Half-width of the data panel: the largest coordinate, rounded up.
  const reach = useMemo(
    () => (dataset ? Math.ceil(Math.max(...dataset.data.x.map(Math.abs), ...dataset.data.y.map(Math.abs))) : 4),
    [dataset],
  )
  const dataRange = useMemo((): [number, number] => [-reach, reach], [reach])

  const k = state.pass
  const at = (path: Vec[], i: number) => path[Math.min(i, path.length - 1)]
  const sgdAt = (run: { steps: Vec[]; ends: number[] }, i: number) =>
    run.steps[run.ends[Math.min(i, run.ends.length - 1)]]

  const overlay = useMemo((): SeriesSpec[] => {
    if (!runs) return []
    const upto = <T,>(xs: T[], n: number) => xs.slice(0, n + 1)
    return [
      { name: 'gradient descent', type: 'line', ...clip(upto(runs.gd, k)), slot: SLOTS.gd, showPoints: true },
      { name: 'Newton', type: 'line', ...clip(upto(runs.nt, k)), slot: SLOTS.newton, showPoints: true },
      ...runs.sg.map((run): SeriesSpec => ({
        name: 'SGD',
        type: 'line',
        ...clip(run.steps.slice(0, (run.ends[k] ?? run.steps.length) + 1)),
        slot: SLOTS.sgd,
      })),
      ...(runs.optimum
        ? [{ name: 'minimum', type: 'scatter' as const, x: [runs.optimum[0]], y: [runs.optimum[1]], emphasis: true }]
        : []),
    ]
  }, [runs, k])

  const boundaries = useMemo((): SeriesSpec[] => {
    if (!runs || !dataset) return []
    return [
      {
        name: 'data',
        type: 'scatter',
        x: dataset.data.x,
        y: dataset.data.y,
        group: dataset.data.group,
        groupNames: dataset.data.group_names ?? undefined,
      },
      { name: 'gradient descent', type: 'line', ...boundary(at(runs.gd, k), reach), slot: SLOTS.gd },
      { name: 'Newton', type: 'line', ...boundary(at(runs.nt, k), reach), slot: SLOTS.newton },
      ...runs.sg.map((run): SeriesSpec => ({
        name: 'SGD',
        type: 'line',
        ...boundary(sgdAt(run, k), reach),
        slot: SLOTS.sgd,
      })),
      ...(runs.optimum
        ? [{ name: 'minimum', type: 'line' as const, ...boundary(runs.optimum, reach), emphasis: true, dashed: true }]
        : []),
    ]
  }, [runs, dataset, k, reach])

  const convergence = useMemo((): SeriesSpec[] => {
    if (!runs || !problem) return []
    const gap = (path: Vec[]) => path.map((w) => Math.max(loss(problem, w) - runs.best, FLOOR))
    const passes = (m: number) => Array.from({ length: m }, (_, i) => i)
    return [
      { name: 'gradient descent', type: 'line', x: passes(runs.gd.length), y: gap(runs.gd), slot: SLOTS.gd },
      { name: 'Newton', type: 'line', x: passes(runs.nt.length), y: gap(runs.nt), slot: SLOTS.newton },
      ...runs.sg.map((run): SeriesSpec => {
        const atPass = run.ends.map((e) => run.steps[e])
        return { name: 'SGD', type: 'line', x: passes(atPass.length), y: gap(atPass), slot: SLOTS.sgd }
      }),
    ]
  }, [runs, problem])

  const xAxis = useAxis({ label: 'x₁', range: dataRange })
  const yAxis = useAxis({ label: 'x₂', range: dataRange, equal: xAxis })
  const xAxis2 = useAxis({ label: dataset?.surface.x_label })
  const yAxis2 = useAxis({ label: dataset?.surface.y_label, equal: xAxis2 })
  const passAxis = useAxis({ label: 'pass over the data', range: PASS_RANGE, integer: true })
  const gapAxis = useAxis({ label: hasMinimum ? 'loss − minimum' : 'loss (no minimum)', range: GAP_RANGE, log: true })

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !dataset || !surface || !runs || !problem) return null

  const gapAt = (path: Vec[], i: number) =>
    i < path.length ? formatNumber(loss(problem, path[i]) - runs.best) : 'diverged'
  const sgdGaps = runs.sg.flatMap((run) =>
    run.ends[k] !== undefined ? [loss(problem, run.steps[run.ends[k]]) - runs.best] : [],
  )

  return (
    <Figure
      title="Three solvers on four datasets"
      state={state}
      caption="Top left: the data and the decision boundary of each solver after the chosen pass; the dashed ink line is the boundary at the minimum. Top right: the mean cross-entropy plus (λ/2)‖w‖² over (w₁, w₂), bias fixed at 0, with each solver's path from the same start; the diamond is the minimum. Drag the start dot anywhere on the surface. Bottom: each solver's loss above the minimum after every pass over the data, on a log scale. One pass is one gradient step, one Newton step, or n/B SGD steps on mini-batches of B examples. SGD runs 10 times from the same start, each visiting the examples in a different random order; the spread of the paths and boundaries is its stochasticity, and a smaller batch, a larger step or no decay widens it. Newton's gap falls faster with each step; gradient descent's falls by a steady factor, and on the correlated data it stalls after a few passes: the valley is narrow, the Hessian's condition number is about 40, and a step size safe across the valley is tiny along it. On the separable data there is no minimum without a penalty: every solver keeps growing ‖w‖, and the bottom panel shows the loss itself. Switch backtracking off to see pure Newton overshoot. Drag the vertical line at the bottom, or step the pass slider, to walk through the passes."
      readouts={
        <>
          <Readout label="GD gap" value={gapAt(runs.gd, k)} />
          <Readout label="Newton gap" value={gapAt(runs.nt, k)} />
          <Readout
            label={`SGD gap, ${SGD_RUNS} runs`}
            value={
              sgdGaps.length === SGD_RUNS
                ? `${formatNumber(Math.min(...sgdGaps))} to ${formatNumber(Math.max(...sgdGaps))}`
                : `${SGD_RUNS - sgdGaps.length} diverged`
            }
          />
          <Readout label="1/L, safe GD step" value={formatNumber(1 / runs.smoothness)} />
          <Readout
            label="condition number of H at minimum"
            value={runs.condition === undefined ? 'no minimum' : formatNumber(runs.condition)}
          />
        </>
      }
    >
      <p className="text-xs text-muted-foreground">{dataset.description}</p>
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(boundaries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Raster x={dataset.surface.x} y={dataset.surface.y} z={surface} range={LOSS_RANGE} valueLabel={'loss'} />
          {seriesLayers(overlay, { live: true })}
          <Handle
            kind="point"
            at={start}
            label="start"
            onDrag={([a, b]) => {
              state.set('w1', a)
              state.set('w2', b)
            }}
          />
        </Plot>
        <div className="md:col-span-2">
          <Plot x={passAxis} y={gapAxis} height={260}>
            {seriesLayers(convergence)}
            <Handle kind="x" at={k} label="pass" onDrag={(v) => state.set('pass', Math.round(v))} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
