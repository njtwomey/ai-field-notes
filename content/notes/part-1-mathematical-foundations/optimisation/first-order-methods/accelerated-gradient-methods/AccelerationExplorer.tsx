import { useMemo, useState } from 'react'
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
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type Vec = [number, number]
type Variant = 'strong' | 'convex'

/** f(x, y) = ½(μx² + Ly²) with L = 1: the flat direction is x, the steep direction is y. */
const L = 1
const RANGE: Vec = [-1.2, 1.2]
const GRID = 81
/** Floor for f − f* on a log axis once a run has reached machine precision. */
const FLOOR = 1e-14
const TARGET = 1e-6

const METHODS = [
  { id: 'gd', label: 'gradient descent', slot: 1 },
  { id: 'hb', label: 'heavy ball', slot: 2 },
  { id: 'nag', label: 'Nesterov', slot: 3 },
] as const

const clamp = (v: number) => Math.round(Math.min(Math.max(v, RANGE[0]), RANGE[1]) * 100) / 100

function run(method: (typeof METHODS)[number]['id'], mu: number, start: Vec, steps: number, variant: Variant) {
  const f = ([x, y]: Vec) => 0.5 * (mu * x * x + L * y * y)
  const grad = ([x, y]: Vec): Vec => [mu * x, L * y]
  const rootK = Math.sqrt(L / mu)
  let x: Vec = [...start]
  let prev: Vec = [...start]
  let t = 1
  const path: Vec[] = [x]
  const values: number[] = [f(x)]
  for (let k = 0; k < steps; k++) {
    let next: Vec
    if (method === 'gd') {
      const g = grad(x)
      next = [x[0] - g[0] / L, x[1] - g[1] / L]
    } else if (method === 'hb') {
      // Polyak's tuned parameters for a quadratic with curvatures in [μ, L].
      const alpha = 4 / (Math.sqrt(L) + Math.sqrt(mu)) ** 2
      const beta = ((rootK - 1) / (rootK + 1)) ** 2
      const g = grad(x)
      next = [x[0] - alpha * g[0] + beta * (x[0] - prev[0]), x[1] - alpha * g[1] + beta * (x[1] - prev[1])]
    } else {
      let beta: number
      if (variant === 'strong') beta = (rootK - 1) / (rootK + 1)
      else if (k === 0) beta = 0
      else {
        // t_1 = 1 and β_k = (t_k − 1)/t_{k+1}, as in the O(1/k²) proof; the first two steps are plain gradient steps.
        const tNext = (1 + Math.sqrt(1 + 4 * t * t)) / 2
        beta = (t - 1) / tNext
        t = tNext
      }
      const y: Vec = [x[0] + beta * (x[0] - prev[0]), x[1] + beta * (x[1] - prev[1])]
      const g = grad(y)
      next = [y[0] - g[0] / L, y[1] - g[1] / L]
    }
    prev = x
    x = next
    path.push(x)
    values.push(f(x))
  }
  return { path, values }
}

export function AccelerationExplorer() {
  const state = useFigureState({
    logKappa: float(2, {
      min: 0.5,
      max: 3,
      step: 0.05,
      label: 'condition number κ = L/μ',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    steps: int(150, { min: 20, max: 500, step: 10, label: 'iterations' }),
    variant: choice<Variant>(
      [
        { value: 'strong', label: 'knows μ' },
        { value: 'convex', label: 'convex schedule' },
      ],
      'strong',
      { label: 'Nesterov momentum' },
    ),
  })
  const [start, setStart] = useState<Vec>([-1, 0.8])

  const kappa = 10 ** state.logKappa
  const mu = L / kappa

  const grid = useMemo(() => {
    const x = toFlat(linspace(RANGE[0], RANGE[1], GRID))
    const y = toFlat(linspace(RANGE[0], RANGE[1], GRID))
    const z = y.map((yv) => x.map((xv) => Math.log10(0.5 * (mu * xv * xv + L * yv * yv) + 1e-4)))
    return { x, y, z }
  }, [mu])

  const runs = useMemo(
    () => METHODS.map((m) => run(m.id, mu, start, state.steps, state.variant)),
    [mu, start, state.steps, state.variant],
  )

  const overlay = useMemo(
    (): SeriesSpec[] => [
      ...METHODS.map((m, i): SeriesSpec => ({
        name: m.label,
        type: 'line',
        x: runs[i].path.map((p) => p[0]),
        y: runs[i].path.map((p) => p[1]),
        slot: m.slot,
      })),
      { name: 'minimum', type: 'scatter', x: [0], y: [0], emphasis: true },
    ],
    [runs],
  )

  const curves = useMemo((): SeriesSpec[] => {
    const f0 = runs[0].values[0]
    const r2 = start[0] ** 2 + start[1] ** 2
    const ks = Array.from({ length: state.steps + 1 }, (_, k) => k)
    const floor = (v: number) => Math.max(v, FLOOR)
    const gdBound = ks.map((k) => floor(Math.min(f0 * (1 - mu / L) ** k, k > 0 ? (L * r2) / (2 * k) : f0)))
    const nagBound = ks.map((k) =>
      floor(
        state.variant === 'strong'
          ? (f0 + (mu / 2) * r2) * (1 - Math.sqrt(mu / L)) ** k
          : Math.min(f0, (2 * L * r2) / (k + 1) ** 2),
      ),
    )
    return [
      ...METHODS.map((m, i): SeriesSpec => ({
        name: m.label,
        type: 'line',
        x: ks,
        y: runs[i].values.map(floor),
        slot: m.slot,
      })),
      { name: 'bound: gradient descent', type: 'line', x: ks, y: gdBound, slot: 1, dashed: true },
      { name: 'bound: Nesterov', type: 'line', x: ks, y: nagBound, slot: 3, dashed: true },
    ]
  }, [runs, start, state.steps, mu, state.variant])

  const stepsTo = (values: number[]) => {
    const hit = values.findIndex((v) => v <= TARGET * values[0])
    return hit < 0 ? `> ${state.steps}` : String(hit)
  }

  const xAxis = useAxis({ label: 'x (flat, curvature μ)' })
  const yAxis = useAxis({ label: 'y (steep, curvature L)' })
  const xAxis2 = useAxis({ label: 'iteration k', hold: 'union' })
  const yAxis2 = useAxis({ label: 'f − f*', range: [FLOOR, undefined], hold: 'union', log: true })
  return (
    <Figure
      title="Acceleration on an ill-conditioned quadratic"
      state={state}
      caption="Left: log₁₀ f for f = ½(μx² + y²), with each method's path from the same start; drag the start point to move it. Right: f − f* per iteration on a log scale, with the guaranteed bounds dashed. Gradient descent uses step 1/L; heavy ball uses Polyak's tuned step and momentum; Nesterov uses step 1/L with momentum (√κ − 1)/(√κ + 1) when μ is known, or the (t_k − 1)/t_{k+1} schedule when it is not. Raise κ and compare the slopes."

      readouts={
        <>
          <Readout label="κ" value={formatNumber(kappa)} />
          <Readout label="√κ" value={formatNumber(Math.sqrt(kappa))} />
          {METHODS.map((m, i) => (
            <Readout key={m.id} label={`${m.label}: steps to 10⁻⁶ f₀`} value={stepsTo(runs[i].values)} />
          ))}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={360}>
          <Raster x={grid.x} y={grid.y} z={grid.z} valueLabel={'log₁₀ f'} />
          {seriesLayers(overlay, { live: true })}
          <Handle kind="point" at={start} label="start" onDrag={([x, y]) => setStart([clamp(x), clamp(y)])} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={360}>
          {seriesLayers(curves)}
        </Plot>
      </div>
    </Figure>
  )
}
