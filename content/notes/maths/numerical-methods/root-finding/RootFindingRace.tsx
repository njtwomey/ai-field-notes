import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

type FnId = 'sqrt2' | 'cosx' | 'double'

type Problem = {
  label: string
  f: (x: number) => number
  df: (x: number) => number
  root: number
  /** Sign-changing bracket for bisection, or null when the root has no sign change. */
  bracket: [number, number] | null
  view: [number, number]
  x0: number
}

const PROBLEMS: Record<FnId, Problem> = {
  sqrt2: {
    label: 'x² − 2',
    f: (x) => x * x - 2,
    df: (x) => 2 * x,
    root: Math.SQRT2,
    bracket: [0, 2],
    view: [0.2, 3],
    x0: 2.5,
  },
  cosx: {
    label: 'cos x − x',
    f: (x) => Math.cos(x) - x,
    df: (x) => -Math.sin(x) - 1,
    root: 0.7390851332151607,
    bracket: [0, 1],
    view: [-1, 3],
    x0: 2.5,
  },
  double: {
    label: '(x − 1)²(x + 2)',
    f: (x) => (x - 1) ** 2 * (x + 2),
    df: (x) => 2 * (x - 1) * (x + 2) + (x - 1) ** 2,
    root: 1,
    bracket: null,
    view: [0.2, 3],
    x0: 2.5,
  },
}

const ITERS = 14
const FLOOR = 1e-17
const STEPS = Array.from({ length: ITERS + 1 }, (_, k) => k)

function newton(p: Problem, x0: number): number[] {
  const xs = [x0]
  for (let k = 0; k < ITERS; k++) {
    const x = xs[k]
    const d = p.df(x)
    if (d === 0 || !Number.isFinite(x)) break
    xs.push(x - p.f(x) / d)
  }
  return xs
}

function secant(p: Problem, x0: number): number[] {
  const xs = [x0, x0 + 0.1]
  for (let k = 1; k < ITERS; k++) {
    const [a, b] = [xs[k - 1], xs[k]]
    const denom = p.f(b) - p.f(a)
    if (denom === 0) break
    xs.push(b - (p.f(b) * (b - a)) / denom)
  }
  return xs
}

function bisection(p: Problem): number[] {
  if (!p.bracket) return []
  let [a, b] = p.bracket
  const xs: number[] = []
  for (let k = 0; k <= ITERS; k++) {
    const m = (a + b) / 2
    xs.push(m)
    if (Math.sign(p.f(m)) === Math.sign(p.f(a))) a = m
    else b = m
  }
  return xs
}

const errors = (xs: number[], root: number) => xs.map((x) => Math.max(FLOOR, Math.abs(x - root)))

/** Newton, secant and bisection on the same equation: the Newton path on f, and the error of each method per step. */
export function RootFindingRace() {
  const [id, setId] = useState<FnId>('sqrt2')
  const p = PROBLEMS[id]
  const x0 = useParam(p.x0, { min: p.view[0], max: p.view[1], step: 0.01 })

  const runs = useMemo(
    () => ({ newton: newton(p, x0.value), secant: secant(p, x0.value), bisection: bisection(p) }),
    [p, x0.value],
  )

  const errorSeries = useMemo((): XYSeries[] => {
    const s: XYSeries[] = [
      {
        name: 'Newton',
        type: 'line',
        x: STEPS.slice(0, runs.newton.length),
        y: errors(runs.newton, p.root),
        slot: 0,
      },
      {
        name: 'secant',
        type: 'line',
        x: STEPS.slice(0, runs.secant.length),
        y: errors(runs.secant, p.root),
        slot: 1,
      },
    ]
    if (runs.bisection.length) {
      s.push({ name: 'bisection', type: 'line', x: STEPS, y: errors(runs.bisection, p.root), slot: 2 })
    }
    return s
  }, [runs, p])

  const grid = useMemo(() => linspace(p.view[0], p.view[1], 200), [p])
  const fSeries = useMemo(
    (): XYSeries[] => [
      { name: 'f(x)', type: 'line', x: grid, y: grid.map(p.f), emphasis: true },
      {
        name: 'Newton iterates',
        type: 'scatter',
        x: runs.newton.slice(0, 5),
        y: runs.newton.slice(0, 5).map(() => 0),
        slot: 0,
      },
    ],
    [grid, p, runs.newton],
  )
  // Tangent steps: from (x_k, f(x_k)) down the tangent to (x_{k+1}, 0), for the first few iterates.
  const tangents: Segment[] = runs.newton.slice(0, 4).flatMap((x, k): Segment[] =>
    k + 1 < runs.newton.length
      ? [
          { from: [x, 0], to: [x, p.f(x)] },
          { from: [x, p.f(x)], to: [runs.newton[k + 1], 0] },
        ]
      : [],
  )
  const handles: Handle[] = [{ kind: 'x', at: x0.value, label: 'x₀', onDrag: x0.set }]

  const last = runs.newton.at(-1) ?? NaN
  return (
    <Interactive
      title="Newton, secant and bisection"
      caption="Left: f with Newton's tangent steps from the start x₀; drag x₀ along the axis. Right: the error |xₖ − x*| after each step, on a log scale. Bisection halves its error each step. Newton's error is squared near a simple root, doubling the correct digits; the secant method is in between. At the double root of (x − 1)²(x + 2) Newton only halves the error, and bisection cannot start because f does not change sign."
      controls={
        <>
          <ParamChoice
            label="equation f(x) = 0"
            value={id}
            onChange={(v: FnId) => {
              setId(v)
              x0.set(PROBLEMS[v].x0)
            }}
            options={(Object.keys(PROBLEMS) as FnId[]).map((k) => ({ value: k, label: PROBLEMS[k].label }))}
          />
          <ParamSlider label="start x₀" param={x0} />
        </>
      }
      readout={
        <>
          <Readout label="root x*" value={p.root.toPrecision(16)} />
          <Readout label={`Newton after ${runs.newton.length - 1} steps`} value={last.toPrecision(16)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          xLabel="x"
          yLabel="f(x)"
          series={fSeries}
          segments={tangents}
          xRange={p.view}
          handles={handles}
        />
        <XYChart
          height={300}
          xLabel="step k"
          yLabel="|xₖ − x*|"
          series={errorSeries}
          xRange={[0, ITERS]}
          yLog
          yRange={[FLOOR, 10]}
        />
      </div>
    </Interactive>
  )
}
