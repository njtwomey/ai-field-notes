import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type FunctionId = 'logcosh' | 'well' | 'cycle'

type Problem = {
  label: string
  f: (x: number) => number
  d1: (x: number) => number
  d2: (x: number) => number
  range: [number, number]
  start: number
  /** The minimiser that a successful run reaches. */
  minimiser: number
}

const PROBLEMS: Record<FunctionId, Problem> = {
  logcosh: {
    label: 'log cosh x',
    f: (x) => Math.log(Math.cosh(x)),
    d1: (x) => Math.tanh(x),
    d2: (x) => 1 / Math.cosh(x) ** 2,
    range: [-3, 3],
    start: 1,
    minimiser: 0,
  },
  well: {
    label: 'x⁴/4 − x²/2 + x/10',
    f: (x) => x ** 4 / 4 - x ** 2 / 2 + x / 10,
    d1: (x) => x ** 3 - x + 0.1,
    d2: (x) => 3 * x ** 2 - 1,
    range: [-1.8, 1.8],
    start: 0.3,
    minimiser: -1.0466805318046022,
  },
  cycle: {
    label: 'x⁴/4 − x² + 2x',
    f: (x) => x ** 4 / 4 - x ** 2 + 2 * x,
    d1: (x) => x ** 3 - 2 * x + 2,
    d2: (x) => 3 * x ** 2 - 2,
    range: [-2.5, 2],
    start: 0,
    minimiser: -1.7692923542386314,
  },
}

const ITERATIONS = 12
const LIMIT = 1e3
/** Floor for |f′| on a log axis once a run has converged to machine precision. */
const FLOOR = 1e-16

/** Pure Newton, or damped Newton: backtracking on the Newton direction, or on −f′ where f″ ≤ 0. */
function newton(p: Problem, x0: number, damped: boolean) {
  const xs = [x0]
  let x = x0
  for (let k = 0; k < ITERATIONS; k++) {
    const g = p.d1(x)
    const h = p.d2(x)
    let d = damped && h <= 0 ? -g : -g / h
    if (!Number.isFinite(d)) break
    let t = 1
    if (damped) {
      while (p.f(x + t * d) > p.f(x) + 0.25 * t * g * d && t > 1e-10) t /= 2
      if (t <= 1e-10) d = 0
    }
    x = x + t * d
    xs.push(x)
    if (!Number.isFinite(x) || Math.abs(x) > LIMIT) break
  }
  return xs
}

export function NewtonExplorer() {
  const state = useFigureState({
    fn: variants(
      Object.fromEntries(
        (Object.keys(PROBLEMS) as FunctionId[]).map((k) => {
          const q = PROBLEMS[k]
          const start = slider(q.range[0], q.range[1], q.start, { step: 0.01, label: 'start x₀' })
          return [k, { label: q.label, params: { start } }]
        }),
      ) as Record<FunctionId, { label: string; params: { start: ReturnType<typeof slider> } }>,
      { label: 'function' },
    ),
    damped: setting(false, 'damped (backtracking)'),
  })
  const p = PROBLEMS[state.fn.key]
  const start = state.fn.values.start
  const damped = state.damped

  const xs = useMemo(() => newton(p, start, damped), [p, start, damped])
  const [lo, hi] = p.range

  const series = useMemo(() => {
    const grid = toFlat(linspace(lo, hi, 241))
    const x0 = start
    const f0 = p.f(x0)
    const g0 = p.d1(x0)
    const h0 = p.d2(x0)
    const inRange = xs.filter((x) => x >= lo && x <= hi)
    return [
      { name: 'f', x: grid, y: grid.map(p.f), slot: 0 },
      {
        name: 'quadratic model at start',
        x: grid,
        y: grid.map((x) => f0 + g0 * (x - x0) + 0.5 * h0 * (x - x0) ** 2),
        slot: 2,
        dashed: true,
      },
      { name: 'iterates', x: inRange, y: inRange.map(p.f), slot: 1 },
    ] as const
  }, [p, lo, hi, xs, start])

  const segments = useMemo((): Segment[] => {
    const out: Segment[] = []
    for (let k = 0; k + 1 < xs.length; k++) {
      const a = xs[k]
      const b = xs[k + 1]
      if (a < lo || a > hi || b < lo || b > hi) break
      out.push({ from: [a, p.f(a)], to: [b, p.f(b)] })
    }
    return out
  }, [xs, p, lo, hi])

  const [yMin, yMax] = useMemo(() => {
    const ys = toFlat(linspace(lo, hi, 241)).map(p.f)
    const min = Math.min(...ys)
    const max = Math.max(...ys)
    const pad = 0.1 * (max - min)
    return [min - pad, max + pad]
  }, [p, lo, hi])

  const gradients = useMemo(
    () =>
      [
        {
          name: '|f′(x_k)|',
          x: xs.map((_, k) => k),
          y: xs.map((x) => (Number.isFinite(x) ? Math.max(Math.abs(p.d1(x)), FLOOR) : FLOOR)),
          slot: 1,
        },
      ] as const,
    [xs, p],
  )

  const last = xs[xs.length - 1]
  const diverged = !Number.isFinite(last) || Math.abs(last) > LIMIT
  const converged = !diverged && Math.abs(p.d1(last)) < 1e-10
  const status = diverged
    ? 'diverged'
    : converged
      ? p.d2(last) > 0
        ? Math.abs(last - p.minimiser) < 1e-6
          ? 'converged to the minimum'
          : 'converged to a local minimum'
        : 'converged to a maximum'
      : 'not converged'

  const xAxis = useAxis({ label: 'x', range: p.range })
  const yAxis = useAxis({ label: 'f(x)', range: [yMin, yMax] })
  const xAxis2 = useAxis({ label: 'iteration k', hold: 'union' })
  const yAxis2 = useAxis({ label: '|f′(x_k)|', hold: 'union', log: true })
  return (
    <Figure
      title="Newton's method in one dimension"
      state={state}
      caption="Left: the function, the quadratic model at the start (dashed) and the Newton iterates joined in order. Drag the vertical line, or use the slider, to move the start. Each pure Newton step jumps to the stationary point of the current quadratic model. Right: |f′| per iteration on a log scale; quadratic convergence shows as a curve that bends sharply downward. Damped Newton halves the step until the function decreases enough, and steps along −f′ where the curvature is negative."
      readouts={
        <>
          <Readout label="f″(x₀)" value={formatNumber(p.d2(start))} />
          <Readout label="last iterate" value={diverged ? '∞' : formatNumber(last)} />
          <Readout label="outcome" value={status} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Points {...series[2]} />
          <Segments segments={segments} />
          <Handle {...state.handle('fn.start', { label: 'start' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve {...gradients[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
