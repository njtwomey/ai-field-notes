import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type V2 = [number, number]
type M2 = [V2, V2]
/** A linear test problem ẋ = A x + b(t) in two dimensions; one-dimensional problems leave the second state at 0. */
type Problem = {
  label: string
  A: M2
  b: (t: number) => V2
  x0: V2
  T: number
  exact: (t: number) => number
  y: [number, number]
}

const PROBLEMS = {
  decay: {
    label: 'decay ẋ = −x',
    A: [
      [-1, 0],
      [0, 0],
    ],
    b: () => [0, 0],
    x0: [1, 0],
    T: 5,
    exact: (t: number) => Math.exp(-t),
    y: [-0.5, 1.2],
  },
  oscillator: {
    label: 'oscillator ẍ = −x',
    A: [
      [0, 1],
      [-1, 0],
    ],
    b: () => [0, 0],
    x0: [1, 0],
    T: 10,
    exact: (t: number) => Math.cos(t),
    y: [-2, 2],
  },
  stiff: {
    label: 'stiff ẋ = −50(x − cos t)',
    A: [
      [-50, 0],
      [0, 0],
    ],
    b: (t: number) => [50 * Math.cos(t), 0],
    x0: [0, 0],
    T: 2,
    exact: (t: number) => (2500 * Math.cos(t) + 50 * Math.sin(t)) / 2501 - (2500 / 2501) * Math.exp(-50 * t),
    y: [-0.5, 1.5],
  },
} satisfies Record<string, Problem>
type ProblemKey = keyof typeof PROBLEMS

const METHODS = ['Euler', 'midpoint', 'RK4', 'implicit Euler'] as const
type Method = (typeof METHODS)[number]
const BOUND = 1e6

const mv = (A: M2, x: V2): V2 => [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]

function step(p: Problem, method: Method, t: number, x: V2, h: number): V2 {
  const f = (s: number, y: V2): V2 => {
    const ax = mv(p.A, y)
    const b = p.b(s)
    return [ax[0] + b[0], ax[1] + b[1]]
  }
  const add = (y: V2, k: V2, c: number): V2 => [y[0] + c * k[0], y[1] + c * k[1]]
  if (method === 'Euler') return add(x, f(t, x), h)
  if (method === 'midpoint') return add(x, f(t + h / 2, add(x, f(t, x), h / 2)), h)
  if (method === 'RK4') {
    const k1 = f(t, x)
    const k2 = f(t + h / 2, add(x, k1, h / 2))
    const k3 = f(t + h / 2, add(x, k2, h / 2))
    const k4 = f(t + h, add(x, k3, h))
    return [
      x[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
      x[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    ]
  }
  // Implicit Euler: (I − hA) x_{n+1} = x_n + h b(t_{n+1}), a 2×2 solve.
  const b = p.b(t + h)
  const r: V2 = [x[0] + h * b[0], x[1] + h * b[1]]
  const m00 = 1 - h * p.A[0][0]
  const m01 = -h * p.A[0][1]
  const m10 = -h * p.A[1][0]
  const m11 = 1 - h * p.A[1][1]
  const det = m00 * m11 - m01 * m10
  return [(m11 * r[0] - m01 * r[1]) / det, (-m10 * r[0] + m00 * r[1]) / det]
}

/** Solve to time T with n = T/h steps; the path stops if the state exceeds BOUND. */
function solve(p: Problem, method: Method, h: number) {
  const n = Math.round(p.T / h)
  const ts = [0]
  const xs = [p.x0[0]]
  let x = p.x0
  for (let k = 0; k < n; k++) {
    x = step(p, method, k * h, x, h)
    if (!(Math.abs(x[0]) < BOUND && Math.abs(x[1]) < BOUND)) return { ts, xs, error: Infinity }
    ts.push((k + 1) * h)
    xs.push(x[0])
  }
  return { ts, xs, error: Math.abs(x[0] - p.exact(n * h)) }
}

const H_GRID = toFlat(linspace(Math.log10(0.005), Math.log10(0.5), 40))

export function SolverComparison() {
  const state = useFigureState({
    problem: choice<ProblemKey>(
      (Object.keys(PROBLEMS) as ProblemKey[]).map((k) => ({ value: k, label: PROBLEMS[k].label })),
      'decay',
      { label: 'problem' },
    ),
    logH: float(-1, {
      min: Math.log10(0.005),
      max: Math.log10(0.5),
      step: 0.01,
      label: 'step size h',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
  })
  const { logH, problem, set } = state
  const h = 10 ** logH
  const p: Problem = PROBLEMS[problem]

  const runs = useMemo(() => METHODS.map((m) => solve(p, m, h)), [p, h])
  const exactT = useMemo(() => toFlat(linspace(0, p.T, 400)), [p])
  const pathSeries = useMemo<SeriesSpec[]>(
    () => [
      { name: 'exact', type: 'line', x: exactT, y: exactT.map(p.exact), emphasis: true, dashed: true },
      ...runs.map((r, i) => ({ name: METHODS[i], type: 'line' as const, x: r.ts, y: r.xs, slot: i })),
    ],
    [runs, exactT, p],
  )

  const errorSeries = useMemo<SeriesSpec[]>(
    () =>
      METHODS.map((m, i) => ({
        name: m,
        type: 'line' as const,
        x: H_GRID,
        // Clamp so blow-ups and round-off floors stay on a readable log axis.
        y: H_GRID.map((lh) => Math.min(1e3, Math.max(1e-14, solve(p, m, 10 ** lh).error))),
        slot: i,
      })),
    [p],
  )
  const handles = useMemo<Handle[]>(
    () => [{ kind: 'x', at: logH, label: 'h', onDrag: (v: number) => set('logH', v) }],
    [logH, set],
  )

  const xAxis = useAxis({ label: 't', range: [0, p.T] })
  const yAxis = useAxis({ label: 'x', range: p.y })
  const xAxis2 = useAxis({ label: 'log₁₀ h', hold: 'union' })
  const yAxis2 = useAxis({ label: 'error at T', range: [1e-14, 1e3], log: true })
  return (
    <Figure
      title="Four solvers on three test problems"
      state={state}
      caption="Left: each method's solution with step size h against the exact solution (dashed). Right: the error at the final time for every step size, on log–log axes; the slopes are the orders 1, 2, 4 and 1. Drag the vertical line or use the slider to change h. On the stiff problem, the explicit methods blow up once h exceeds their stability limit, while implicit Euler stays stable for every h."

      readouts={
        <>
          <Readout label="steps" value={Math.round(p.T / h)} />
          {runs.map((r, i) => (
            <Readout
              key={METHODS[i]}
              label={`${METHODS[i]} error`}
              value={Number.isFinite(r.error) ? r.error.toExponential(1) : 'blew up'}
            />
          ))}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(pathSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers(errorSeries)}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
