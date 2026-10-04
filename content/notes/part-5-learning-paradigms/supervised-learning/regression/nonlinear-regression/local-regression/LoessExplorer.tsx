import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { solve } from '../_shared/splines'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 80
const GRID = toFlat(linspace(0, 1, 101))
const truth = (x: number) => Math.sin(2 * Math.PI * x)
const OUTLIERS = [9, 30, 55]
const tricube = (u: number) => (u < 1 ? (1 - u ** 3) ** 3 : 0)

type Local = { value: number; coef: number[]; h: number; selfWeight: number }

/** Weighted polynomial fit of the given degree in (x − x₀), with tricube weights over the nearest span·n points. */
function localFit(x0: number, x: number[], y: number[], robust: number[], span: number, degree: number): Local {
  const dist = x.map((xi) => Math.abs(xi - x0))
  const q = Math.max(degree + 1, Math.ceil(span * x.length))
  const h = [...dist].sort((a, b) => a - b)[q - 1] * 1.0001 || 1e-9
  const w = dist.map((d, i) => tricube(d / h) * robust[i])
  const p = degree + 1
  const XtWX = Array.from({ length: p }, () => Array(p).fill(0))
  const XtWy = Array(p).fill(0)
  x.forEach((xi, i) => {
    if (w[i] === 0) return
    const row = Array.from({ length: p }, (_, k) => (xi - x0) ** k)
    for (let j = 0; j < p; j++) {
      XtWy[j] += w[i] * row[j] * y[i]
      for (let k = 0; k < p; k++) XtWX[j][k] += w[i] * row[j] * row[k]
    }
  })
  const coef = solve(XtWX, XtWy)
  // Weight of y_i in its own fitted value: l_i(x_i) = w_i [(XᵀWX)⁻¹]₀₀ when x₀ = x_i.
  const e0 = solve(
    XtWX,
    Array.from({ length: p }, (_, k) => (k === 0 ? 1 : 0)),
  )[0]
  return { value: coef[0], coef, h, selfWeight: e0 }
}

/** LOESS at the data points and on a grid, with optional bisquare robustness iterations (Cleveland 1979). */
function loess(x: number[], y: number[], span: number, degree: number, iterations: number) {
  let robust = Array(x.length).fill(1)
  for (let it = 0; it <= iterations; it++) {
    const fitted = x.map((xi) => localFit(xi, x, y, robust, span, degree).value)
    if (it === iterations) {
      const edf = x.reduce((s, xi, i) => s + robust[i] * localFit(xi, x, y, robust, span, degree).selfWeight, 0)
      return { robust, edf }
    }
    const res = y.map((yi, i) => Math.abs(yi - fitted[i]))
    const s = [...res].sort((a, b) => a - b)[Math.floor(res.length / 2)]
    robust = res.map((r) => {
      const u = r / (6 * s)
      return u < 1 ? (1 - u * u) ** 2 : 0
    })
  }
  return { robust, edf: NaN }
}

const DEGREES = [
  { value: '0', label: 'constant' },
  { value: '1', label: 'linear' },
  { value: '2', label: 'quadratic' },
] as const

export function LoessExplorer() {
  const state = useFigureState({
    x0: float(0.05, { min: 0, max: 1, step: 0.005, label: 'x₀' }),
    span: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'span α' }),
    degree: choice<'0' | '1' | '2'>(DEGREES, '1', { label: 'local polynomial' }),
    outliers: setting(true, 'outliers'),
    robustOn: setting(false, 'robustness iterations'),
  })
  const d = Number(state.degree)

  const data = useMemo(() => {
    const r = stream(12)
    const x = Array.from({ length: N }, () => uniform(r)).sort((a, b) => a - b)
    const y = x.map((xi, i) => truth(xi) + 0.3 * normal(r) + (state.outliers && OUTLIERS.includes(i) ? 3 : 0))
    return { x, y }
  }, [state.outliers])

  const fit = useMemo(() => {
    const { robust, edf } = loess(data.x, data.y, state.span, d, state.robustOn ? 2 : 0)
    const curve = GRID.map((g) => localFit(g, data.x, data.y, robust, state.span, d).value)
    return { robust, edf, curve }
  }, [data, state.span, d, state.robustOn])

  const local = localFit(state.x0, data.x, data.y, fit.robust, state.span, d)
  const lx = toFlat(linspace(Math.max(0, state.x0 - local.h), Math.min(1, state.x0 + local.h), 31))
  const ly = lx.map((g) => local.coef.reduce((s, c, k) => s + c * (g - state.x0) ** k, 0))
  const inWindow: number[] = data.x.map((xi) => (Math.abs(xi - state.x0) < local.h ? 1 : 0))

  const series = [
    {
      name: 'data',
      x: data.x,
      y: data.y,
      group: inWindow,
      groupNames: ['outside the window', 'inside the window'],
    },
    { name: 'true curve', x: GRID, y: GRID.map(truth), muted: true, dashed: true },
    { name: 'LOESS', x: GRID, y: fit.curve, slot: 2 },
    { name: 'local fit at x₀', x: lx, y: ly, slot: 3 },
    { name: 'estimate at x₀', x: [state.x0], y: [local.value], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2, 4.5] })
  return (
    <Figure
      title="Local regression"
      state={state}
      caption={
        <>
          Drag the vertical line to move the target point x₀. The points inside the window get tricube weights, and a
          weighted polynomial fitted to them gives the estimate at x₀ (the black dot). Sweeping x₀ traces the LOESS
          curve. Near the edges a local constant is pulled towards the interior; a local line is not. Three points are
          shifted up by 3; robustness iterations downweight them.
        </>
      }

      readouts={
        <>
          <Readout label="points in window" value={String(inWindow.reduce((a, b) => a + b, 0))} />
          <Readout label="half-width h(x₀)" value={formatNumber(local.h)} />
          <Readout label="effective df" value={formatNumber(fit.edf)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Points {...series[4]} />
        <Handle {...state.handle('x0', { label: 'x₀' })} />
      </Plot>
    </Figure>
  )
}
