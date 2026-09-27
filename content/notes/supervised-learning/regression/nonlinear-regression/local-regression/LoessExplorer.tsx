import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { solve } from '../_shared/splines'

const N = 80
const GRID = linspace(0, 1, 101)
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
  const [span, setSpan] = useState(0.3)
  const [degree, setDegree] = useState<'0' | '1' | '2'>('1')
  const [outliers, setOutliers] = useState(true)
  const [robustOn, setRobustOn] = useState(false)
  const x0 = useParam(0.05, { min: 0, max: 1, step: 0.005 })
  const d = Number(degree)

  const data = useMemo(() => {
    const r = rng(12)
    const x = Array.from({ length: N }, () => r.uniform()).sort((a, b) => a - b)
    const y = x.map((xi, i) => truth(xi) + 0.3 * r.normal() + (outliers && OUTLIERS.includes(i) ? 3 : 0))
    return { x, y }
  }, [outliers])

  const fit = useMemo(() => {
    const { robust, edf } = loess(data.x, data.y, span, d, robustOn ? 2 : 0)
    const curve = GRID.map((g) => localFit(g, data.x, data.y, robust, span, d).value)
    return { robust, edf, curve }
  }, [data, span, d, robustOn])

  const local = localFit(x0.value, data.x, data.y, fit.robust, span, d)
  const lx = linspace(Math.max(0, x0.value - local.h), Math.min(1, x0.value + local.h), 31)
  const ly = lx.map((g) => local.coef.reduce((s, c, k) => s + c * (g - x0.value) ** k, 0))
  const inWindow: number[] = data.x.map((xi) => (Math.abs(xi - x0.value) < local.h ? 1 : 0))

  const series: XYSeries[] = [
    {
      name: 'data',
      type: 'scatter',
      x: data.x,
      y: data.y,
      group: inWindow,
      groupNames: ['outside the window', 'inside the window'],
    },
    { name: 'true curve', type: 'line', x: GRID, y: GRID.map(truth), muted: true, dashed: true },
    { name: 'LOESS', type: 'line', x: GRID, y: fit.curve, slot: 2 },
    { name: 'local fit at x₀', type: 'line', x: lx, y: ly, slot: 3 },
    { name: 'estimate at x₀', type: 'scatter', x: [x0.value], y: [local.value], emphasis: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: x0.value, label: 'x₀', onDrag: x0.set }]

  return (
    <Interactive
      title="Local regression"
      caption={
        <>
          Drag the vertical line to move the target point x₀. The points inside the window get tricube weights, and a
          weighted polynomial fitted to them gives the estimate at x₀ (the black dot). Sweeping x₀ traces the LOESS
          curve. Near the edges a local constant is pulled towards the interior; a local line is not. Three points are
          shifted up by 3; robustness iterations downweight them.
        </>
      }
      controls={
        <>
          <ParamSlider label="x₀" param={x0} />
          <ParamSlider label="span α" value={span} onChange={setSpan} min={0.05} max={1} step={0.05} />
          <ParamChoice label="local polynomial" value={degree} onChange={setDegree} options={DEGREES} />
          <ParamSwitch label="outliers" checked={outliers} onChange={setOutliers} />
          <ParamSwitch label="robustness iterations" checked={robustOn} onChange={setRobustOn} />
        </>
      }
      readout={
        <>
          <Readout label="points in window" value={String(inWindow.reduce((a, b) => a + b, 0))} />
          <Readout label="half-width h(x₀)" value={formatNumber(local.h)} />
          <Readout label="effective df" value={formatNumber(fit.edf)} />
        </>
      }
    >
      <XYChart series={series} xRange={[0, 1]} yRange={[-2, 4.5]} xLabel="x" yLabel="y" handles={handles} />
    </Interactive>
  )
}
