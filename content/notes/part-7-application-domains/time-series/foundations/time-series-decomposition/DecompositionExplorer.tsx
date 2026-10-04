import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

const M = 12
const YEARS = 8
const T = M * YEARS
const TIMES = Array.from({ length: T }, (_, t) => t + 1)
/** Times at which the centred moving average exists: half a period is lost at each end. */
const INNER = TIMES.slice(M / 2, T - M / 2)
const inner = (xs: number[]) => xs.slice(M / 2, T - M / 2)
/** A fixed seasonal shape with mean 0 and peak near 1: a main annual cycle plus a smaller half-year harmonic. */
const SHAPE = Array.from(
  { length: M },
  (_, j) => Math.sin((2 * Math.PI * j) / M) + 0.4 * Math.cos((4 * Math.PI * j) / M),
)

type Form = 'additive' | 'multiplicative'

/** Classical decomposition: a centred 2×m moving average for the trend, averaged detrended values for the season. */
function decompose(y: number[], form: Form) {
  const trend = y.map((_, t) => {
    if (t < M / 2 || t >= T - M / 2) return NaN
    let s = 0.5 * y[t - M / 2] + 0.5 * y[t + M / 2]
    for (let k = -M / 2 + 1; k < M / 2; k++) s += y[t + k]
    return s / M
  })
  const detrended = y.map((v, t) => (form === 'additive' ? v - trend[t] : v / trend[t]))
  const raw = Array.from({ length: M }, (_, j) => {
    const vals = detrended.filter((v, t) => t % M === j && Number.isFinite(v))
    return vals.reduce((a, b) => a + b, 0) / vals.length
  })
  // Normalise so the seasonal indices sum to 0 (additive) or average 1 (multiplicative) over one period.
  const avg = raw.reduce((a, b) => a + b, 0) / M
  const index = raw.map((v) => (form === 'additive' ? v - avg : v / avg))
  const seasonal = y.map((_, t) => index[t % M])
  // The remainder in the data's own units, so the two forms can be compared on one axis.
  const remainder = y.map((v, t) => (form === 'additive' ? v - trend[t] - seasonal[t] : v - trend[t] * seasonal[t]))
  return { trend, seasonal, remainder }
}

const spread = (xs: number[]) => {
  const v = xs.filter(Number.isFinite)
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}

/** Sample autocorrelation at one lag; near 0 when the remainder holds no seasonal pattern. */
const lagCorrelation = (xs: number[], h: number) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  let num = 0
  let den = 0
  xs.forEach((v, t) => {
    den += (v - m) ** 2
    if (t + h < xs.length) num += (v - m) * (xs[t + h] - m)
  })
  return den > 0 ? num / den : 0
}

/** Generate additive or multiplicative seasonal data and decompose it either way. */
export function DecompositionExplorer() {
  const state = useFigureState({
    truth: choice<Form>(
      [
        { value: 'additive', label: 'additive' },
        { value: 'multiplicative', label: 'multiplicative' },
      ],
      'multiplicative',
      { label: 'data generated as' },
    ),
    form: choice<Form>(
      [
        { value: 'additive', label: 'additive' },
        { value: 'multiplicative', label: 'multiplicative' },
      ],
      'additive',
      { label: 'decomposition' },
    ),
    noise: float(1, { min: 0, max: 5, step: 0.1, label: 'noise sd' }),
    seed: int(2, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    const level = TIMES.map((t) => 50 + 1.2 * t + 8 * Math.sin((2 * Math.PI * t) / 60))
    const y = level.map((l, t) => {
      const s = SHAPE[t % M]
      const e = state.noise * normal(g)
      return state.truth === 'additive' ? l + 15 * s + e : l * (1 + 0.2 * s) + e
    })
    const d = decompose(y, state.form)
    const rem = inner(d.remainder)
    const data = [
      { name: 'data y_t', x: TIMES, y, slot: 0 },
      { name: 'trend (2×12 moving average)', x: INNER, y: inner(d.trend), emphasis: true },
    ] as const
    const seasonal = [{ name: 'seasonal', x: TIMES, y: d.seasonal, slot: 1 }] as const
    const remainder = [{ name: 'remainder', x: INNER, y: inner(d.remainder), slot: 2 }] as const
    return { data, seasonal, remainder, lag12: lagCorrelation(rem, M), sd: spread(rem) }
  }, [state.truth, state.form, state.noise, state.seed])

  const xAxis = useAxis({ label: 'month t', hold: 'union' })
  const yAxis = useAxis({ label: 'y_t', hold: 'union' })
  const xAxis2 = useAxis({ label: 'month t', hold: 'union' })
  const yAxis2 = useAxis({ label: state.form === 'additive' ? 'S_t' : 'S_t (ratio)', hold: 'union' })
  const xAxis3 = useAxis({ label: 'month t', hold: 'union' })
  const yAxis3 = useAxis({ label: 'remainder', hold: 'union' })
  return (
    <Figure
      title="Classical decomposition"
      state={state}
      caption="Monthly data over eight years: a rising trend with a slow wave, an annual seasonal pattern, and Gaussian noise. The data are generated with additive seasonality (a fixed swing) or multiplicative seasonality (a swing proportional to the level). The decomposition estimates the trend with a centred 2×12 moving average, then averages the detrended values month by month. Fitting an additive decomposition to multiplicative data forces one average swing on every year. The remainder then holds a seasonal pattern of its own, large at both ends of the series where the true swing is furthest from the average, and its lag-12 autocorrelation is high. The matching form leaves only noise."

      readouts={
        <>
          <Readout label="remainder sd" value={formatNumber(r.sd)} />
          <Readout label="remainder autocorrelation at lag 12" value={formatNumber(r.lag12)} />
        </>
      }
    >
      <div className="space-y-3">
        <Plot x={xAxis} y={yAxis} height={220}>
          <Curve {...r.data[0]} />
          <Curve {...r.data[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={130}>
          <Curve {...r.seasonal[0]} />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={130}>
          <Bars {...r.remainder[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
