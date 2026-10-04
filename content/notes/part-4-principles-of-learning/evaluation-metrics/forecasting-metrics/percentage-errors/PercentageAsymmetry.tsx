import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const ACTUAL = 100
const FORECASTS = toFlat(linspace(0, 300, 301))
const Y = toFlat(linspace(0.01, 4, 400))

/** Absolute and symmetric percentage errors against the forecast, and where MAPE puts the best forecast. */
export function PercentageAsymmetry() {
  const state = useFigureState({
    forecast: int(150, { min: 0, max: 300, step: 1, label: 'forecast', format: (v) => String(v) }),
    sigma: float(0.5, { min: 0.1, max: 1, step: 0.05, label: 'skew of the target, σ of log' }),
  })
  const f = state.forecast
  const ape = (100 * Math.abs(ACTUAL - f)) / ACTUAL
  const sape = (200 * Math.abs(ACTUAL - f)) / (ACTUAL + f)

  const errors = useMemo(
    () =>
      [
        {
          name: 'absolute percentage error',
          x: FORECASTS,
          y: FORECASTS.map((v) => (100 * Math.abs(ACTUAL - v)) / ACTUAL),
          slot: 0,
        },
        {
          name: 'symmetric percentage error',
          x: FORECASTS,
          y: FORECASTS.map((v) => (200 * Math.abs(ACTUAL - v)) / (ACTUAL + v)),
          slot: 1,
        },
      ] as const,
    [],
  )

  // Lognormal target with μ = 0: mean e^{σ²/2}, median 1, MAPE-optimal forecast e^{−σ²}.
  const s = state.sigma
  const density = useMemo(
    () => Y.map((y) => Math.exp(-(Math.log(y) ** 2) / (2 * s * s)) / (y * s * Math.sqrt(2 * Math.PI))),
    [s],
  )
  const peak = Math.max(...density)
  const marks = [
    { name: 'mean (best for MSE)', at: Math.exp((s * s) / 2), slot: 2 },
    { name: 'median (best for MAE)', at: 1, slot: 3 },
    { name: 'best for MAPE', at: Math.exp(-s * s), slot: 4 },
  ]
  const distribution: SeriesSpec[] = [
    { name: 'density of the target', type: 'line', x: Y, y: density, muted: true, area: true },
    ...marks.map((m): SeriesSpec => ({
      name: m.name,
      type: 'line',
      x: [m.at, m.at],
      y: [0, peak],
      slot: m.slot,
      dashed: true,
    })),
  ]

  const xAxis = useAxis({ label: 'forecast (actual = 100)', range: [0, 300] })
  const yAxis = useAxis({ label: 'error (%)', range: [0, 210] })
  const xAxis2 = useAxis({ label: 'target value', range: [0, 4] })
  const yAxis2 = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Percentage errors are asymmetric"
      state={state}
      caption="Left: the actual value is 100. Drag the forecast. Absolute percentage error can reach at most 100% by forecasting too low, but grows without bound by forecasting too high. Symmetric percentage error is bounded at 200% either way, but for the same absolute miss it penalises a low forecast more than a high one. Right: for a right-skewed target, the forecast that minimises expected absolute percentage error lies below the median, which lies below the mean."

      readouts={
        <>
          <Readout label="absolute percentage error" value={`${formatNumber(ape)}%`} />
          <Readout label="symmetric percentage error" value={`${formatNumber(sape)}%`} />
          <Readout label="MAPE-optimal / median" value={formatNumber(Math.exp(-s * s))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...errors[0]} />
          <Curve {...errors[1]} />
          <Handle
            kind="x"
            at={f}
            label="forecast"
            onDrag={(x) => state.set('forecast', Math.round(Math.min(300, Math.max(0, x))))}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers(distribution)}
        </Plot>
      </div>
    </Figure>
  )
}
