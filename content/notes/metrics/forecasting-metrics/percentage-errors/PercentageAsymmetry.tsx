import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

const ACTUAL = 100
const FORECASTS = linspace(0, 300, 301)
const Y = linspace(0.01, 4, 400)

/** Absolute and symmetric percentage errors against the forecast, and where MAPE puts the best forecast. */
export function PercentageAsymmetry() {
  const forecast = useParam(150, { min: 0, max: 300, step: 1 })
  const sigma = useParam(0.5, { min: 0.1, max: 1, step: 0.05 })
  const f = forecast.value
  const ape = (100 * Math.abs(ACTUAL - f)) / ACTUAL
  const sape = (200 * Math.abs(ACTUAL - f)) / (ACTUAL + f)

  const errors = useMemo(
    (): XYSeries[] => [
      {
        name: 'absolute percentage error',
        type: 'line',
        x: FORECASTS,
        y: FORECASTS.map((v) => (100 * Math.abs(ACTUAL - v)) / ACTUAL),
        slot: 0,
      },
      {
        name: 'symmetric percentage error',
        type: 'line',
        x: FORECASTS,
        y: FORECASTS.map((v) => (200 * Math.abs(ACTUAL - v)) / (ACTUAL + v)),
        slot: 1,
      },
    ],
    [],
  )

  // Lognormal target with μ = 0: mean e^{σ²/2}, median 1, MAPE-optimal forecast e^{−σ²}.
  const s = sigma.value
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
  const distribution: XYSeries[] = [
    { name: 'density of the target', type: 'line', x: Y, y: density, muted: true, area: true },
    ...marks.map((m): XYSeries => ({
      name: m.name,
      type: 'line',
      x: [m.at, m.at],
      y: [0, peak],
      slot: m.slot,
      dashed: true,
    })),
  ]

  const handles: Handle[] = [
    { kind: 'x', at: f, label: 'forecast', onDrag: (x) => forecast.set(Math.round(Math.min(300, Math.max(0, x)))) },
  ]

  return (
    <Interactive
      title="Percentage errors are asymmetric"
      caption="Left: the actual value is 100. Drag the forecast. Absolute percentage error can reach at most 100% by forecasting too low, but grows without bound by forecasting too high. Symmetric percentage error is bounded at 200% either way, but for the same absolute miss it penalises a low forecast more than a high one. Right: for a right-skewed target, the forecast that minimises expected absolute percentage error lies below the median, which lies below the mean."
      controls={
        <>
          <ParamSlider label="forecast" param={forecast} format={(v) => String(v)} />
          <ParamSlider label="skew of the target, σ of log" param={sigma} />
        </>
      }
      readout={
        <>
          <Readout label="absolute percentage error" value={`${formatNumber(ape)}%`} />
          <Readout label="symmetric percentage error" value={`${formatNumber(sape)}%`} />
          <Readout label="MAPE-optimal / median" value={formatNumber(Math.exp(-s * s))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={errors}
          handles={handles}
          xLabel="forecast (actual = 100)"
          yLabel="error (%)"
          xRange={[0, 300]}
          yRange={[0, 210]}
          height={300}
        />
        <XYChart
          series={distribution}
          xLabel="target value"
          yLabel="density"
          xRange={[0, 4]}
          yRange={[0, undefined]}
          height={300}
        />
      </div>
    </Interactive>
  )
}
