import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
} from '@/components/viz'
import { linspace, mean, olsFit, rng } from '@/lib/math'

const TRUE_SLOPE = 1.5
const TRUE_INTERCEPT = -0.5
/** x position of the slope handle. The intercept handle sits at x = 0. */
const SLOPE_AT = 2

export function FitExplorer() {
  const [n, setN] = useState(40)
  const [noise, setNoise] = useState(0.8)
  const slopeParam = useParam(0.5, { min: -2, max: 3, step: 0.05 })
  const interceptParam = useParam(0.5, { min: -3, max: 3, step: 0.05 })
  const slope = slopeParam.value
  const intercept = interceptParam.value
  const [showOls, setShowOls] = useState(true)

  const data = useMemo(() => {
    const r = rng(7)
    const x = Array.from({ length: n }, () => -3 + 6 * r.uniform())
    const eps = Array.from({ length: n }, () => r.normal())
    return { x, y: x.map((xi, i) => TRUE_SLOPE * xi + TRUE_INTERCEPT + noise * eps[i]) }
  }, [n, noise])

  const ols = useMemo(() => olsFit(data.x, data.y), [data])
  const mse = (m: number, c: number) => mean(data.x.map((xi, i) => (data.y[i] - (m * xi + c)) ** 2))
  const grid = linspace(-3, 3, 2)
  // Two points on the line: (0, b) moves the intercept, (2, 2m + b) turns the line about the intercept.
  const handles: Handle[] = [
    { kind: 'point', at: [0, intercept], label: 'intercept', onDrag: ([, y]) => interceptParam.set(y) },
    {
      kind: 'point',
      at: [SLOPE_AT, SLOPE_AT * slope + intercept],
      label: 'slope',
      onDrag: ([, y]) => slopeParam.set((y - intercept) / SLOPE_AT),
    },
  ]

  return (
    <Interactive
      title="Fit a line by hand"
      caption="Move the slope and intercept with the sliders, or drag the two dots on the line: the dot at x = 0 sets the intercept, the other turns the line. The grey segments are residuals. Least squares minimises the mean of their squared lengths."
      controls={
        <>
          <ParamSlider label="slope" param={slopeParam} />
          <ParamSlider label="intercept" param={interceptParam} />
          <ParamSlider label="noise σ" value={noise} onChange={setNoise} min={0} max={3} step={0.1} />
          <ParamSlider label="samples" value={n} onChange={setN} min={5} max={200} step={1} />
          <ParamSwitch label="show least-squares fit" checked={showOls} onChange={setShowOls} />
        </>
      }
      readout={
        <>
          <Readout label="your MSE" value={formatNumber(mse(slope, intercept))} />
          <Readout label="least-squares MSE" value={formatNumber(mse(ols.slope, ols.intercept))} />
          <Readout
            label="least-squares fit"
            value={`y = ${formatNumber(ols.slope)}x ${ols.intercept < 0 ? '−' : '+'} ${formatNumber(Math.abs(ols.intercept))}`}
          />
        </>
      }
    >
      <XYChart
        xRange={[-3, 3]}
        yRange={[-9, 6]}
        series={[
          { name: 'data', type: 'scatter', x: data.x, y: data.y, slot: 0 },
          { name: 'your line', type: 'line', x: grid, y: grid.map((g) => slope * g + intercept), slot: 1 },
          ...(showOls
            ? [
                {
                  name: 'least squares',
                  type: 'line' as const,
                  x: grid,
                  y: grid.map((g) => ols.slope * g + ols.intercept),
                  slot: 2,
                  dashed: true,
                },
              ]
            : []),
        ]}
        segments={data.x.map((xi, i) => ({ from: [xi, data.y[i]], to: [xi, slope * xi + intercept] }))}
        xLabel="x"
        yLabel="y"
        handles={handles}
      />
    </Interactive>
  )
}
