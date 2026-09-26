import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, ParamSwitch, Readout, XYChart, formatNumber } from '@/components/viz'
import { linspace, mean, olsFit, rng } from '@/lib/math'

const TRUE_SLOPE = 1.5
const TRUE_INTERCEPT = -0.5

export function FitExplorer() {
  const [n, setN] = useState(40)
  const [noise, setNoise] = useState(0.8)
  const [slope, setSlope] = useState(0.5)
  const [intercept, setIntercept] = useState(0.5)
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

  return (
    <Interactive
      title="Fit a line by hand"
      caption="Move the slope and intercept. The grey segments are residuals. Least squares minimises the mean of their squared lengths."
      controls={
        <>
          <ParamSlider label="slope" value={slope} onChange={setSlope} min={-2} max={3} step={0.05} />
          <ParamSlider label="intercept" value={intercept} onChange={setIntercept} min={-3} max={3} step={0.05} />
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
      />
    </Interactive>
  )
}
