import { useMemo, useState } from 'react'
import { Interactive, Readout, XYChart, formatNumber, type Handle, type Vec2, type XYSeries } from '@/components/viz'

const BASE_X = [1, 2, 3, 4, 5, 6, 7, 8]
const BASE_Y = [3.1, 3.9, 6.8, 6.2, 9.5, 10.1, 13.4, 12.9]
const X_RANGE: [number, number] = [0, 20]
const Y_RANGE: [number, number] = [-5, 35]
const P = 2

function fit(x: number[], y: number[]) {
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  const sxx = x.reduce((a, xi) => a + (xi - mx) ** 2, 0)
  const slope = x.reduce((a, xi, i) => a + (xi - mx) * (y[i] - my), 0) / sxx
  return { slope, intercept: my - slope * mx, mx, sxx }
}

const BASE_FIT = fit(BASE_X, BASE_Y)

/**
 * Eight fixed points and one movable point. Leverage h depends only on the point's x; Cook's distance combines h with
 * the point's residual and equals the scaled shift in all fitted values when the point is deleted.
 */
export function InfluenceDrag() {
  const [pt, setPt] = useState<Vec2>([16, 14])

  const r = useMemo(() => {
    const x = [...BASE_X, pt[0]]
    const y = [...BASE_Y, pt[1]]
    const n = x.length
    const f = fit(x, y)
    const resid = x.map((xi, i) => y[i] - f.intercept - f.slope * xi)
    const s2 = resid.reduce((a, e) => a + e * e, 0) / (n - P)
    const h = 1 / n + (pt[0] - f.mx) ** 2 / f.sxx
    const e = resid[n - 1]
    const rStd = e / Math.sqrt(s2 * (1 - h))
    const s2i = ((n - P) * s2 - (e * e) / (1 - h)) / (n - P - 1)
    const tExt = e / Math.sqrt(s2i * (1 - h))
    const cook = (rStd * rStd * h) / (P * (1 - h))
    const dfbetas = (f.slope - BASE_FIT.slope) / Math.sqrt(s2i / f.sxx)
    const line = (a: number, b: number) => X_RANGE.map((xv) => a + b * xv)
    const series: XYSeries[] = [
      { name: 'fixed points', type: 'scatter', x: BASE_X, y: BASE_Y, slot: 0 },
      { name: 'fit with the movable point', type: 'line', x: X_RANGE, y: line(f.intercept, f.slope), slot: 1 },
      {
        name: 'fit without it',
        type: 'line',
        x: X_RANGE,
        y: line(BASE_FIT.intercept, BASE_FIT.slope),
        muted: true,
        dashed: true,
      },
      { name: 'movable point', type: 'scatter', x: [pt[0]], y: [pt[1]], emphasis: true },
    ]
    return { h, tExt, cook, dfbetas, slope: f.slope, series }
  }, [pt])

  const handles: Handle[] = [
    {
      kind: 'point',
      at: pt,
      label: 'movable point',
      onDrag: ([x, y]) =>
        setPt([Math.min(Math.max(x, X_RANGE[0]), X_RANGE[1]), Math.min(Math.max(y, Y_RANGE[0]), Y_RANGE[1])]),
    },
  ]

  return (
    <Interactive
      title="Leverage, outliers and influence"
      caption="Drag the dark point anywhere on the plot. Its leverage h depends only on how far its x lies from the mean of x. A high-leverage point on the line of the other points changes nothing (Cook's distance near 0). The same point moved off the line drags the fit toward it (Cook's distance far above 1). A point near the centre with a large residual is an outlier but moves the slope little."
      readout={
        <>
          <Readout label="leverage h" value={formatNumber(r.h)} />
          <Readout label="studentised residual" value={formatNumber(r.tExt)} />
          <Readout label="Cook's distance" value={formatNumber(r.cook)} />
          <Readout label="DFBETAS (slope)" value={formatNumber(r.dfbetas)} />
          <Readout label="slope with / without" value={`${formatNumber(r.slope)} / ${formatNumber(BASE_FIT.slope)}`} />
        </>
      }
    >
      <XYChart
        height={320}
        series={r.series}
        xRange={X_RANGE}
        yRange={Y_RANGE}
        xLabel="x"
        yLabel="y"
        handles={handles}
      />
    </Interactive>
  )
}
