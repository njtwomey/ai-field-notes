import { useMemo, useState } from 'react'
import { Interactive, ParamButton, Readout, XYChart, formatNumber, type Handle, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'
import { concordance, mae, mse, pearson, r2, spearman } from './scores'

const RANGE: [number, number] = [0, 12]
// Fourteen well-predicted points: predictions scatter around the truth with standard deviation 0.5.
const BASE = (() => {
  const g = rng(11)
  return Array.from({ length: 14 }, (_, i) => {
    const truth = 1 + (8 * i) / 13
    return { truth, pred: truth + 0.5 * g.normal() }
  })
})()

const PRESETS: { label: string; at: [number, number] }[] = [
  { label: 'On the line', at: [5, 5] },
  { label: 'Far off the line', at: [3, 10] },
  { label: 'Far along the line', at: [11.5, 11.5] },
  { label: 'Large miss, same rank', at: [9.2, 12] },
]

/**
 * One draggable prediction among fourteen good ones. Squared-error scores, R² and Pearson's r react strongly when it
 * leaves the line; MAE grows only linearly; Spearman's ρ changes only when the point's rank order changes.
 */
export function OutlierScores() {
  const [point, setPoint] = useState<[number, number]>([3, 10])

  const r = useMemo(() => {
    const score = (y: number[], p: number[]) => ({
      mse: mse(y, p),
      mae: mae(y, p),
      r2: r2(y, p),
      pearson: pearson(y, p),
      spearman: spearman(y, p),
      ccc: concordance(y, p),
    })
    const y0 = BASE.map((b) => b.truth)
    const p0 = BASE.map((b) => b.pred)
    return { without: score(y0, p0), with: score([...y0, point[0]], [...p0, point[1]]) }
  }, [point])

  const series: XYSeries[] = [
    { name: 'perfect prediction', type: 'line', x: RANGE, y: RANGE, dashed: true, muted: true },
    { name: 'predictions', type: 'scatter', x: BASE.map((b) => b.truth), y: BASE.map((b) => b.pred), slot: 0 },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: point,
      label: 'movable prediction',
      onDrag: ([x, y]) => setPoint([Math.min(12, Math.max(0, x)), Math.min(12, Math.max(0, y))]),
    },
  ]
  const pair = (key: keyof typeof r.with) => `${formatNumber(r.with[key])} (${formatNumber(r.without[key])})`

  return (
    <Interactive
      title="One prediction, six scores"
      caption="Each point is a prediction plotted against the true value; the dashed line is perfect prediction. Drag the white point. Mean squared error, R², Pearson's r and the concordance coefficient all react sharply when it leaves the line, because they square the miss. Mean absolute error grows only in proportion. Spearman's ρ changes only when the point's rank among the predictions changes. Values in brackets are the scores without the movable point."
      controls={
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <ParamButton key={p.label} onClick={() => setPoint(p.at)}>
              {p.label}
            </ParamButton>
          ))}
        </div>
      }
      readout={
        <>
          <Readout label="MSE" value={pair('mse')} />
          <Readout
            label="RMSE"
            value={`${formatNumber(Math.sqrt(r.with.mse))} (${formatNumber(Math.sqrt(r.without.mse))})`}
          />
          <Readout label="MAE" value={pair('mae')} />
          <Readout label="R²" value={pair('r2')} />
          <Readout label="Pearson r" value={pair('pearson')} />
          <Readout label="Spearman ρ" value={pair('spearman')} />
          <Readout label="concordance" value={pair('ccc')} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          series={series}
          handles={handles}
          xLabel="true value y"
          yLabel="prediction ŷ"
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
        />
      </div>
    </Interactive>
  )
}
