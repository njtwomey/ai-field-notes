import { useMemo, useState } from 'react'
import { Button, Curve, Figure, formatNumber, Handle, Plot, Points, Readout, useAxis } from 'aifn-render'
import { concordance, mae, mse, pearson, r2, spearman } from './scores'
import { normal, stream } from 'aifn/foundation/random'

const RANGE: [number, number] = [0, 12]
// Fourteen well-predicted points: predictions scatter around the truth with standard deviation 0.5.
const BASE = (() => {
  const g = stream(11)
  return Array.from({ length: 14 }, (_, i) => {
    const truth = 1 + (8 * i) / 13
    return { truth, pred: truth + 0.5 * normal(g) }
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

  const series = [
    { name: 'perfect prediction', x: RANGE, y: RANGE, dashed: true, muted: true },
    { name: 'predictions', x: BASE.map((b) => b.truth), y: BASE.map((b) => b.pred), slot: 0 },
  ] as const
  const pair = (key: keyof typeof r.with) => `${formatNumber(r.with[key])} (${formatNumber(r.without[key])})`

  const xAxis = useAxis({ label: 'true value y', range: RANGE })
  const yAxis = useAxis({ label: 'prediction ŷ', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="One prediction, six scores"
      caption="Each point is a prediction plotted against the true value; the dashed line is perfect prediction. Drag the white point. Mean squared error, R², Pearson's r and the concordance coefficient all react sharply when it leaves the line, because they square the miss. Mean absolute error grows only in proportion. Spearman's ρ changes only when the point's rank among the predictions changes. Values in brackets are the scores without the movable point."
      controls={
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <Button variant="outline" size="sm" key={p.label} onClick={() => setPoint(p.at)}>
              {p.label}
            </Button>
          ))}
        </div>
      }
      readouts={
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
        <Plot x={xAxis} y={yAxis}>
          <Curve {...series[0]} />
          <Points {...series[1]} />
          <Handle
            kind="point"
            at={point}
            label="movable prediction"
            onDrag={([x, y]) => setPoint([Math.min(12, Math.max(0, x)), Math.min(12, Math.max(0, y))])}
          />
        </Plot>
      </div>
    </Figure>
  )
}
