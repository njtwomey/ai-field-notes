import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { averagePrecision, binormalAuc, binormalEer, curves, nearestIndex, rates } from './binormal'

/**
 * ROC and precision-recall curves of the binormal model side by side. Prevalence changes the PR curve and leaves the
 * ROC curve alone; the operating point is dragged along the ROC curve and shown on both.
 */
export function RocPrCurves({
  title = 'ROC and precision-recall curves',
  caption,
  prevalence = 0.1,
  separation = 1.5,
  showEer = false,
}: {
  title?: string
  caption?: string
  prevalence?: number
  separation?: number
  showEer?: boolean
}) {
  const state = useFigureState({
    pi: slider(0.01, 0.5, prevalence, { step: 0.01, label: 'prevalence π' }),
    d: float(separation, { min: 0, max: 4, step: 0.1, label: 'separation d' }),
    t: slider(-4, 8, separation / 2, { step: 0.01, label: 'threshold t' }),
  })

  const c = useMemo(() => curves(state.d, state.pi), [state.d, state.pi])
  const ap = useMemo(() => averagePrecision(state.d, state.pi), [state.d, state.pi])
  const at = rates(state.t, state.d)
  const precisionAt = (state.pi * at.tpr) / (state.pi * at.tpr + (1 - state.pi) * at.fpr || 1)
  const eer = binormalEer(state.d)

  const roc: SeriesSpec[] = [
    { name: 'chance', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'ROC curve', type: 'line', x: c.fpr, y: c.tpr, slot: 0 },
    ...(showEer
      ? [
          { name: 'FPR = FNR', type: 'line' as const, x: [0, 1], y: [1, 0], dashed: true, slot: 2 },
          { name: 'equal error rate', type: 'scatter' as const, x: [eer], y: [1 - eer], slot: 2 },
        ]
      : []),
  ]
  const pr = [
    { name: 'chance (precision = π)', x: [0, 1], y: [state.pi, state.pi], dashed: true, muted: true },
    { name: 'PR curve', x: c.tpr, y: c.precision, slot: 1 },
    { name: 'operating point', x: [at.tpr], y: [precisionAt], emphasis: true },
  ] as const
  // The operating point is a location on the ROC curve: dragging moves it along the curve, which sets the threshold.

  const xAxis = useAxis({ label: 'false-positive rate', range: [0, 1] })
  const yAxis = useAxis({ label: 'true-positive rate', range: [0, 1], equal: xAxis })
  const xAxis2 = useAxis({ label: 'recall', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'precision', range: [0, 1], equal: xAxis2 })
  return (
    <Figure
      title={title}
      state={state}
      caption={
        caption ??
        'Scores are N(0, 1) for negatives and N(d, 1) for positives. Left: the ROC curve, true-positive rate against false-positive rate over all thresholds. Right: precision against recall. Drag the operating point along the ROC curve to move the threshold. Lower the prevalence: the ROC curve and AUROC do not move, while precision and average precision fall.'
      }

      readouts={
        <>
          <Readout label="AUROC" value={formatNumber(binormalAuc(state.d))} />
          <Readout label="average precision" value={formatNumber(ap)} />
          <Readout label="TPR" value={formatNumber(at.tpr)} />
          <Readout label="FPR" value={formatNumber(at.fpr)} />
          <Readout label="precision" value={formatNumber(precisionAt)} />
          {showEer && <Readout label="EER" value={formatNumber(eer)} />}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(roc)}
          <Handle
            kind="point"
            at={[at.fpr, at.tpr]}
            label="operating point"
            onDrag={(p) => state.set('t', c.thresholds[nearestIndex(c.fpr, c.tpr, p)])}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...pr[0]} />
          <Curve {...pr[1]} />
          <Points {...pr[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
