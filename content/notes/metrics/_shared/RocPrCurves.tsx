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
  const pi = useParam(prevalence, { min: 0.01, max: 0.5, step: 0.01 })
  const d = useParam(separation, { min: 0, max: 4, step: 0.1 })
  const t = useParam(separation / 2, { min: -4, max: 8, step: 0.01 })

  const c = useMemo(() => curves(d.value, pi.value), [d.value, pi.value])
  const ap = useMemo(() => averagePrecision(d.value, pi.value), [d.value, pi.value])
  const at = rates(t.value, d.value)
  const precisionAt = (pi.value * at.tpr) / (pi.value * at.tpr + (1 - pi.value) * at.fpr || 1)
  const eer = binormalEer(d.value)

  const roc: XYSeries[] = [
    { name: 'chance', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'ROC curve', type: 'line', x: c.fpr, y: c.tpr, slot: 0 },
    ...(showEer
      ? [
          { name: 'FPR = FNR', type: 'line' as const, x: [0, 1], y: [1, 0], dashed: true, slot: 2 },
          { name: 'equal error rate', type: 'scatter' as const, x: [eer], y: [1 - eer], slot: 2 },
        ]
      : []),
  ]
  const pr: XYSeries[] = [
    { name: 'chance (precision = π)', type: 'line', x: [0, 1], y: [pi.value, pi.value], dashed: true, muted: true },
    { name: 'PR curve', type: 'line', x: c.tpr, y: c.precision, slot: 1 },
    { name: 'operating point', type: 'scatter', x: [at.tpr], y: [precisionAt], emphasis: true },
  ]
  // The operating point is a location on the ROC curve: dragging moves it along the curve, which sets the threshold.
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [at.fpr, at.tpr],
      label: 'operating point',
      onDrag: (p) => t.set(c.thresholds[nearestIndex(c.fpr, c.tpr, p)]),
    },
  ]

  return (
    <Interactive
      title={title}
      caption={
        caption ??
        'Scores are N(0, 1) for negatives and N(d, 1) for positives. Left: the ROC curve, true-positive rate against false-positive rate over all thresholds. Right: precision against recall. Drag the operating point along the ROC curve to move the threshold. Lower the prevalence: the ROC curve and AUROC do not move, while precision and average precision fall.'
      }
      controls={
        <>
          <ParamSlider label="prevalence π" param={pi} />
          <ParamSlider label="separation d" param={d} />
          <ParamSlider label="threshold t" param={t} />
        </>
      }
      readout={
        <>
          <Readout label="AUROC" value={formatNumber(binormalAuc(d.value))} />
          <Readout label="average precision" value={formatNumber(ap)} />
          <Readout label="TPR" value={formatNumber(at.tpr)} />
          <Readout label="FPR" value={formatNumber(at.fpr)} />
          <Readout label="precision" value={formatNumber(precisionAt)} />
          {showEer && <Readout label="EER" value={formatNumber(eer)} />}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={roc}
          xLabel="false-positive rate"
          yLabel="true-positive rate"
          xRange={[0, 1]}
          yRange={[0, 1]}
          equalAspect
          handles={handles}
        />
        <XYChart series={pr} xLabel="recall" yLabel="precision" xRange={[0, 1]} yRange={[0, 1]} equalAspect />
      </div>
    </Interactive>
  )
}
