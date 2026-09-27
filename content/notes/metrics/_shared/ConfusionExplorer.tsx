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
import { normalPdf } from '@/lib/math/special'
import { METRIC_LABELS, expectedCounts, metricsFrom, type MetricKey } from './binormal'

const N = 1000
const X = linspace(-4, 7, 221)

const show = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '—')

/**
 * Two class-conditional score densities, each scaled by its share of the population, a draggable decision threshold,
 * the resulting confusion matrix for 1,000 cases, and the chosen metrics. Scaling by prevalence makes imbalance
 * visible: at 1% the positive density is a sliver under the negative one.
 */
export function ConfusionExplorer({
  title = 'A confusion matrix from a threshold',
  caption,
  metrics,
  prevalence = 0.05,
  separation = 2,
  threshold = 1.5,
}: {
  title?: string
  caption?: string
  metrics: MetricKey[]
  prevalence?: number
  separation?: number
  threshold?: number
}) {
  const pi = useParam(prevalence, { min: 0.01, max: 0.5, step: 0.01 })
  const d = useParam(separation, { min: 0, max: 4, step: 0.1 })
  const t = useParam(threshold, { min: -3, max: 6, step: 0.05 })

  const counts = expectedCounts(t.value, d.value, pi.value, N)
  const m = metricsFrom(counts)
  const series = useMemo(
    (): XYSeries[] => [
      { name: 'negatives', type: 'line', x: X, y: X.map((x) => (1 - pi.value) * normalPdf(x)), slot: 0, area: true },
      {
        name: 'positives',
        type: 'line',
        x: X,
        y: X.map((x) => pi.value * normalPdf(x - d.value)),
        slot: 1,
        area: true,
      },
    ],
    [pi.value, d.value],
  )
  const handles: Handle[] = [{ kind: 'x', at: t.value, label: 'threshold', onDrag: (x) => t.set(x) }]

  const cell = (label: string, value: number, tone: 'right' | 'wrong') => (
    <div
      className={
        'flex flex-col items-center justify-center rounded-md border px-3 py-2 ' +
        (tone === 'right' ? 'bg-muted/60' : 'bg-destructive/10')
      }
    >
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-lg tabular-nums">{Math.round(value)}</span>
    </div>
  )

  return (
    <Interactive
      title={title}
      caption={
        caption ??
        'Negative scores follow N(0, 1) and positive scores N(d, 1); each density is scaled by its share of the population. Cases scoring above the threshold are predicted positive. Drag the threshold, or change the prevalence and the separation d, and watch the confusion matrix for 1,000 cases and the metrics below it.'
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
          {metrics.map((k) => (
            <Readout key={k} label={METRIC_LABELS[k]} value={show(m[k])} />
          ))}
        </>
      }
    >
      <div className="grid items-center gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <XYChart
          series={series}
          xLabel="score"
          yLabel="density × share"
          xRange={[-4, 7]}
          handles={handles}
          height={260}
        />
        <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center text-xs">
          <span />
          <span className="text-muted-foreground">predicted +</span>
          <span className="text-muted-foreground">predicted −</span>
          <span className="self-center text-muted-foreground">actual +</span>
          {cell('TP', counts.tp, 'right')}
          {cell('FN', counts.fn, 'wrong')}
          <span className="self-center text-muted-foreground">actual −</span>
          {cell('FP', counts.fp, 'wrong')}
          {cell('TN', counts.tn, 'right')}
        </div>
      </div>
    </Interactive>
  )
}
