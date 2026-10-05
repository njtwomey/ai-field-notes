import { useMemo } from 'react'
import { Area, Figure, float, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { METRIC_LABELS, expectedCounts, metricsFrom, type MetricKey } from './binormal'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalPdf } from 'aifn-compute/numerics/special'

const N = 1000
const X = toFlat(linspace(-4, 7, 221))

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
  const state = useFigureState({
    pi: slider(0.01, 0.5, prevalence, { step: 0.01, label: 'prevalence π' }),
    d: float(separation, { min: 0, max: 4, step: 0.1, label: 'separation d' }),
    t: slider(-3, 6, threshold, { step: 0.05, label: 'threshold t' }),
  })

  const counts = expectedCounts(state.t, state.d, state.pi, N)
  const m = metricsFrom(counts)
  const series = useMemo(
    () =>
      [
        { name: 'negatives', x: X, y: X.map((x) => (1 - state.pi) * normalPdf(x)), slot: 0 },
        {
          name: 'positives',
          x: X,
          y: X.map((x) => state.pi * normalPdf(x - state.d)),
          slot: 1,
        },
      ] as const,
    [state.pi, state.d],
  )

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

  const xAxis = useAxis({ label: 'score', range: [-4, 7] })
  const yAxis = useAxis({ label: 'density × share', hold: 'union' })
  return (
    <Figure
      title={title}
      state={state}
      caption={
        caption ??
        'Negative scores follow N(0, 1) and positive scores N(d, 1); each density is scaled by its share of the population. Cases scoring above the threshold are predicted positive. Drag the threshold, or change the prevalence and the separation d, and watch the confusion matrix for 1,000 cases and the metrics below it.'
      }

      readouts={
        <>
          {metrics.map((k) => (
            <Readout key={k} label={METRIC_LABELS[k]} value={show(m[k])} />
          ))}
        </>
      }
    >
      <div className="grid items-center gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Area {...series[0]} />
          <Area {...series[1]} />
          <Handle {...state.handle('t', { label: 'threshold' })} />
        </Plot>
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
    </Figure>
  )
}
