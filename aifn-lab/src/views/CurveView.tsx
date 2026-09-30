import { useMemo } from 'react'
import type {
  PrecisionRecallCurve,
  ReliabilityDiagram,
  RocCurve,
  costCurve,
  detCurve,
  gainCurve,
  precisionRecallGainCurve,
} from 'aifn/metrics'
import { normalQuantile } from 'aifn/special'
import { toFlat } from 'aifn/tensor'
import { Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { formatValue } from './format'
import type { FrameProps } from './frame'

/** Every typed curve `aifn/metrics` returns; each carries a `kind`. */
export type Curve =
  | RocCurve
  | PrecisionRecallCurve
  | ReliabilityDiagram
  | ReturnType<typeof detCurve>
  | ReturnType<typeof gainCurve>
  | ReturnType<typeof costCurve>
  | ReturnType<typeof precisionRecallGainCurve>

export type CurveViewProps = FrameProps & {
  /** One curve, or several of the same kind to compare (drawn in slots 0, 1, …). */
  curve: Curve | readonly { name: string; curve: Curve }[]
  /** A highlighted operating point, in the chart's own coordinates (e.g. [FPR, TPR] on a ROC curve). */
  point?: [number, number]
}

/**
 * Axis labels and fixed ranges per kind. Every kind with a range lives on the unit square: it is drawn with equal
 * units on exactly [0, 1]², so its plot area is square, the diagonal is at 45° and areas read truly.
 */
const AXES: Record<Curve['kind'], { x: string; y: string; range?: [number, number] }> = {
  roc: { x: 'false-positive rate', y: 'true-positive rate', range: [0, 1] },
  'precision-recall': { x: 'recall', y: 'precision', range: [0, 1] },
  reliability: { x: 'mean predicted probability', y: 'observed frequency', range: [0, 1] },
  det: { x: 'false-positive rate (probit)', y: 'false-negative rate (probit)' },
  gain: { x: 'fraction targeted', y: 'fraction of positives captured', range: [0, 1] },
  cost: { x: 'probability cost PC(+)', y: 'normalised expected cost', range: [0, 1] },
  'precision-recall-gain': { x: 'recall gain', y: 'precision gain', range: [0, 1] },
}

/** A probit axis value, clipped away from 0 and 1 so that the ends stay on the chart. */
const probit = (p: number) => normalQuantile(Math.min(0.999, Math.max(0.001, p)))

/** The (x, y) points of a curve, in the chart's coordinates. */
function points(c: Curve): { x: number[]; y: number[] } {
  switch (c.kind) {
    case 'roc':
      return { x: toFlat(c.fpr), y: toFlat(c.tpr) }
    case 'precision-recall':
      return { x: toFlat(c.recall), y: toFlat(c.precision) }
    case 'reliability': {
      const x = toFlat(c.meanPredicted)
      const y = toFlat(c.observed)
      const keep = x.map((v, i) => Number.isFinite(v) && Number.isFinite(y[i]))
      return { x: x.filter((_, i) => keep[i]), y: y.filter((_, i) => keep[i]) }
    }
    case 'det':
      return { x: toFlat(c.fpr).map(probit), y: toFlat(c.fnr).map(probit) }
    case 'gain':
      return { x: toFlat(c.fraction), y: toFlat(c.gain) }
    case 'cost':
      return { x: toFlat(c.probabilityCost), y: toFlat(c.normalisedCost) }
    case 'precision-recall-gain':
      return { x: toFlat(c.recallGain), y: toFlat(c.precisionGain) }
  }
}

/** The chance or ideal reference line of a kind, if it has one. */
function reference(c: Curve): XYSeries | null {
  switch (c.kind) {
    case 'roc':
    case 'gain':
      return { name: 'chance', type: 'line', x: [0, 1], y: [0, 1], muted: true, dashed: true }
    case 'reliability':
      return { name: 'calibrated', type: 'line', x: [0, 1], y: [0, 1], muted: true, dashed: true }
    case 'precision-recall':
      return {
        name: 'chance (π)',
        type: 'line',
        x: [0, 1],
        y: [c.prevalence, c.prevalence],
        muted: true,
        dashed: true,
      }
    case 'cost':
      // The trivial classifiers: always negative costs PC(+), always positive 1 − PC(+).
      return { name: 'trivial', type: 'line', x: [0, 0.5, 1], y: [0, 0.5, 0], muted: true, dashed: true }
    default:
      return null
  }
}

/** The summary number a kind reports, with its label. */
function summary(c: Curve): [string, number] | null {
  switch (c.kind) {
    case 'roc':
      return ['AUROC', c.auc]
    case 'precision-recall':
      return ['average precision', c.averagePrecision]
    case 'reliability':
      return ['ECE', c.ece]
    case 'cost':
      return ['area', c.area]
    default:
      return null
  }
}

/** The curves of one kind with their reference line and an optional operating point, as chart series. */
function useCurveSeries(
  list: readonly { name: string; curve: Curve }[],
  point?: [number, number],
  slot = 0,
  extra?: { name: string; curve: Curve },
) {
  const [px, py] = point ?? []
  return useMemo(() => {
    const out: XYSeries[] = []
    const ref = list.length ? reference(list[0].curve) : null
    if (ref) out.push(ref)
    if (extra) {
      const p = points(extra.curve)
      out.push({ name: extra.name, type: 'line', x: p.x, y: p.y, muted: true })
    }
    list.forEach(({ name, curve }, k) => {
      const p = points(curve)
      out.push({ name, type: 'line', x: p.x, y: p.y, slot: slot + k, showPoints: curve.kind === 'reliability' })
    })
    if (px !== undefined && py !== undefined)
      out.push({ name: 'operating point', type: 'scatter', x: [px], y: [py], emphasis: true })
    return out
  }, [list, px, py, slot, extra])
}

export type CurveChartProps = {
  /** One curve, or several of the same kind (drawn in slots 0, 1, …). Pass a memoised value. */
  curve: Curve | readonly { name: string; curve: Curve }[]
  /** A highlighted operating point in the chart's coordinates, redrawn as a patch (e.g. while a threshold moves). */
  point?: [number, number]
  /** Draggable handles, e.g. the operating point itself, bound to a threshold. */
  handles?: Handle[]
  /** The legend names each curve by its summary (e.g. "ROC, AUROC 0.93") rather than by name alone. */
  summaryInLegend?: boolean
  /** The palette slot of the (first) curve (default 0), e.g. to keep slots that mean classes elsewhere free. */
  slot?: number
  /** A reference curve of the same kind drawn faint under the others, e.g. the Bayes-optimal classifier's. */
  reference?: { name: string; curve: Curve }
}

/**
 * The chart of a typed curve without a figure frame, for dashboards: the curve with its chance or ideal line and an
 * operating point. Unit-square kinds (ROC, PR, gain, cost, PRG) are drawn with equal units on exactly [0, 1]², so the
 * plot area is square; put the chart in a square cell (`DashboardCell aspect="square"`).
 */
export function CurveChart({
  curve,
  point,
  handles,
  summaryInLegend = false,
  slot = 0,
  reference: extra,
}: CurveChartProps) {
  const list = useMemo(() => {
    const raw = 'kind' in curve ? [{ name: kindName(curve as Curve), curve: curve as Curve }] : [...curve]
    if (!summaryInLegend) return raw
    return raw.map(({ name, curve: c }) => {
      const s = summary(c)
      const short = s?.[0] === 'average precision' ? 'AP' : s?.[0]
      return { name: s ? `${name}, ${short} ${formatValue(Number(s[1].toPrecision(3)))}` : name, curve: c }
    })
  }, [curve, summaryInLegend])
  const series = useCurveSeries(list, undefined, slot, extra)
  const kind = list[0]?.curve.kind ?? 'roc'
  const axes = AXES[kind]
  const [px, py] = point ?? []
  const live = useMemo(
    (): XYSeries[] =>
      px !== undefined && py !== undefined && !handles
        ? [{ name: 'operating point', type: 'scatter', x: [px], y: [py], emphasis: true }]
        : [],
    [px, py, handles],
  )
  return (
    <XYChart
      series={series}
      live={live}
      handles={handles}
      xLabel={axes.x}
      yLabel={axes.y}
      xRange={axes.range}
      yRange={axes.range ?? [undefined, undefined]}
      aspect={axes.range ? 'equal' : 'fit'}
      zoom={!axes.range}
      legend
    />
  )
}

/** A short name for a curve's kind, for its legend. */
function kindName(c: Curve): string {
  const names: Record<Curve['kind'], string> = {
    roc: 'ROC',
    'precision-recall': 'PR',
    reliability: 'reliability',
    det: 'DET',
    gain: 'gain',
    cost: 'cost',
    'precision-recall-gain': 'PRG',
  }
  return names[c.kind]
}

/**
 * A typed curve from `aifn/metrics` (ROC, precision–recall, reliability, DET, gain, cost, precision–recall–gain) as a
 * figure: the curve with its chance or ideal line, an optional operating point, and its summary number (AUROC, AP,
 * ECE, area) as a readout. A reliability diagram adds the bin counts as a panel beneath. Several curves of one kind
 * are drawn together for comparison.
 */
export function CurveView(props: CurveViewProps) {
  const list = useMemo(
    () => ('kind' in props.curve ? [{ name: 'curve', curve: props.curve as Curve }] : [...props.curve]),
    [props.curve],
  )
  const kind = list[0]?.curve.kind ?? 'roc'
  const axes = AXES[kind]
  const series = useCurveSeries(list, props.point)
  const counts = kind === 'reliability' ? (list[0].curve as ReliabilityDiagram) : null
  const countSeries = useMemo((): XYSeries[] => {
    if (!counts) return []
    const e = toFlat(counts.edges)
    const c = toFlat(counts.counts)
    return [{ name: 'cases per bin', type: 'bar', x: c.map((_, m) => (e[m] + e[m + 1]) / 2), y: c, muted: true }]
  }, [counts])
  const readouts = (
    <>
      {props.readouts}
      {list.map(({ name, curve }) => {
        const s = summary(curve)
        return s ? (
          <Readout key={name} label={list.length > 1 ? `${s[0]} (${name})` : s[0]} value={formatValue(s[1])} />
        ) : null
      })}
    </>
  )
  return (
    <Figure
      title={props.title ?? `${kind} curve`}
      id={props.id}
      description={props.description}
      controls={props.controls}
      readouts={readouts}
      caption={props.caption}
      defaultSize={props.defaultSize}
    >
      {counts ? (
        // The counts share the diagram's probability axis, so bins line up with their points: a short strip right
        // under the square diagram. Both ranges are fixed, so neither panel has zoom controls.
        <Subplots rows={2} sharex heightRatios={[1, 0.22]} ratiosOf="equal" toolbar={false} tight hoverGroup>
          <Panel aspect="equal">
            <XYChart
              series={series}
              xLabel={axes.x}
              yLabel={axes.y}
              xRange={axes.range}
              yRange={axes.range}
              zoom={false}
            />
          </Panel>
          <Panel>
            <XYChart series={countSeries} xLabel="predicted probability" yLabel="cases" xRange={[0, 1]} zoom={false} />
          </Panel>
        </Subplots>
      ) : (
        <XYChart
          series={series}
          xLabel={axes.x}
          yLabel={axes.y}
          xRange={axes.range}
          yRange={axes.range ?? [undefined, undefined]}
          aspect={axes.range ? 'equal' : 'fit'}
          zoom={!axes.range}
        />
      )}
    </Figure>
  )
}
