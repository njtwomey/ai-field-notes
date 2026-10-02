import { useMemo } from 'react'
import type { Curve, CurveName } from 'aifn/foundation/contracts'
import type { ReliabilityDiagram } from 'aifn/learning/metrics'
import { normalQuantile } from 'aifn/numerics/special'
import { toFlat } from 'aifn/foundation/tensor'
import { PanelSlot } from '@lab/layout'
import { Bars, Curve as CurveLayer, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { formatValue } from './format'
import { registerKind, registerView } from './registry'

export type CurvePanelProps = {
  /** One curve, or several of the same kind to compare (drawn in slots 0, 1, …). */
  curve: Curve | readonly { name: string; curve: Curve }[]
  /** A highlighted operating point, in the chart's own coordinates (e.g. [FPR, TPR] on a ROC curve). */
  point?: [number, number]
}

/**
 * Axis labels and fixed ranges per kind. Every kind with a range lives on the unit square: it is drawn with equal
 * units on exactly [0, 1]², so its plot area is square, the diagonal is at 45° and areas read truly.
 */
const AXES: Record<CurveName, { x: string; y: string; range?: [number, number] }> = {
  roc: { x: 'false-positive rate', y: 'true-positive rate', range: [0, 1] },
  pr: { x: 'recall', y: 'precision', range: [0, 1] },
  reliability: { x: 'mean predicted probability', y: 'observed frequency', range: [0, 1] },
  calibration: { x: 'mean predicted probability', y: 'observed frequency', range: [0, 1] },
  det: { x: 'false-positive rate (probit)', y: 'false-negative rate (probit)' },
  gain: { x: 'fraction targeted', y: 'fraction of positives captured', range: [0, 1] },
  cost: { x: 'probability cost PC(+)', y: 'normalised expected cost', range: [0, 1] },
  prg: { x: 'recall gain', y: 'precision gain', range: [0, 1] },
}

/** A probit axis value, clipped away from 0 and 1 so that the ends stay on the chart. */
const probit = (p: number) => normalQuantile(Math.min(0.999, Math.max(0.001, p)))

/** The (x, y) points of a curve, in the chart's coordinates. */
function points(c: Curve): { x: number[]; y: number[] } {
  const x = toFlat(c.x)
  const y = toFlat(c.y)
  if (c.curve === 'det') return { x: x.map(probit), y: y.map(probit) }
  // Binned curves hold NaN in empty bins.
  const keep = x.map((v, i) => Number.isFinite(v) && Number.isFinite(y[i]))
  return { x: x.filter((_, i) => keep[i]), y: y.filter((_, i) => keep[i]) }
}

/** The chance or ideal reference line of a kind, if it has one. */
function reference(c: Curve): { name: string; x: number[]; y: number[] } | null {
  switch (c.curve) {
    case 'roc':
    case 'gain':
      return { name: 'chance', x: [0, 1], y: [0, 1] }
    case 'reliability':
    case 'calibration':
      return { name: 'calibrated', x: [0, 1], y: [0, 1] }
    case 'pr':
      return c.prevalence === undefined ? null : { name: 'chance (π)', x: [0, 1], y: [c.prevalence, c.prevalence] }
    case 'cost':
      // The trivial classifiers: always negative costs PC(+), always positive 1 − PC(+).
      return { name: 'trivial', x: [0, 0.5, 1], y: [0, 0.5, 0] }
    default:
      return null
  }
}

/** The summary number a kind reports, with its label. */
function summary(c: Curve): [string, number] | null {
  if (c.curve === 'reliability' && 'ece' in c) return ['ECE', (c as ReliabilityDiagram).ece]
  if (c.area === undefined) return null
  const label: Partial<Record<CurveName, string>> = { roc: 'AUROC', pr: 'average precision' }
  return [label[c.curve] ?? 'area', c.area]
}

/** The curves of one kind with their reference line and an optional faint reference curve, as Plot layers. */
function useCurveLayers(
  list: readonly { name: string; curve: Curve }[],
  slot = 0,
  extra?: { name: string; curve: Curve },
) {
  return useMemo(() => {
    const ref = list.length ? reference(list[0].curve) : null
    const faint = extra ? points(extra.curve) : null
    const drawn = list.map(({ name, curve }) => ({
      name,
      reliability: curve.curve === 'reliability',
      ...points(curve),
    }))
    return (
      <>
        {ref && <CurveLayer id="reference" name={ref.name} x={ref.x} y={ref.y} muted dashed />}
        {faint && extra && <CurveLayer id="extra" name={extra.name} x={faint.x} y={faint.y} muted />}
        {drawn.map((c, k) => (
          <CurveLayer
            key={c.name}
            id={`curve${k}`}
            name={c.name}
            x={c.x}
            y={c.y}
            slot={slot + k}
            showPoints={c.reliability}
          />
        ))}
      </>
    )
  }, [list, slot, extra])
}

/** The two axes of a kind: fixed unit-square kinds have equal units and no zoom. */
function useCurveAxes(kind: CurveName) {
  const axes = AXES[kind]
  const x = useAxis({ label: axes.x, range: axes.range, zoom: !axes.range })
  const y = useAxis({ label: axes.y, range: axes.range, zoom: !axes.range, equal: axes.range ? x : undefined })
  return { x, y }
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
  const layers = useCurveLayers(list, slot, extra)
  const kind = list[0]?.curve.curve ?? 'roc'
  const axes = useCurveAxes(kind)
  const [px, py] = point ?? []
  return (
    <Plot x={axes.x} y={axes.y} legend>
      {layers}
      {px !== undefined && py !== undefined && !handles && (
        <Points id="point" name="operating point" x={[px]} y={[py]} emphasis live />
      )}
      {handles?.map((h, i) => (
        <Handle key={i} {...h} />
      ))}
    </Plot>
  )
}

/** A short name for a curve's kind, for its legend. */
function kindName(c: Curve): string {
  const names: Record<CurveName, string> = {
    roc: 'ROC',
    pr: 'PR',
    reliability: 'reliability',
    calibration: 'calibration',
    det: 'DET',
    gain: 'gain',
    cost: 'cost',
    prg: 'PRG',
  }
  return names[c.curve]
}

/**
 * A typed curve from `aifn/learning/metrics` (ROC, precision–recall, reliability, DET, gain, cost, precision–recall–gain) as a
 * figure: the curve with its chance or ideal line, an optional operating point, and its summary number (AUROC, AP,
 * ECE, area) as a readout. A reliability diagram adds the bin counts as a panel beneath. Several curves of one kind
 * are drawn together for comparison.
 */
export function CurvePanel(props: CurvePanelProps) {
  const list = useMemo(
    () => ('kind' in props.curve ? [{ name: 'curve', curve: props.curve as Curve }] : [...props.curve]),
    [props.curve],
  )
  const kind = list[0]?.curve.curve ?? 'roc'
  const layers = useCurveLayers(list)
  const axes = useCurveAxes(kind)
  const [px, py] = props.point ?? []
  const first = list[0]?.curve
  const counts = first && 'counts' in first ? (first as ReliabilityDiagram) : null
  const bins = useMemo(() => {
    if (!counts) return null
    const e = toFlat(counts.edges)
    const c = toFlat(counts.counts)
    return { edges: e, x: c.map((_, m) => (e[m] + e[m + 1]) / 2), y: c }
  }, [counts])
  const casesAxis = useAxis({ label: 'cases', zoom: false })
  const point =
    px !== undefined && py !== undefined ? (
      <Points id="point" name="operating point" x={[px]} y={[py]} emphasis />
    ) : null
  const readouts = (
    <>
      {list.map(({ name, curve }) => {
        const s = summary(curve)
        return s ? (
          <Readout key={name} label={list.length > 1 ? `${s[0]} (${name})` : s[0]} value={formatValue(s[1])} />
        ) : null
      })}
    </>
  )
  return (
    <>
      <PanelSlot slot="readouts">{readouts}</PanelSlot>
      {bins ? (
        // The counts share the diagram's probability axis, so bins line up with their points: a short strip right
        // under the square diagram. Both ranges are fixed, so neither panel has zoom controls.
        <Plots rows={2} heights={[1, 0.22]} ratiosOf="equal" toolbar={false} tight hoverGroup>
          <Plot x={axes.x} y={axes.y}>
            {layers}
            {point}
          </Plot>
          <Plot x={axes.x} y={casesAxis}>
            <Bars name="cases per bin" x={bins.x} y={bins.y} edges={bins.edges} muted />
          </Plot>
        </Plots>
      ) : (
        <Plot x={axes.x} y={axes.y}>
          {layers}
          {point}
        </Plot>
      )}
    </>
  )
}

registerKind(
  'curve',
  (o) => typeof o === 'object' && o !== null && (o as Curve).kind === 'curve' && (o as Curve).curve in AXES,
)
registerView<Curve>({
  key: 'curve/unit-square',
  kind: 'curve',
  description:
    'A typed evaluation curve (ROC, PR, reliability, DET, gain, cost, PRG) with its reference line and summary number.',
  title: (c) => `${kindName(c)} curve`,
  render: (c) => <CurvePanel curve={c} />,
})
