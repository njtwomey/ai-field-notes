import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { lineInSquare } from '../../_shared/rocExample'
import type { Point } from '../../_shared/calibration'

type Metric = 'accuracy' | 'precision' | 'f1'
const LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

/** Metric value at (fpr, tpr) for skew ratio c = NEG/POS (Flach 2003, Table 2). */
const value = (metric: Metric, [f, t]: Point, c: number) =>
  metric === 'accuracy'
    ? (t + c * (1 - f)) / (1 + c)
    : metric === 'precision'
      ? t / (t + c * f)
      : (2 * t) / (t + c * f + 1)

/** The isometric of `metric` at level v: a point on it and its slope in ROC space. */
function isometric(metric: Metric, v: number, c: number): { at: Point; slope: number } {
  if (metric === 'accuracy') return { at: [0, v * (1 + c) - c], slope: c }
  if (metric === 'precision') return { at: [0, 0], slope: (v * c) / (1 - v) }
  return { at: [0, v / (2 - v)], slope: (v * c) / (2 - v) }
}

/**
 * ROC isometrics: contour lines of accuracy, precision or F1 in ROC space for a given skew ratio c. Accuracy
 * isometrics are parallel; precision isometrics rotate about the origin; F1 isometrics rotate about (−1/c, 0).
 */
export function IsometricPlot() {
  const state = useFigureState({
    metric: choice<Metric>(
      [
        { value: 'accuracy', label: 'accuracy' },
        { value: 'precision', label: 'precision' },
        { value: 'f1', label: 'F1' },
      ],
      'accuracy',
      { label: 'metric' },
    ),
    logC: float(0, { min: -3, max: 3, step: 0.1, label: 'log₂ c' }),
    fpr: slider(0.001, 1, 0.2, { step: 0.001, onChart: true }),
    tpr: slider(0, 1, 0.7, { step: 0.001, onChart: true }),
  })
  const c = 2 ** state.logC
  const here: Point = [state.fpr, state.tpr]
  const v = value(state.metric, here, c)
  const through = isometric(state.metric, Math.min(Math.max(v, 1e-6), 1 - 1e-6), c)
  const own = lineInSquare(through.at, through.slope)

  const segments: Segment[] = LEVELS.flatMap((lv) => {
    const iso = isometric(state.metric, lv, c)
    const seg = lineInSquare(iso.at, iso.slope)
    return seg.x.length === 2 ? [{ from: [seg.x[0], seg.y[0]], to: [seg.x[1], seg.y[1]] } as Segment] : []
  })
  const series = [
    { name: 'isometric through the point', x: own.x, y: own.y, slot: 0 },
    { name: 'classifier', x: [state.fpr], y: [state.tpr], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'false-positive rate', range: [0, 1] })
  const yAxis = useAxis({ label: 'true-positive rate', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="ROC isometrics"
      state={state}
      caption="Thin lines join ROC points with equal metric value, at levels 0.1 to 0.9, for skew ratio c (negatives per positive, times the cost ratio). Accuracy isometrics are parallel with slope c. Precision isometrics all pass through the origin. F1 isometrics all pass through (−1/c, 0), off the chart to the left. Drag the classifier and change c: the slope of its isometric is the trade-off the metric makes at that point."

      readouts={
        <>
          <Readout label="c" value={formatNumber(c)} />
          <Readout label={state.metric} value={formatNumber(v)} />
          <Readout label="effective skew (slope)" value={formatNumber(through.slope)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
        <Segments segments={segments} />
        <Handle
          kind="point"
          at={here}
          label="classifier"
          onDrag={([x, y]) => {
            state.set('fpr', x)
            state.set('tpr', y)
          }}
        />
      </Plot>
    </Figure>
  )
}
