import { useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
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
  const [metric, setMetric] = useState<Metric>('accuracy')
  const logC = useParam(0, { min: -3, max: 3, step: 0.1 })
  const fpr = useParam(0.2, { min: 0.001, max: 1, step: 0.001 })
  const tpr = useParam(0.7, { min: 0, max: 1, step: 0.001 })
  const c = 2 ** logC.value
  const here: Point = [fpr.value, tpr.value]
  const v = value(metric, here, c)
  const through = isometric(metric, Math.min(Math.max(v, 1e-6), 1 - 1e-6), c)
  const own = lineInSquare(through.at, through.slope)

  const segments: Segment[] = LEVELS.flatMap((lv) => {
    const iso = isometric(metric, lv, c)
    const seg = lineInSquare(iso.at, iso.slope)
    return seg.x.length === 2 ? [{ from: [seg.x[0], seg.y[0]], to: [seg.x[1], seg.y[1]] } as Segment] : []
  })
  const series: XYSeries[] = [
    { name: 'isometric through the point', type: 'line', x: own.x, y: own.y, slot: 0 },
    { name: 'classifier', type: 'scatter', x: [fpr.value], y: [tpr.value], emphasis: true },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: here,
      label: 'classifier',
      onDrag: ([x, y]) => {
        fpr.set(x)
        tpr.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="ROC isometrics"
      caption="Thin lines join ROC points with equal metric value, at levels 0.1 to 0.9, for skew ratio c (negatives per positive, times the cost ratio). Accuracy isometrics are parallel with slope c. Precision isometrics all pass through the origin. F1 isometrics all pass through (−1/c, 0), off the chart to the left. Drag the classifier and change c: the slope of its isometric is the trade-off the metric makes at that point."
      controls={
        <>
          <ParamChoice
            label="metric"
            value={metric}
            onChange={setMetric}
            options={[
              { value: 'accuracy', label: 'accuracy' },
              { value: 'precision', label: 'precision' },
              { value: 'f1', label: 'F1' },
            ]}
          />
          <ParamSlider label="log₂ c" param={logC} />
        </>
      }
      readout={
        <>
          <Readout label="c" value={formatNumber(c)} />
          <Readout label={metric} value={formatNumber(v)} />
          <Readout label="effective skew (slope)" value={formatNumber(through.slope)} />
        </>
      }
    >
      <XYChart
        equalAspect
        xLabel="false-positive rate"
        yLabel="true-positive rate"
        xRange={[0, 1]}
        yRange={[0, 1]}
        handles={handles}
        segments={segments}
        series={series}
      />
    </Interactive>
  )
}
