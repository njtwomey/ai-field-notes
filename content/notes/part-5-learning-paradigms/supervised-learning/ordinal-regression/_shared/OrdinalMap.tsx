import { useMemo } from 'react'
import { Heatmap, XYChart, type Handle, type HeatmapOverlay, type XYSeries } from 'aifn-render'
import { CLASS_SCALE, useClassColors } from './classColor'
import { expectedClass, type Dataset, type Fitted, type Point } from './ordinal'
import { useGrid, type FillMode, type Resolution } from './useOrdinalData'

/** Marker shape by the size of the model's error on a training point (circle, square, triangle). */
const ERROR_GROUPS = ['exact', 'off by one', 'off by two or more']

type Props = {
  fitted: Fitted
  data: Dataset
  resolution: Resolution
  fill: FillMode
  query: Point
  setQuery: (p: Point) => void
}

/**
 * The shared view of a fitted ordinal model on 2-D data: the muted fill (expected class or decision) on the class
 * colour scale with contours of E[y | x] at k + ½, the training points in their class colours with marker shapes for
 * the model's error, and the class probabilities at a draggable query point.
 */
export function OrdinalMap({ fitted, data: d, resolution, fill, query, setQuery }: Props) {
  const grid = useGrid(d.range, resolution)
  const colors = useClassColors(d.k)

  // Both fills come from one pass over the grid, so switching between them only recolours.
  const { decision, expected, contours, ticks } = useMemo(() => {
    const decision = grid.map((y) => grid.map((x) => fitted.predict([x, y]) + 1))
    const expected = grid.map((y) => grid.map((x) => expectedClass(fitted.probs([x, y]))))
    const levels = Array.from({ length: d.k - 1 }, (_, k) => k + 1.5)
    const ticks = Array.from({ length: d.k }, (_, k) => k + 1)
    return { decision, expected, contours: { levels, field: expected }, ticks }
  }, [fitted, grid, d.k])

  const overlay = useMemo<HeatmapOverlay[]>(
    () => [
      {
        name: 'training point',
        type: 'scatter',
        x: d.train.x.map((p) => p[0]),
        y: d.train.x.map((p) => p[1]),
        values: d.train.y.map((y) => y + 1),
        group: d.train.x.map((p, i) => Math.min(Math.abs(fitted.predict(p) - d.train.y[i]), 2)),
        groupNames: ERROR_GROUPS,
      },
    ],
    [d, fitted],
  )

  const probs = fitted.probs(query)
  const lowest = Math.min(0, ...probs)
  const bars: XYSeries[] = [
    { name: 'P(y = k | x)', type: 'bar', x: probs.map((_, k) => k + 1), y: probs, pointColors: colors },
  ]
  const handles: Handle[] = [{ kind: 'point', at: query, label: 'x', onDrag: setQuery }]

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
      <Heatmap
        x={grid}
        y={grid}
        z={fill === 'expectation' ? expected : decision}
        range={[1, d.k]}
        scale={CLASS_SCALE}
        scaleTicks={ticks}
        fillOpacity={0.45}
        contours={contours}
        overlay={overlay}
        handles={handles}
        xLabel="x₁"
        yLabel="x₂"
        valueLabel={fill === 'expectation' ? 'E[y | x]' : 'predicted class'}
        height={400}
        ariaLabel="Expected or predicted class over the input plane, with contours and training points"
      />
      <XYChart
        series={bars}
        xRange={[0.5, d.k + 0.5]}
        yRange={[lowest < 0 ? Math.floor(lowest * 10) / 10 : 0, 1]}
        integerX
        xLabel="class k"
        yLabel="probability"
        height={300}
        ariaLabel="Class probabilities at the query point"
      />
    </div>
  )
}
