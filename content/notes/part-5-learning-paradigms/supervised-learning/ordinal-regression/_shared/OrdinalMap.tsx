import { useMemo } from 'react'
import { Bars, Contours, Handle, Plot, Points, Raster, useAxis } from 'aifn-render'
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
  const { decision, expected, levels, ticks } = useMemo(() => {
    const decision = grid.map((y) => grid.map((x) => fitted.predict([x, y]) + 1))
    const expected = grid.map((y) => grid.map((x) => expectedClass(fitted.probs([x, y]))))
    const levels = Array.from({ length: d.k - 1 }, (_, k) => k + 1.5)
    const ticks = Array.from({ length: d.k }, (_, k) => k + 1)
    return { decision, expected, levels, ticks }
  }, [fitted, grid, d.k])

  // Training points in their class colours; the marker shape is the size of the model's error on the point.
  const points = useMemo(
    () => ({
      x: d.train.x.map((p) => p[0]),
      y: d.train.x.map((p) => p[1]),
      colors: d.train.y.map((y) => colors[y]),
      group: d.train.x.map((p, i) => Math.min(Math.abs(fitted.predict(p) - d.train.y[i]), 2)),
    }),
    [d, fitted, colors],
  )

  const probs = fitted.probs(query)
  const lowest = Math.min(0, ...probs)
  const classes = probs.map((_, k) => k + 1)

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  const kAxis = useAxis({ label: 'class k', range: [0.5, d.k + 0.5], integer: true })
  const pAxis = useAxis({ label: 'probability', range: [lowest < 0 ? Math.floor(lowest * 10) / 10 : 0, 1] })
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
      <Plot
        x={xAxis}
        y={yAxis}
        height={400}
        ariaLabel={'Expected or predicted class over the input plane, with contours and training points'}
      >
        <Raster
          x={grid}
          y={grid}
          z={fill === 'expectation' ? expected : decision}
          scale={CLASS_SCALE}
          range={[1, d.k]}
          scaleTicks={ticks}
          fillOpacity={0.45}
          valueLabel={fill === 'expectation' ? 'E[y | x]' : 'predicted class'}
        />
        <Contours x={grid} y={grid} z={expected} levels={levels} />
        <Points name="training point" {...points} groupNames={ERROR_GROUPS} live />
        <Handle kind="point" at={query} label="x" onDrag={setQuery} />
      </Plot>
      <Plot x={kAxis} y={pAxis} height={300} ariaLabel="Class probabilities at the query point">
        <Bars name="P(y = k | x)" x={classes} y={probs} colors={colors} />
      </Plot>
    </div>
  )
}
