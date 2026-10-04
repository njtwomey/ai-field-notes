import { useMemo, useState } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  Vectors,
} from 'aifn-render'

type Vec = [number, number]
const R = 4
/** Grid lines are drawn for coordinates −N … N along each basis vector. */
const N = 8
const clamp = (v: number) => Math.round(Math.min(Math.max(v, -R), R) * 20) / 20

/** Coordinates of b in the basis v₁, v₂, with the skewed grid of integer combinations. */
export function BasisCoordinates() {
  const [v1, setV1] = useState<Vec>([1.5, 0.5])
  const [v2, setV2] = useState<Vec>([0.5, 1.2])
  const [b, setB] = useState<Vec>([2.5, 3])

  const r = useMemo(() => {
    const det = v1[0] * v2[1] - v2[0] * v1[1]
    // A basis only if |det| is not tiny; otherwise the two vectors span a line.
    const independent = Math.abs(det) > 1e-3
    const c: Vec | undefined = independent
      ? [(b[0] * v2[1] - v2[0] * b[1]) / det, (v1[0] * b[1] - b[0] * v1[1]) / det]
      : undefined
    const grid: Segment[] = []
    if (independent) {
      for (let i = -N; i <= N; i++) {
        grid.push({
          from: [i * v1[0] - N * v2[0], i * v1[1] - N * v2[1]],
          to: [i * v1[0] + N * v2[0], i * v1[1] + N * v2[1]],
        })
        grid.push({
          from: [i * v2[0] - N * v1[0], i * v2[1] - N * v1[1]],
          to: [i * v2[0] + N * v1[0], i * v2[1] + N * v1[1]],
        })
      }
    }
    const series: SeriesSpec[] = [{ name: 'b', type: 'scatter', x: [b[0]], y: [b[1]], slot: 2 }]
    if (c) {
      // The route to b: c₁ v₁ along the first basis vector, then c₂ v₂.
      const corner: Vec = [c[0] * v1[0], c[0] * v1[1]]
      series.push({
        name: 'c₁ v₁ + c₂ v₂',
        type: 'line',
        x: [0, corner[0], b[0]],
        y: [0, corner[1], b[1]],
        slot: 2,
        dashed: true,
      })
    } else {
      // Dependent vectors span only a line (or just the origin).
      const d = Math.hypot(...v1) > 1e-9 ? v1 : v2
      const n = Math.hypot(...d) || 1
      series.push({
        name: 'span',
        type: 'line',
        x: [(-3 * R * d[0]) / n, (3 * R * d[0]) / n],
        y: [(-3 * R * d[1]) / n, (3 * R * d[1]) / n],
        slot: 0,
      })
    }
    return { det, c, grid, series }
  }, [v1, v2, b])

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Coordinates in a basis"
      caption="Drag the tips of v₁ and v₂, and the point b. The grey grid marks the integer combinations of v₁ and v₂. The dashed route reaches b by going c₁ times along v₁, then c₂ times along v₂; (c₁, c₂) are the coordinates of b in this basis. Line the two vectors up and the grid collapses: the vectors become dependent, their span shrinks to a line, and most points b can no longer be reached."
      readouts={
        <>
          <Readout label="c₁" value={r.c ? formatNumber(r.c[0]) : 'none'} />
          <Readout label="c₂" value={r.c ? formatNumber(r.c[1]) : 'none'} />
          <Readout label="det [v₁ v₂]" value={formatNumber(r.det)} />
          <Readout label="span" value={r.c ? 'the plane' : 'a line'} />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(r.series)}
          <Segments segments={r.grid} />
          <Vectors
            vectors={[
              { from: [0, 0], to: v1 },
              { from: [0, 0], to: v2 },
            ]}
          />
          <Handle kind="point" at={v1} label="v₁" onDrag={([x, y]) => setV1([clamp(x), clamp(y)])} />
          <Handle kind="point" at={v2} label="v₂" onDrag={([x, y]) => setV2([clamp(x), clamp(y)])} />
          <Handle kind="point" at={b} label="b" onDrag={([x, y]) => setB([clamp(x), clamp(y)])} />
        </Plot>
      </div>
    </Figure>
  )
}
