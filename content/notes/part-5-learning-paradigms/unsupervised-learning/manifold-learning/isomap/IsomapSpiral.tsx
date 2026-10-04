import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { classicalMds, geodesics, knnEdges, spiral } from './isomap'

const N = 100
/** An edge that skips more than this fraction of the spiral jumps between turns: a short circuit. */
const SHORT_CIRCUIT = 0.15

export function IsomapSpiral() {
  const data = useMemo(() => spiral(N, 2), [])
  const state = useFigureState({
    k: int(5, { min: 3, max: 20, step: 1, label: 'k (neighbours)' }),
  })
  const fit = useMemo(() => {
    const edges = knnEdges(data.points, state.k)
    const g = geodesics(N, edges)
    const finite = g.flat().filter((v) => v < Infinity)
    const connected = finite.length === N * N
    // A disconnected graph has infinite geodesics; cap them so the embedding still renders, and say so in the readout.
    const cap = 1.5 * Math.max(...finite)
    const { coords } = classicalMds(
      g.map((row) => row.map((v) => (v === Infinity ? cap : v))),
      1,
    )
    const z = coords.map((c) => c[0])
    // Fix the sign so that the embedding increases along the spiral.
    const sign = z[N - 1] >= z[0] ? 1 : -1
    const shortCircuits = edges.filter(([i, j]) => Math.abs(data.t[i] - data.t[j]) > SHORT_CIRCUIT).length
    const segments: Segment[] = edges.map(([i, j]) => ({ from: data.points[i], to: data.points[j] }))
    return { z: z.map((v) => sign * v), connected, shortCircuits, segments, edges: edges.length }
  }, [data, state.k])
  const arcMean = data.arc.reduce((a, b) => a + b, 0) / N
  const zMean = fit.z.reduce((a, b) => a + b, 0) / N
  const corr =
    fit.z.reduce((s, v, i) => s + (v - zMean) * (data.arc[i] - arcMean), 0) /
    Math.sqrt(fit.z.reduce((s, v) => s + (v - zMean) ** 2, 0) * data.arc.reduce((s, v) => s + (v - arcMean) ** 2, 0))

  const xAxis = useAxis({ label: 'x₁', range: [-3, 3] })
  const yAxis = useAxis({ label: 'x₂', range: [-3, 3], equal: xAxis })
  const xAxis2 = useAxis({ label: 'arc length along the spiral', hold: 'union' })
  const yAxis2 = useAxis({ label: 'Isomap coordinate', hold: 'union' })
  return (
    <Figure
      title="Isomap unrolls a spiral, until the graph short-circuits"
      state={state}
      caption="Left: 100 points on a spiral and their k-nearest-neighbour graph. Right: the one-dimensional Isomap coordinate of each point against its arc length along the spiral. With k from 5 to 7 the graph follows the curve, graph distances approximate arc length, and the points fall on a straight line. From k = 8, edges jump between turns, shortest paths cut across, and the embedding folds. With k of 3 or 4 the graph falls apart."

      readouts={
        <>
          <Readout label="graph" value={fit.connected ? 'connected' : 'disconnected'} />
          <Readout label="edges" value={fit.edges} />
          <Readout label="edges between turns" value={fit.shortCircuits} />
          <Readout label="correlation with arc length" value={formatNumber(corr)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points name="points" x={data.points.map((p) => p[0])} y={data.points.map((p) => p[1])} slot={0} />
          <Segments segments={fit.segments} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Points name="embedding" x={data.arc} y={fit.z} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
