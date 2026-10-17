import { useMemo } from 'react'
import { fromRows, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import {
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { hopDistances } from 'aifn-compute/graph/traversal'
import { graphormerAttention, spatialBias } from 'aifn-compute/nn/graph'

// A 6-cycle (nodes 0-5) with a tail 5-6-7: hop distances from 0 to 5.
const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 0],
  [5, 6],
  [6, 7],
]
const N = 8
const GRAPH = fromEdges(N, EDGES, { directed: false })
const DIST = toRows(hopDistances(GRAPH))
const POS: [number, number][] = [
  ...Array.from({ length: 6 }, (_, i): [number, number] => [
    Math.cos((Math.PI / 3) * (i + 1)) * 1.2,
    Math.sin((Math.PI / 3) * (i + 1)) * 1.2,
  ]),
  [2.4, 0],
  [3.6, 0],
]
// Node 5 sits at angle 2π, on the right of the ring, where the tail attaches.
// Fixed node features and projections: the content part of the scores.
const FEATURES = fromRows([
  [1.0, 0.2],
  [0.4, 0.9],
  [-0.6, 0.7],
  [-1.0, -0.1],
  [-0.3, -0.9],
  [0.6, -0.6],
  [0.9, 0.5],
  [-0.2, 1.1],
])
const W_QUERY = fromRows([
  [1.2, 0.3],
  [-0.4, 1.0],
])
const W_KEY = fromRows([
  [1.0, -0.2],
  [0.5, 1.1],
])
const W_VALUE = fromRows([
  [1, 0],
  [0, 1],
])
const ZEROS = fromRows(Array.from({ length: N }, () => [0, 0]))
const MAX_HOPS = Math.max(...DIST.flat())

/** One head of Graphormer attention on an 8-node graph, with a spatial bias that penalises distance beyond one hop. */
export function SpatialBias() {
  const state = useFigureState({
    beta: slider(0, 6, 1.5, { step: 0.1, label: 'bias slope β' }),
    content: setting(true, 'content scores qᵢᵀkⱼ/√d'),
    query: slider(0, N - 1, 7, { step: 1, onChart: true, label: 'query node i' }),
  })
  const scale = useScaleColor('sequential')

  const result = useMemo(() => {
    // b(φ) = 0 for φ ≤ 1 and −β(φ − 1) beyond: β = 0 is global attention, large β attention over the 1-hop ball.
    const table = Array.from({ length: MAX_HOPS + 1 }, (_, d) => -state.beta * Math.max(0, d - 1))
    const h = state.content ? FEATURES : ZEROS
    const { weights } = graphormerAttention(h, { query: W_QUERY, key: W_KEY, value: W_VALUE }, spatialBias(DIST, table))
    return { table, weights: toRows(weights as Tensor) }
  }, [state.beta, state.content])

  const i = Math.round(state.query)
  const row = result.weights[i]
  const near = row.reduce((s, w, j) => s + (DIST[i][j] <= 1 ? w : 0), 0)
  const far = row.reduce((s, w, j) => s + (DIST[i][j] >= 3 ? w : 0), 0)
  const entropy = -row.reduce((s, w) => s + (w > 0 ? w * Math.log(w) : 0), 0)
  const top = Math.max(...row)
  const colors = row.map((w) => scale(top > 0 ? w / top : 0))
  const nearest = (p: [number, number]) => {
    let best = 0
    for (let k = 1; k < N; k++)
      if (Math.hypot(POS[k][0] - p[0], POS[k][1] - p[1]) < Math.hypot(POS[best][0] - p[0], POS[best][1] - p[1]))
        best = k
    return best
  }

  const gx = useAxis({ range: [-1.8, 4.2], nice: false, zoom: false })
  const gy = useAxis({ range: [-1.7, 1.7], nice: false, zoom: false, equal: gx })
  const kx = useAxis({ label: 'key node j', range: [-0.5, N - 0.5], nice: false, integer: true })
  const qy = useAxis({ label: 'query node i', range: [-0.5, N - 0.5], nice: false, integer: true, inverse: true })
  const nodes = Array.from({ length: N }, (_, k) => k)

  return (
    <Figure
      title="Attention with a shortest-path bias"
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="bias b(φ) for φ = 0 … 5" value={result.table.map((b) => formatNumber(b)).join(', ')} />
          <Readout label={`weight of node ${i} on itself and its neighbours`} value={formatNumber(near)} />
          <Readout label={`weight of node ${i} on nodes 3 or more hops away`} value={formatNumber(far)} />
          <Readout label="entropy of the row (nats; uniform = 2.08)" value={formatNumber(entropy)} />
        </>
      }
      caption="A 6-cycle with a two-node tail, so shortest-path distances run from 0 to 5. Every node attends to every node; the score of the pair (i, j) is the content term qᵢᵀkⱼ/√d plus a bias b(φ) that depends only on their distance φ: zero up to one hop and −β per extra hop. Left: the graph, each node coloured and labelled by the attention weight the query node i puts on it; drag the ringed query to another node. Right: the whole 8 × 8 attention matrix, one row per query. At β = 0 the graph plays no part and the weights follow the features alone. As β grows, weight moves onto each node's 1-hop neighbourhood, and for large β the layer becomes attention restricted to neighbours, as in a graph attention network. Switch the content scores off to see the bias alone."
    >
      <Plots cols={2} widths={[3, 2]}>
        <Plot x={gx} y={gy} bare legend={false} title={`attention of node ${i}`}>
          <Segments name="edges" segments={EDGES.map(([a, b]) => ({ from: POS[a], to: POS[b] }))} width={1.5} />
          <Points name="query" x={[POS[i][0]]} y={[POS[i][1]]} emphasis size={26} />
          <Points
            name="nodes"
            x={POS.map((p) => p[0])}
            y={POS.map((p) => p[1])}
            colors={colors}
            labels={row.map((w, k) => `${k}: ${w.toFixed(2)}`)}
            size={16}
          />
          <Handle kind="point" at={POS[i]} onDrag={(p) => state.set('query', nearest(p))} label="i" />
        </Plot>
        <Plot x={kx} y={qy} title="attention matrix">
          <Raster x={nodes} y={nodes} z={result.weights} scale="sequential" range={[0, 1]} valueLabel="weight" />
        </Plot>
      </Plots>
    </Figure>
  )
}
