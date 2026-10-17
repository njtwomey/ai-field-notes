import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { graphAttention } from 'aifn-compute/nn/graph'
import type { Graph } from 'aifn-compute/graph'

const NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
// Node features z_j (the map W is the identity, so z = h), and a drawing of the graph.
const FEATURES: [number, number][] = [
  [1, 0],
  [0, 1],
  [-1, 0.5],
  [0.5, -1],
  [1.5, 1],
  [-0.5, -0.5],
]
const LAYOUT: [number, number][] = [
  [0, 2],
  [1.6, 1],
  [-1.6, 1],
  [1.6, -1],
  [0, 0.4],
  [0, -1],
]
const EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [0, 4],
  [1, 4],
  [1, 3],
  [2, 5],
  [3, 5],
  [4, 5],
]
const GRAPH: Graph = { kind: 'graph', nodes: 6, edges: EDGES.map(([from, to]) => ({ from, to })), directed: false }
const H = tensor(FEATURES)
const IDENTITY = tensor([
  [1, 0],
  [0, 1],
])
type Variant = 'gat' | 'v2'

/** GAT against GATv2 on one small graph: attention into a chosen node, and whether rankings depend on the query. */
export function AttentionOnGraph() {
  const state = useFigureState({
    variant: choice<Variant>(
      [
        { value: 'gat', label: 'GAT: LeakyReLU(aᵀ(z_i + z_j))' },
        { value: 'v2', label: 'GATv2: aᵀ LeakyReLU(z_i + z_j)' },
      ],
      'gat',
      { label: 'score' },
    ),
    query: choice(
      NAMES.map((n, i) => ({ value: i, label: n })),
      3,
      { label: 'query node i' },
    ),
    ax: slider(-2, 2, 1, { step: 0.05, onChart: true, label: 'a₁' }),
    ay: slider(-2, 2, -1, { step: 0.05, onChart: true, label: 'a₂' }),
  })

  const result = useMemo(() => {
    const a = tensor([[state.ax, state.ay]])
    const params =
      state.variant === 'gat' ? { weight: IDENTITY, attSource: a, attTarget: a } : { weight: IDENTITY, att: a }
    const r = graphAttention(GRAPH, H, params, { variant: state.variant })
    const alpha = toFlat(r.attention as Tensor)
    // weights[i] maps each node j in the closed neighbourhood of i to alpha_ij.
    const weights = NAMES.map(() => new Map<number, number>())
    r.source.forEach((j, k) => weights[r.destination[k]].set(j, alpha[k]))
    // Pairs of nodes that two queries rank in opposite orders: always 0 for static attention.
    let flips = 0
    for (let j1 = 0; j1 < NAMES.length; j1++)
      for (let j2 = j1 + 1; j2 < NAMES.length; j2++) {
        let above = false
        let below = false
        for (const m of weights)
          if (m.has(j1) && m.has(j2)) {
            const d = m.get(j1)! - m.get(j2)!
            if (d > 1e-9) above = true
            if (d < -1e-9) below = true
          }
        if (above && below) flips++
      }
    const rankings = weights.map(
      (m, i) =>
        `${NAMES[i]}: ${[...m.entries()]
          .sort((p, q) => q[1] - p[1])
          .map(([j]) => NAMES[j])
          .join(' > ')}`,
    )
    return { weights, flips, rankings }
  }, [state.ax, state.ay, state.variant])

  const i = state.query
  const into = result.weights[i]
  const arrows: Vector[] = [...into.entries()]
    .filter(([j]) => j !== i)
    .map(([j, alpha]): Vector => {
      const [x0, y0] = LAYOUT[j]
      const [x1, y1] = LAYOUT[i]
      const cut = 0.2 / Math.hypot(x1 - x0, y1 - y0)
      return {
        from: [x0 + (x1 - x0) * cut, y0 + (y1 - y0) * cut],
        to: [x1 - (x1 - x0) * cut, y1 - (y1 - y0) * cut],
        label: formatNumber(alpha),
        labelAt: 'middle',
        width: 1 + 6 * alpha,
        slot: 1,
      }
    })
  const others = NAMES.map((_, j) => j).filter((j) => j !== i)

  const gx = useAxis({ label: '', range: [-2.4, 2.4], nice: false })
  const gy = useAxis({ label: '', range: [-1.6, 2.6], nice: false, equal: gx })
  const fx = useAxis({ label: 'feature 1', range: [-2.2, 2.2] })
  const fy = useAxis({ label: 'feature 2', range: [-2.2, 2.2], equal: fx })

  return (
    <Figure
      title="Graph attention: static against dynamic"
      state={state}
      defaultSize="L"
      caption={`Left: a graph of six nodes; the arrows show the attention weights α_ij into the query node ${NAMES[i]} from its neighbours (its weight on itself is a readout), thicker for larger weights. Right: the node features z_j, with W the identity, and the attention vector a, whose tip can be dragged. With GAT the score is LeakyReLU(aᵀz_i + aᵀz_j), so every query ranks its neighbours by aᵀz_j, the projection onto a: rankings never contradict each other and the count of key pairs ranked oppositely stays 0 wherever a is dragged. With GATv2 the nonlinearity sits inside, aᵀ LeakyReLU(z_i + z_j), and the ranking of the same two keys can differ between queries (open the rankings readout).`}
      readouts={
        <>
          <Readout label={`α of ${NAMES[i]} on itself`} value={formatNumber(into.get(i) ?? 0)} />
          <Readout label="key pairs ranked oppositely by two queries" value={String(result.flips)} />
          <Readout label="ranking of each closed neighbourhood" value={result.rankings.join('; ')} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={gx} y={gy} title={`attention into ${NAMES[i]}`}>
          <Segments name="edges" segments={EDGES.map(([a, b]) => ({ from: LAYOUT[a], to: LAYOUT[b] }))} />
          <Points
            name="nodes"
            x={others.map((j) => LAYOUT[j][0])}
            y={others.map((j) => LAYOUT[j][1])}
            labels={others.map((j) => NAMES[j])}
            muted
            size={16}
          />
          <Points name="query" x={[LAYOUT[i][0]]} y={[LAYOUT[i][1]]} labels={[NAMES[i]]} emphasis size={18} />
          {arrows.length > 0 && <Vectors name="attention α_ij" vectors={arrows} />}
        </Plot>
        <Plot x={fx} y={fy} title="features and attention vector">
          <Points
            name="features z_j"
            x={FEATURES.map((p) => p[0])}
            y={FEATURES.map((p) => p[1])}
            labels={NAMES}
            slot={0}
          />
          <Points name="query feature" x={[FEATURES[i][0]]} y={[FEATURES[i][1]]} emphasis size={12} />
          <Vectors vectors={[{ from: [0, 0], to: [state.ax, state.ay], label: 'a' }]} />
          <Handle {...state.handle(['ax', 'ay'], { label: 'a' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
