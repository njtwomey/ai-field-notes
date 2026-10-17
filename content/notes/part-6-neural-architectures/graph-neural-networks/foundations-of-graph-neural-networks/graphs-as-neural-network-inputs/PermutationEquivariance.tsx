import { useContext, useMemo, type ReactNode } from 'react'
import {
  Bars,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { child, normals, permutation, stream } from 'aifn-compute/foundation/random'
import { matmul, tanh, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'
import { graphConv } from 'aifn-compute/nn/graph'

// The worked example's graph: a triangle 1–2–3, the edge 3–4, and two leaves 5 and 6 on node 4 (0-based here).
const N = 6
const EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 2],
  [2, 3],
  [3, 4],
  [3, 5],
]
const FEATURES = [
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 0],
  [0, 2],
  [1, 0],
]
// Drawing positions belong to the nodes themselves, so a node stays where it is when its label changes.
const POSITIONS: [number, number][] = [
  [0, 1],
  [0, -1],
  [1.2, 0],
  [2.6, 0],
  [3.6, 1],
  [3.6, -1],
]
const SEGMENTS = EDGES.map(([i, j]) => ({ from: POSITIONS[i], to: POSITIONS[j] }))
const INDEX = Array.from({ length: N }, (_, i) => i + 1)
// Cell edges for the adjacency matrix (a Raster paints opaquely).
const GRID = Array.from({ length: N + 1 }, (_, k) => k + 0.5).flatMap((v) => [
  { from: [v, 0.5] as const, to: [v, N + 0.5] as const },
  { from: [0.5, v] as const, to: [N + 0.5, v] as const },
])

// One sum-aggregation graph convolution, (A + I) X W, with a fixed W that subtracts the second feature from the first.
const W_GCN = tensor([[1], [-1]])
// A fixed random two-layer perceptron on the 36 adjacency entries and the 12 feature entries, flattened in label order.
const MLP_STREAM = stream('gnn-foundations/permutation/mlp')
const W1 = normals(child(MLP_STREAM, 'w1'), [N * N + 2 * N, 16], 0, 1 / Math.sqrt(N * N + 2 * N))
const W2 = normals(child(MLP_STREAM, 'w2'), [16, N], 0, 0.5)

/** The graph, features and outputs under the labelling `order` (order[new label] = node). */
function labelled(order: number[]) {
  const labelOf = new Array<number>(N)
  order.forEach((node, label) => (labelOf[node] = label))
  const g = fromEdges(
    N,
    EDGES.map(([a, b]) => [labelOf[a], labelOf[b]] as [number, number]),
    { directed: false },
  )
  const x = order.map((node) => FEATURES[node])
  const a = toRows(adjacencyMatrix(g)) as number[][]
  const gcn = Array.from(toFlat(graphConv(g, tensor(x), W_GCN, undefined, { normalisation: 'none' }) as Tensor))
  const input = tensor([[...a.flat(), ...x.flat()]])
  const mlp = Array.from(toFlat(matmul(tanh(matmul(input, W1)), W2)))
  return { labelOf, a, gcn, mlp }
}

const IDENTITY = Array.from({ length: N }, (_, i) => i)
const ORIGINAL = labelled(IDENTITY)

function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * The same graph under different node labellings: the adjacency matrix and the feature rows are permuted, a graph
 * convolution's outputs are permuted with them, and a perceptron on the flattened matrix gives unrelated outputs.
 */
export function PermutationEquivariance() {
  const state = useFigureState({
    relabelling: int(0, { ge: 0, le: 999, label: 'relabelling (0 = original labels)' }),
  })
  const { relabelling } = state
  const order = useMemo(
    () =>
      relabelling === 0
        ? IDENTITY
        : Array.from(toFlat(permutation(stream(`gnn-foundations/permutation/${relabelling}`), N))),
    [relabelling],
  )
  const now = useMemo(() => labelled(order), [order])
  // What an equivariant layer must output: the original outputs, moved to each node's new label.
  const expectedGcn = order.map((node) => ORIGINAL.gcn[node])
  const expectedMlp = order.map((node) => ORIGINAL.mlp[node])
  const gap = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
  const colour = useScaleColor('diverging')
  const bound = Math.max(...ORIGINAL.gcn.map(Math.abs))
  // Node colour: its GCN output, read at its current label.
  const colours = POSITIONS.map((_, node) => colour((now.gcn[now.labelOf[node]] / bound + 1) / 2))
  const labels = POSITIONS.map((_, node) => String(now.labelOf[node] + 1))

  const gx = useAxis({ label: '', range: [-0.6, 4.2], nice: false })
  const gy = useAxis({ label: '', range: [-1.5, 1.5], nice: false, equal: gx })
  const ax = useAxis({ label: 'column j', range: [0.5, N + 0.5], nice: false, integer: true })
  const ay = useAxis({ label: 'row i', range: [0.5, N + 0.5], nice: false, integer: true, inverse: true })
  const bx = useAxis({ label: 'node label', range: [0.5, N + 0.5], nice: false, integer: true })
  const gcnY = useAxis({ label: 'GCN output', range: [-1.5, 3.5] })
  const mlpY = useAxis({ label: 'MLP output', hold: 'union' })

  return (
    <Figure
      title="Relabelling the nodes of one graph"
      state={state}
      hoverReadout={false}
      caption={`One graph on ${N} nodes with 2-dimensional features, under a labelling chosen by the relabelling number (0 keeps the labels of the worked example). Top left: the graph, each node at a fixed place, numbered by its current label and coloured by the output of one graph convolution (A + I)XW. Top right: the adjacency matrix in the current labelling; it changes with every relabelling. Bottom: the outputs at each label (bars) and the original outputs moved to the nodes' new labels (dots). The graph convolution's bars always sit on the dots: it is permutation equivariant. A perceptron on the flattened adjacency matrix and features has fixed weights per position, so its outputs change when the same graph is listed in another order.`}
      readouts={
        <>
          <Readout label="GCN: max |F(PX, PAPᵀ) − P F(X, A)|" value={formatNumber(gap(now.gcn, expectedGcn))} />
          <Readout label="MLP: max |F(PX, PAPᵀ) − P F(X, A)|" value={formatNumber(gap(now.mlp, expectedMlp))} />
          <Readout label="node order (labels 1–6 hold nodes)" value={order.map((v) => v + 1).join(' ')} />
        </>
      }
    >
      <FixedHeight height={260}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={gx} y={gy} title="graph (colour: GCN output)" bare>
            <Segments segments={SEGMENTS} muted width={2} />
            <Points
              name="node"
              x={POSITIONS.map((p) => p[0])}
              y={POSITIONS.map((p) => p[1])}
              colors={colours}
              labels={labels}
              size={18}
            />
          </Plot>
          <Plot x={ax} y={ay} title="adjacency matrix A">
            <Raster x={INDEX} y={INDEX} z={now.a} scale="sequential" range={[0, 1]} valueLabel="Aᵢⱼ" colorBar={false} />
            <Segments segments={GRID} muted width={1} />
          </Plot>
          <Plot x={bx} y={gcnY} title="graph convolution">
            <Bars name="output at label" x={INDEX} y={now.gcn} slot={0} />
            <Points name="original output, relabelled" x={INDEX} y={expectedGcn} emphasis size={8} />
          </Plot>
          <Plot x={bx} y={mlpY} title="perceptron on flattened A and X">
            <Bars name="output at label" x={INDEX} y={now.mlp} slot={1} />
            <Points name="original output, relabelled" x={INDEX} y={expectedMlp} emphasis size={8} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
