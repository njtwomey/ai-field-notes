import { useContext, useMemo, type ReactNode } from 'react'
import {
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
} from 'aifn-render'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import { graphConv } from 'aifn-compute/nn/graph'

type Small = { name: string; edges: [number, number][]; positions: [number, number][] }

// Three small graphs, each drawn in a unit box and placed side by side.
const GRAPHS: Small[] = [
  {
    name: 'triangle',
    edges: [
      [0, 1],
      [1, 2],
      [0, 2],
    ],
    positions: [
      [0, 0],
      [1, 0],
      [0.5, 0.87],
    ],
  },
  {
    name: 'path',
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
    positions: [
      [0, 0],
      [0.33, 0.6],
      [0.67, 0],
      [1, 0.6],
    ],
  },
  {
    name: 'star',
    edges: [
      [0, 1],
      [0, 2],
      [0, 3],
    ],
    positions: [
      [0.5, 0.4],
      [0, 0],
      [1, 0],
      [0.5, 1],
    ],
  },
]
const SPACING = 1.6

function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The first `count` graphs as one disconnected graph: offsets, the block-diagonal adjacency and the batch vector. */
function batch(count: number) {
  const parts = GRAPHS.slice(0, count)
  const offsets: number[] = []
  let total = 0
  for (const p of parts) {
    offsets.push(total)
    total += p.positions.length
  }
  const edges = parts.flatMap((p, b) => p.edges.map(([i, j]) => [i + offsets[b], j + offsets[b]] as [number, number]))
  const membership = parts.flatMap((p, b) => p.positions.map(() => b))
  const xs = parts.flatMap((p, b) => p.positions.map(([x]) => x + b * SPACING))
  const ys = parts.flatMap((p) => p.positions.map(([, y]) => y))
  // A categorical matrix: the graph's slot where A_ij = 1, unassigned (−1) elsewhere.
  const z = Array.from({ length: total }, () => new Array<number>(total).fill(-1))
  for (const [i, j] of edges) z[i][j] = z[j][i] = membership[i]
  const segments = edges.map(([i, j]) => ({ from: [xs[i], ys[i]] as const, to: [xs[j], ys[j]] as const }))
  return { parts, offsets, total, edges, membership, xs, ys, z, segments }
}

/** A graph convolution with symmetric normalisation on one scalar feature (the node's degree). */
function gcn(nodes: number, edges: [number, number][]): number[] {
  const g = fromEdges(nodes, edges, { directed: false })
  const degree = new Array<number>(nodes).fill(0)
  for (const [i, j] of edges) {
    degree[i]++
    degree[j]++
  }
  const h = tensor(degree.map((d) => [d]))
  return Array.from(toFlat(graphConv(g, h, tensor([[1]])) as Tensor))
}

/**
 * Batching graphs for a graph network: the graphs become the connected components of one graph, whose adjacency
 * matrix is block diagonal, so one layer call processes all of them and gives each graph the output it would get alone.
 */
export function BlockDiagonalBatch() {
  const state = useFigureState({
    graphs: int(3, { ge: 1, le: GRAPHS.length, label: 'graphs in the batch' }),
  })
  const b = useMemo(() => batch(state.graphs), [state.graphs])
  const together = useMemo(() => gcn(b.total, b.edges), [b])
  const separately = useMemo(() => b.parts.flatMap((p) => gcn(p.positions.length, p.edges)), [b])
  const gap = Math.max(...together.map((v, i) => Math.abs(v - separately[i])))
  const index = Array.from({ length: b.total }, (_, i) => i + 1)
  const grid = useMemo(
    () =>
      Array.from({ length: b.total + 1 }, (_, k) => k + 0.5).flatMap((v) => [
        { from: [v, 0.5] as const, to: [v, b.total + 0.5] as const },
        { from: [0.5, v] as const, to: [b.total + 0.5, v] as const },
      ]),
    [b],
  )
  const blocks = useMemo(
    () =>
      b.offsets
        .map((o, k) => {
          const lo = o + 0.5
          const hi = o + b.parts[k].positions.length + 0.5
          return [
            { from: [lo, lo] as const, to: [hi, lo] as const },
            { from: [hi, lo] as const, to: [hi, hi] as const },
            { from: [hi, hi] as const, to: [lo, hi] as const },
            { from: [lo, hi] as const, to: [lo, lo] as const },
          ]
        })
        .flat(),
    [b],
  )

  const gx = useAxis({ label: '', range: [-0.3, GRAPHS.length * SPACING - 0.3], nice: false })
  const gy = useAxis({ label: '', range: [-0.3, 1.3], nice: false, equal: gx })
  const ax = useAxis({ label: 'column j', range: [0.5, b.total + 0.5], nice: false, integer: true })
  const ay = useAxis({ label: 'row i', range: [0.5, b.total + 0.5], nice: false, integer: true, inverse: true })

  return (
    <Figure
      title="A batch of graphs is one block-diagonal graph"
      state={state}
      hoverReadout={false}
      caption={`A batch of up to ${GRAPHS.length} small graphs (a triangle, a path and a star). The batch numbers their nodes consecutively and keeps every edge inside its own graph, so the batch is one graph with one connected component per input graph. Its adjacency matrix (right) is block diagonal: each coloured block is one graph's adjacency matrix and every entry between graphs is zero. A batch vector records which graph each node belongs to, for the readout. A graph convolution applied once to the batch gives every node the value it gets when its graph is processed alone; the readout reports the largest difference.`}
      readouts={
        <>
          <Readout label="nodes / edges in the batch" value={`${b.total} / ${b.edges.length}`} />
          <Readout label="batch vector" value={b.membership.join(' ')} />
          <Readout
            label="non-zero entries of the batch adjacency"
            value={`${2 * b.edges.length} of ${b.total * b.total}`}
          />
          <Readout label="max |batched − separate| GCN output" value={formatNumber(gap)} />
        </>
      }
    >
      <FixedHeight height={260}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={gx} y={gy} title="the graphs, numbered as one batch" bare>
            <Segments segments={b.segments} muted width={2} />
            <Points
              x={b.xs}
              y={b.ys}
              group={b.membership}
              groupNames={b.parts.map((p) => p.name)}
              labels={index.map(String)}
              size={14}
            />
          </Plot>
          <Plot x={ax} y={ay} title="batch adjacency matrix">
            <Raster
              x={index}
              y={index}
              z={b.z}
              scale="categorical"
              categoryNames={b.parts.map((p) => p.name)}
              valueLabel="edge of"
            />
            <Segments segments={grid} muted width={1} />
            <Segments segments={blocks} emphasis width={2} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
