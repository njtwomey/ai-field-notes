import { useMemo } from 'react'
import {
  choice,
  Figure,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { balancedTreeGraph, cycleGraph, gridGraph } from 'aifn-compute/graph/structures'
import { breadthFirstSearch } from 'aifn-compute/graph/traversal'
import { adjacencySequence, sequenceBandwidth } from 'aifn-methods/neural/graph'
import { circleLayout, drawingRange, FixedHeight, usePlayerPosition } from '../_shared/figure-helpers'

const PANEL = 300

type Kind = 'grid' | 'cycle' | 'tree'
const GRAPHS: { value: Kind; label: string }[] = [
  { value: 'grid', label: '3 × 4 grid' },
  { value: 'cycle', label: '10-cycle' },
  { value: 'tree', label: 'binary tree of depth 3' },
]

/** The graph and its drawing positions. */
function graphOf(kind: Kind): { g: Graph; pos: [number, number][] } {
  if (kind === 'grid') {
    const g = gridGraph(3, 4)
    return { g, pos: Array.from({ length: 12 }, (_, v) => [v % 4, -Math.floor(v / 4)] as [number, number]) }
  }
  if (kind === 'cycle') return { g: cycleGraph(10), pos: circleLayout(10, 0, 0, 1.6) }
  const g = balancedTreeGraph(2, 3)
  const pos = Array.from({ length: g.nodes }, (_, v) => {
    const depth = Math.floor(Math.log2(v + 1))
    const k = v + 1 - 2 ** depth
    return [((k + 0.5) / 2 ** depth) * 6 - 3, -depth] as [number, number]
  })
  return { g, pos }
}

/** GraphRNN's sequence view: a graph built node by node, each new node choosing its edges to the nodes before it. */
export function SequenceGeneration() {
  const state = useFigureState({
    graph: choice(GRAPHS, 'grid', { label: 'graph' }),
    ordering: choice(
      [
        { value: 'bfs', label: 'breadth-first from a root' },
        { value: 'random', label: 'random' },
      ] as const,
      'bfs',
      { label: 'node ordering' },
    ),
    root: int(0, { ge: 0, le: 14, label: 'root node (breadth-first)' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed (random ordering)' }),
  })
  const { graph, ordering, root, seed } = state

  const model = useMemo(() => {
    const { g, pos } = graphOf(graph)
    const n = g.nodes
    const start = Math.min(root, n - 1)
    // Breadth-first from the root; any other component from its lowest node.
    const sources = [start, ...Array.from({ length: n }, (_, v) => v).filter((v) => v !== start)]
    const order = toFlat(
      ordering === 'bfs' ? breadthFirstSearch(g, sources).order : permutation(stream(`graph-sequence/${seed}`), n),
    )
    const rows = adjacencySequence(g, order)
    return { g, pos, n, order, rows, bandwidth: sequenceBandwidth(rows) }
  }, [graph, ordering, root, seed])
  const [step, setStep] = usePlayerPosition(model)
  const { g, pos, n, order, rows, bandwidth } = model
  const placed = step // nodes at positions 0 … step − 1 are placed
  const positionOf = new Map(order.map((v, i) => [v, i]))

  const shownEdges = g.edges.filter((e) => positionOf.get(e.from)! < placed && positionOf.get(e.to)! < placed)
  const newest = placed > 0 ? order[placed - 1] : -1
  const newEdges = shownEdges.filter((e) => e.from === newest || e.to === newest)
  const oldEdges = shownEdges.filter((e) => e.from !== newest && e.to !== newest)
  const seg = (es: typeof g.edges) => es.map((e) => ({ from: pos[e.from], to: pos[e.to] }))
  const placedNodes = order.slice(0, placed)
  const waiting = order.slice(placed)

  // The lower triangle of the reordered adjacency, rows filled up to the current step; the rest left empty.
  const z = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (j < i && i < placed ? rows[i][j] : Number.NaN)),
  )
  const band =
    bandwidth > 0
      ? [{ from: [-0.5, bandwidth] as [number, number], to: [n - 1 - bandwidth, n - 0.5] as [number, number] }]
      : []

  const range = drawingRange([{ pos, edges: [] }], 0.5)
  const gx = useAxis({ range: range.x, nice: false, zoom: false })
  const gy = useAxis({ range: range.y, nice: false, zoom: false, equal: gx })
  const jx = useAxis({ label: 'earlier position j', range: [-0.5, n - 0.5], nice: false, integer: true, zoom: false })
  const iy = useAxis({
    label: 'position i',
    range: [-0.5, n - 0.5],
    nice: false,
    integer: true,
    inverse: true,
    zoom: false,
  })
  const added = placed > 0 ? rows[placed - 1].flatMap((x, j) => (x ? [j] : [])) : []

  return (
    <Figure
      title="Generating a graph one node at a time"
      state={state}
      defaultSize="L"
      caption={`An autoregressive generator such as GraphRNN builds a graph by adding nodes in some order; with each new node it decides, for every node already placed, whether to join them. The decisions for node i form row i of the lower triangle of the adjacency matrix in that order (right; filled cells are edges, empty cells non-edges, unfilled rows still to come). Left: the graph so far, with the newest node's edges drawn thick and nodes still to come muted. Under a breadth-first ordering every edge reaches back at most M positions (the dashed line marks the band), so the generator only has to predict the last M entries of each row; under a random ordering edges reach anywhere and M grows towards n − 1. Step through the construction with the player; change the graph, the ordering, the root or the seed.`}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={n + 1}
          label="nodes placed"
          format={(s) => (s === 0 ? 'empty graph' : `${s} of ${n} nodes`)}
        />
      }
      readouts={
        <>
          <Readout label="newest node" value={placed > 0 ? `${newest} (position ${placed - 1})` : '·'} />
          <Readout label="its edges back to positions" value={added.length ? added.join(', ') : 'none'} />
          <Readout label="bandwidth M of this ordering" value={String(bandwidth)} />
          <Readout
            label="entries to predict: all, within the band"
            value={`${(n * (n - 1)) / 2}, ${Array.from({ length: n }, (_, i) => Math.min(i, bandwidth)).reduce((a, b) => a + b, 0)}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={gx} y={gy} bare title="graph so far" legend={false}>
            {oldEdges.length > 0 && <Segments id="old" name="edges" segments={seg(oldEdges)} width={1.5} />}
            {newEdges.length > 0 && <Segments id="new" name="new edges" segments={seg(newEdges)} width={3.5} />}
            {waiting.length > 0 && (
              <Points
                id="waiting"
                name="still to come"
                x={waiting.map((v) => pos[v][0])}
                y={waiting.map((v) => pos[v][1])}
                muted
                size={10}
              />
            )}
            {placedNodes.length > 0 && (
              <Points
                id="placed"
                name="placed"
                x={placedNodes.map((v) => pos[v][0])}
                y={placedNodes.map((v) => pos[v][1])}
                slot={0}
                labels={placedNodes.map((v) => String(positionOf.get(v)))}
                size={14}
              />
            )}
            {newest >= 0 && (
              <Points id="newest" name="newest" x={[pos[newest][0]]} y={[pos[newest][1]]} emphasis size={22} />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={jx} y={iy} title="adjacency sequence S (lower triangle)">
            <Raster
              name="S"
              x={Array.from({ length: n }, (_, j) => j)}
              y={Array.from({ length: n }, (_, i) => i)}
              z={z}
              range={[0, 1]}
              colorBar={false}
              valueLabel="edge"
            />
            {band.length > 0 && <Segments id="band" name="band edge" segments={band} dashed />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
