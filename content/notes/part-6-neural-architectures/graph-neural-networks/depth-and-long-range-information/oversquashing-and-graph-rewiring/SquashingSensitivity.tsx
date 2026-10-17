import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
  useScaleColor,
  when,
} from 'aifn-render'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { randomWalkMatrix } from 'aifn-compute/graph/propagation'
import { cheegerConstant } from 'aifn-compute/graph/measures'
import { fosrRewire, withVirtualNode } from 'aifn-compute/graph/rewiring'
import { barbellGraph } from 'aifn-compute/graph/structures'
import { unweightedShortestPaths } from 'aifn-compute/graph/traversal'
import { propagationSensitivity } from 'aifn-compute/nn/graph'
import { spectralGap } from 'aifn-compute/probability/markov'

const CLIQUE = 5
const PATH = 3
const BARBELL = barbellGraph(CLIQUE, PATH)
const N = BARBELL.nodes
const EDGES = BARBELL.edges.map((e): [number, number] => [e.from, e.to])
const MAX_DEPTH = 16
const VIRTUAL_POS: [number, number] = [0, 1.9]
/** log10 sensitivities below this are drawn at the pale end of the scale. */
const FLOOR = -6
type Repair = 'none' | 'virtual' | 'fosr'

/**
 * A left-to-right drawing of the barbell: each clique on a unit circle with its attachment node facing the path,
 * the path nodes one unit apart between them.
 */
function barbellLayout(c: number, p: number): [number, number][] {
  const half = (p + 1) / 2
  const pos: [number, number][] = new Array(2 * c + p)
  for (let i = 0; i < c; i++) {
    const t = (2 * Math.PI * (i - (c - 1))) / c
    pos[i] = [-half - 1 + Math.cos(t), Math.sin(t)]
    const s = Math.PI + (2 * Math.PI * i) / c
    pos[c + p + i] = [half + 1 + Math.cos(s), Math.sin(s)]
  }
  for (let k = 0; k < p; k++) pos[c + k] = [-half + (k + 1), 0]
  return pos
}
const POS = barbellLayout(CLIQUE, PATH)

/** The spectral gap 1 − λ₂ of the random walk on a graph. */
const gapOf = (g: Graph) => spectralGap(randomWalkMatrix(g)).gap

/** Sensitivity of one node to every other through the powers of Â, on a barbell, with and without repairs. */
export function SquashingSensitivity() {
  const state = useFigureState({
    repair: choice<Repair>(
      [
        { value: 'none', label: 'original graph' },
        { value: 'virtual', label: 'add a virtual node' },
        { value: 'fosr', label: 'add edges by FoSR' },
      ],
      'none',
      { label: 'repair' },
    ),
    added: int(2, { min: 1, max: 8, label: 'FoSR edges added', when: when('repair', 'fosr') }),
    depth: slider(1, MAX_DEPTH, 6, { step: 1, label: 'layers r' }),
    target: slider(0, N - 1, N - 1, { step: 1, onChart: true, label: 'target node v' }),
  })
  const scale = useScaleColor('sequential')

  const versions = useMemo(() => {
    const fosr = fosrRewire(BARBELL, state.added)
    const graphs: Record<Repair, Graph> = {
      none: BARBELL,
      virtual: withVirtualNode(BARBELL),
      fosr: fosr.graph,
    }
    const gaps = {
      none: gapOf(graphs.none),
      virtual: gapOf(graphs.virtual),
      fosr: gapOf(graphs.fosr),
    }
    const cheeger = {
      none: cheegerConstant(graphs.none).value,
      virtual: cheegerConstant(graphs.virtual).value,
      fosr: cheegerConstant(graphs.fosr).value,
    }
    return { graphs, added: fosr.added, gaps, cheeger }
  }, [state.added])

  const v = Math.round(state.target)
  const far = useMemo(() => {
    const d = toFlat(unweightedShortestPaths(BARBELL, v).distance)
    const max = Math.max(...d)
    return { node: d.indexOf(max), distance: max }
  }, [v])

  const curves = useMemo(() => {
    const out = {} as Record<Repair, number[]>
    for (const r of ['none', 'virtual', 'fosr'] as Repair[]) {
      const s = toRows(propagationSensitivity(versions.graphs[r], v, MAX_DEPTH))
      out[r] = s.map((row) => (row[far.node] > 0 ? row[far.node] : NaN))
    }
    return out
  }, [versions, v, far.node])

  const shown = useMemo(
    () => toRows(propagationSensitivity(versions.graphs[state.repair], v, state.depth))[state.depth],
    [versions, state.repair, v, state.depth],
  )

  const pos = POS
  const logs = Array.from({ length: N }, (_, u) => (shown[u] > 0 ? Math.log10(shown[u]) : -Infinity))
  const colors = logs.map((l) => scale(Number.isFinite(l) ? Math.min(1, Math.max(0, (l - FLOOR) / -FLOOR)) : 0))
  const labels = logs.map((l) => (Number.isFinite(l) ? l.toFixed(1) : '0'))
  const seg = (es: readonly (readonly [number, number])[], p: (i: number) => [number, number]) =>
    es.map(([a, b]) => ({ from: p(a), to: p(b) }))
  const at = (i: number): [number, number] => (i === N ? VIRTUAL_POS : pos[i])
  const nearest = (p: [number, number]) => {
    let best = 0
    for (let i = 1; i < N; i++)
      if (Math.hypot(pos[i][0] - p[0], pos[i][1] - p[1]) < Math.hypot(pos[best][0] - p[0], pos[best][1] - p[1]))
        best = i
    return best
  }

  const gx = useAxis({ range: [-4.6, 4.6], nice: false, zoom: false })
  const gy = useAxis({ range: [-1.5, 2.4], nice: false, zoom: false, equal: gx })
  const rx = useAxis({ label: 'layers r', range: [0, MAX_DEPTH], integer: true })
  const ry = useAxis({ label: '(Âʳ)ᵥᵤ', log: true })
  const depths = Array.from({ length: MAX_DEPTH + 1 }, (_, r) => r)
  const repairName: Record<Repair, string> = {
    none: 'original',
    virtual: 'virtual node',
    fosr: `FoSR, ${state.added} edges`,
  }

  return (
    <Figure
      title="Oversquashing on a barbell graph"
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="farthest node u from v (original graph)" value={`${far.node} at distance ${far.distance}`} />
          <Readout label={`(Âʳ)ᵥᵤ at r = ${state.depth}`} value={formatNumber(shown[far.node])} />
          <Readout label="spectral gap λ₁" value={formatNumber(versions.gaps[state.repair])} />
          <Readout label="Cheeger constant h" value={formatNumber(versions.cheeger[state.repair])} />
          {state.repair === 'fosr' && (
            <Readout label="added edges" value={versions.added.map(([a, b]) => `${a}–${b}`).join(', ')} />
          )}
        </>
      }
      caption="Two 5-node cliques joined by a path of three nodes. Each node u is labelled and coloured by log₁₀ of (Âʳ)ᵥᵤ, the entry of the r-th power of the GCN matrix that bounds how much the input at u can change the output of an r-layer network at the target node v (pale: insensitive, '0': out of reach). Drag the ringed target to another node. Right: the entry for the node farthest from v against depth, for the original graph and both repairs. On the original graph the far clique enters only at r equal to the distance, with a value near 2·10⁻⁴, and grows slowly because every message crosses the single path. A virtual node joined to all 13 nodes brings every pair within two hops; FoSR adds edges between the two sides of the weakest cut, chosen from the second eigenvector of the normalised adjacency, and raises the spectral gap."
    >
      <Plots cols={2} widths={[3, 2]}>
        <Plot x={gx} y={gy} bare legend={false} title={repairName[state.repair]}>
          <Segments name="edges" segments={seg(EDGES, at)} width={1.5} />
          {state.repair === 'virtual' && (
            <Segments
              name="virtual edges"
              segments={seg(
                Array.from({ length: N }, (_, i) => [i, N] as const),
                at,
              )}
              dashed
              muted
            />
          )}
          {state.repair === 'fosr' && versions.added.length > 0 && (
            <Segments name="added edges" segments={seg(versions.added, at)} width={2} slot={1} dashed />
          )}
          {state.repair === 'virtual' && (
            <Points
              name="virtual node"
              x={[VIRTUAL_POS[0]]}
              y={[VIRTUAL_POS[1]]}
              muted
              size={14}
              labels={['virtual']}
            />
          )}
          <Points name="target" x={[pos[v][0]]} y={[pos[v][1]]} emphasis size={24} />
          <Points
            name="nodes"
            x={pos.map((p) => p[0])}
            y={pos.map((p) => p[1])}
            colors={colors}
            labels={labels}
            size={15}
          />
          <Handle kind="point" at={pos[v]} onDrag={(p) => state.set('target', nearest(p))} label="v" />
        </Plot>
        <Plot x={rx} y={ry} title={`sensitivity of node ${v} to node ${far.node}`}>
          <Curve name="original" x={depths} y={curves.none} slot={0} showPoints />
          <Curve name="virtual node" x={depths} y={curves.virtual} slot={1} showPoints />
          <Curve name={`FoSR, ${state.added} edges`} x={depths} y={curves.fosr} slot={2} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
