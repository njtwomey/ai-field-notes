import { useMemo, useState } from 'react'
import { choice, Figure, Player, Plot, Plots, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  folkloreWeisfeilerLeman2,
  subgraphWeisfeilerLeman,
  weisfeilerLeman,
  type Subgraph,
  type SubgraphPolicy,
} from 'aifn-compute/graph/isomorphism'
import { drawingRange, graphLayers } from '../_shared/graph-layers'
import { pairById, PAIRS, type Drawn } from '../_shared/graph-pairs'

const POLICIES: { value: SubgraphPolicy; label: string }[] = [
  { value: 'node-deleted', label: 'node-deleted' },
  { value: 'edge-deleted', label: 'edge-deleted' },
  { value: 'ego', label: 'ego network, depth 1, root marked' },
]

/** Indices of the drawing's edges that the subgraph no longer has. */
function removed(d: Drawn, s: Subgraph): Set<number> {
  const kept = new Set(s.graph.edges.map((e) => `${Math.min(e.from, e.to)}-${Math.max(e.from, e.to)}`))
  const out = new Set<number>()
  d.edges.forEach(([a, b], k) => {
    if (!kept.has(`${Math.min(a, b)}-${Math.max(a, b)}`)) out.add(k)
  })
  return out
}

/** Bags of subgraphs of two graphs, each refined by 1-WL, against 1-WL and 2-FWL on the whole graphs. */
export function SubgraphBags() {
  const state = useFigureState({
    pair: choice(
      PAIRS.map((p) => ({ value: p.id, label: p.label })),
      'hexagon-triangles',
      { label: 'pair of graphs' },
    ),
    policy: choice(POLICIES, 'node-deleted', { label: 'subgraph policy' }),
  })
  const pair = pairById(state.pair)
  const bags = useMemo(
    () => subgraphWeisfeilerLeman([pair.left.graph, pair.right.graph], state.policy),
    [pair, state.policy],
  )
  const whole = useMemo(() => {
    const graphs = [pair.left.graph, pair.right.graph]
    return { wl: weisfeilerLeman(graphs).distinguishedAt, fwl: folkloreWeisfeilerLeman2(graphs).distinguishedAt }
  }, [pair])
  const [pick, setPick] = useState(0)
  const count = Math.min(bags.subgraphs[0].length, bags.subgraphs[1].length)
  const i = Math.min(pick, count - 1)
  const sl = bags.subgraphs[0][i]
  const sr = bags.subgraphs[1][i]
  const shown = useMemo(() => weisfeilerLeman([sl.graph, sr.graph], { initial: [sl.initial, sr.initial] }), [sl, sr])
  const last = shown.colours[shown.colours.length - 1]
  const names = Array.from({ length: shown.classes[shown.classes.length - 1] }, (_, c) => `colour ${c + 1}`)
  const range = drawingRange([pair.left, pair.right])
  const lx = useAxis({ range: range.x, nice: false, zoom: false })
  const ly = useAxis({ range: range.y, nice: false, zoom: false, equal: lx })
  const rx = useAxis({ range: range.x, nice: false, zoom: false })
  const ry = useAxis({ range: range.y, nice: false, zoom: false, equal: rx })
  const panel = (d: Drawn, s: Subgraph, k: 0 | 1) =>
    graphLayers(d, {
      classes: last[k],
      classNames: names,
      labels: Array.from(last[k], (c) => String(c + 1)),
      removedEdges: removed(d, s),
      marked: s.node >= 0 ? [s.node] : [],
      id: k === 0 ? 'left' : 'right',
    })
  const verdict = (round: number) => (round >= 0 ? 'separated' : 'tied')
  return (
    <Figure
      title="Bags of subgraphs refined by colour refinement"
      state={state}
      controls={<Player value={i} onChange={setPick} count={count} label="subgraph" format={(k) => String(k + 1)} />}
      readouts={{
        'whole graphs': (
          <>
            <Readout label="1-WL" value={verdict(whole.wl)} />
            <Readout label="2-FWL (as strong as 3-WL)" value={verdict(whole.fwl)} />
          </>
        ),
        'bags of subgraphs (DS-WL)': (
          <>
            <Readout label="subgraphs per graph" value={count} />
            <Readout
              label="this pair of subgraphs"
              value={shown.distinguishedAt >= 0 ? 'histograms differ' : 'histograms equal'}
            />
            <Readout label="the two bags" value={bags.distinguished ? 'separated' : 'tied'} />
          </>
        ),
      }}
      caption={`${pair.note} The policy turns each graph into a bag of subgraphs on the same nodes: delete one node's edges (ringed node), delete one edge (dashed), or keep the edges within one hop of a root (ringed, given its own initial colour). Step through the subgraphs with the player; nodes show the stable colour-refinement colours of the two subgraphs shown, with one shared palette. The bags are compared as multisets of stable colour histograms. Node deletion separates the first three pairs, depth-1 ego networks miss decalin and bicyclopentyl, and only edge deletion separates the two strongly regular graphs, which even 2-FWL ties.`}
    >
      <Plots cols={2}>
        <Plot x={lx} y={ly} bare title={pair.left.name} legend={false}>
          {panel(pair.left, sl, 0)}
        </Plot>
        <Plot x={rx} y={ry} bare title={pair.right.name} legend={false}>
          {panel(pair.right, sr, 1)}
        </Plot>
      </Plots>
    </Figure>
  )
}
