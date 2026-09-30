import {
  edmondsKarpSteps,
  heapSorted,
  kruskalSteps,
  primSteps,
  type EdmondsKarpState,
  type KruskalState,
  type PrimState,
} from 'aifn/graph'
import { trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import type { ElementState } from '@lab/diagram'
import { Figure } from '@lab/layout'
import { formatValue, GraphView } from '@lab/views'
import { Readout } from '@lab/viz'
import { Columns, Framed, Sequence } from './common'
import { lettered, placed, stateAt } from './data'

// ── Kruskal against Prim ────────────────────────────────────────────────────────────────────────────────────────────

/** CLRS Figure 23.1: a connected weighted graph whose minimum spanning tree weighs 37. */
const MST = lettered(
  'abcdefghi',
  [
    ['ab', 4],
    ['ah', 8],
    ['bc', 8],
    ['bh', 11],
    ['cd', 7],
    ['cf', 4],
    ['ci', 2],
    ['de', 9],
    ['df', 14],
    ['ef', 10],
    ['fg', 2],
    ['gh', 1],
    ['gi', 6],
    ['hi', 7],
  ],
  false,
)
const MST_AT = placed('abcdefghi', {
  a: [0, 1.5],
  b: [2, 0],
  c: [4, 0],
  d: [6, 0],
  e: [8, 1.5],
  f: [6, 3],
  g: [4, 3],
  h: [2, 3],
  i: [3, 1.5],
})

const edgeName = (k: number) => {
  const e = MST.edges[k]
  return `${MST.labels![e.from]}${MST.labels![e.to]}`
}

function describeKruskal(s: KruskalState): string {
  if (s.edge < 0) return s.done ? 'done' : 'start'
  const w = MST.edges[s.edge].weight
  return `${edgeName(s.edge)} (${w}): ${s.event === 'accept' ? 'joins two trees' : 'closes a cycle'}`
}

function describePrim(s: PrimState): string {
  if (s.event === 'root') return `start at ${MST.labels![s.current]}`
  if (s.edge < 0) return s.event
  const w = MST.edges[s.edge].weight
  return `${edgeName(s.edge)} (${w}): ${s.event === 'accept' ? `adds ${MST.labels![s.current]}` : 'stale, both ends in the tree'}`
}

export function SpanningTreeSpecimen() {
  const kruskal = useMemo(() => trace(kruskalSteps, MST, 200), [])
  const prim = useMemo(() => trace(primSteps, MST, 200), [])
  const count = Math.max(kruskal.steps.length, prim.steps.length)
  const [step, setStep] = useState(0)
  const k = stateAt(kruskal, step)
  const p = stateAt(prim, step)

  const kTree = new Set(Array.from(k.tree.data))
  const kSeen = new Set(Array.from(k.sorted.data).slice(0, k.position))
  const kTouched = new Set(Array.from(k.tree.data).flatMap((e) => [MST.edges[e].from, MST.edges[e].to]))
  const kEdge = (e: number): ElementState => (e === k.edge ? 'active' : kTree.has(e) ? 'done' : 'idle')

  const pTree = new Set(Array.from(p.tree.data))
  const pQueued = new Set(
    heapSorted(p.queue)
      .filter((c) => !p.inTree.data[c.value.to])
      .map((c) => c.value.edge),
  )
  const pEdge = (e: number): ElementState =>
    e === p.edge ? 'active' : pTree.has(e) ? 'done' : pQueued.has(e) ? 'done' : 'idle'

  return (
    <Figure
      title="Kruskal against Prim"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="step" value={step} onChange={setStep} count={count} defaultSpeed={2} />
        </div>
      }
      readouts={
        <>
          <Readout label="Kruskal" value={`${describeKruskal(k)}; weight ${k.weight}`} />
          <Readout label="Prim" value={`${describePrim(p)}; weight ${p.weight}`} />
        </>
      }
      caption="CLRS Figure 23.1. Kruskal takes edges lightest first and keeps each that joins two trees of the forest (union–find decides); rejected edges are dashed. Prim grows one tree from a: it pops the lightest queued edge leaving the tree (queued edges in slot 2) and skips edges gone stale. Tree edges are in slot 0. Both end at weight 37."
    >
      <Columns
        panels={[
          {
            title: 'Kruskal: edges by weight, union–find',
            body: (
              <GraphView
                graph={MST}
                layout={MST_AT}
                height="fill"
                showWeights
                ariaLabel="Kruskal's algorithm"
                nodeState={(v) => (kTouched.has(v) ? 'done' : 'idle')}
                edgeState={kEdge}
                edgeTone={(e) => (kTree.has(e) ? 0 : undefined)}
                edgeDashed={(e) => kSeen.has(e) && !kTree.has(e)}
              />
            ),
            footer: (
              <Sequence
                label="next"
                ends="lightest first"
                items={Array.from(k.sorted.data)
                  .slice(k.position, k.position + 6)
                  .map((e) => `${edgeName(e)} ${MST.edges[e].weight}`)}
              />
            ),
          },
          {
            title: 'Prim: one tree, a heap of leaving edges',
            body: (
              <GraphView
                graph={MST}
                layout={MST_AT}
                height="fill"
                showWeights
                ariaLabel="Prim's algorithm"
                nodeState={(v) => (p.inTree.data[v] ? 'done' : 'idle')}
                nodeHighlight={(v) => v === p.current}
                edgeState={pEdge}
                edgeTone={(e) => (pTree.has(e) ? 0 : pQueued.has(e) ? 2 : undefined)}
                edgeDashed={(e) => e === p.edge && p.event === 'reject'}
              />
            ),
            footer: (
              <Sequence
                label="heap"
                ends="lightest first"
                items={heapSorted(p.queue)
                  .slice(0, 6)
                  .map((c) => `${edgeName(c.value.edge)} ${c.priority}`)}
              />
            ),
          },
        ]}
      />
    </Figure>
  )
}

// ── Edmonds–Karp ────────────────────────────────────────────────────────────────────────────────────────────────────

/** CLRS Figure 26.1: s, v₁ … v₄, t with capacities; the maximum flow is 23. */
const NET = lettered(
  'sabcdt',
  [
    ['sa', 16],
    ['sb', 13],
    ['ba', 4],
    ['ac', 12],
    ['cb', 9],
    ['bd', 14],
    ['dc', 7],
    ['ct', 20],
    ['dt', 4],
  ],
  true,
)
const NET_LABELS = ['s', 'v_1', 'v_2', 'v_3', 'v_4', 't'].map((x) => `$${x}$`)
const NET_AT = placed('sabcdt', { s: [0, 1.5], a: [2.5, 0], b: [2.5, 3], c: [5, 0], d: [5, 3], t: [7.5, 1.5] })

function describeFlow(s: EdmondsKarpState): string {
  if (s.iteration === 0 && !s.done) return 'no flow yet'
  if (s.done) return 'no augmenting path: the flow is maximum'
  const path = Array.from(s.pathNodes.data, (v) => NET_LABELS[v].replace(/\$/g, '')).join(' → ')
  return `augment ${formatValue(s.bottleneck)} along ${path}`
}

export function MaxFlowSpecimen() {
  const t = useMemo(() => trace(edmondsKarpSteps, { graph: NET, source: 0, sink: 5 }, 100), [])
  const [step, setStep] = useState(0)
  const s = stateAt(t, step)
  const onPath = new Map(Array.from(s.path.data, (m) => [m >= 0 ? m : -m - 1, m >= 0]))
  const side = s.reached.data
  const cut = (k: number) => s.done && side[NET.edges[k].from] === 1 && side[NET.edges[k].to] === 0
  let cutCapacity = 0
  NET.edges.forEach((e, k) => cut(k) && (cutCapacity += e.weight!))

  return (
    <Figure
      title="Edmonds–Karp: augmenting paths and the minimum cut"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="augmentation" value={step} onChange={setStep} count={t.steps.length} defaultSpeed={1} />
        </div>
      }
      readouts={
        <>
          <Readout label="step" value={describeFlow(s)} />
          <Readout label="flow value" value={formatValue(s.value)} />
          {s.done && <Readout label="cut capacity" value={formatValue(cutCapacity)} />}
        </>
      }
      caption="CLRS Figure 26.1; edges are labelled flow/capacity. Each step finds a shortest augmenting path by breadth-first search in the residual network (active edges; a path may cancel flow on an edge by using it backwards) and pushes its bottleneck. When the sink is out of reach, the nodes the last search reached (slot 0) and the rest (slot 1) form a minimum cut: the highlighted saturated edges, whose capacities sum to the flow value."
    >
      <Framed
        footer={
          <Sequence label="path" ends="source first" items={Array.from(s.pathNodes.data, (v) => NET.labels![v])} />
        }
      >
        <GraphView
          graph={NET}
          layout={NET_AT}
          height="fill"
          ariaLabel="Flow network (CLRS Figure 26.1)"
          nodeLabels={NET_LABELS}
          nodeState={(v) => (s.done ? 'done' : side[v] ? 'active' : 'idle')}
          nodeTone={(v) => (s.done ? (side[v] ? 0 : 1) : undefined)}
          edgeLabels={(k) => `${formatValue(s.flow.data[k])}/${NET.edges[k].weight}`}
          edgeState={(k) => (onPath.has(k) ? 'active' : s.flow.data[k] > 0 ? 'done' : 'idle')}
          edgeHighlight={cut}
          edgeDashed={(k) => onPath.get(k) === false}
        />
      </Framed>
    </Figure>
  )
}
