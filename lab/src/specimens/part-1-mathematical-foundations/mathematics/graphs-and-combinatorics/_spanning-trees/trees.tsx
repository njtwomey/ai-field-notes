import { heapSorted } from 'aifn-compute/graph'
import { kruskalSteps, primSteps, type KruskalState, type PrimState } from 'aifn-compute/graph/spanning-trees'
import { trace } from 'aifn-compute/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from 'aifn-render/controls'
import type { ElementState } from 'aifn-render/diagram'
import { Columns, Figure } from 'aifn-render/layout'
import { GraphView } from '@lab/views'
import { Readout } from 'aifn-render/viz'
import { Sequence } from '../_shared/common'
import { lettered, placed, stateAt } from '../_shared/data'

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
  const kruskal = useMemo(() => trace(kruskalSteps(MST), undefined, 200), [])
  const prim = useMemo(() => trace(primSteps(MST), undefined, 200), [])
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
      purpose="Kruskal merges a forest by taking edges lightest first; Prim grows one tree by its lightest leaving edge. Both reach the same minimum weight."
      title="Kruskal against Prim"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="step" value={step} onChange={setStep} count={count} />
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
