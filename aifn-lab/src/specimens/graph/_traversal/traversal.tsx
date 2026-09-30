import {
  breadthFirstSearch,
  breadthFirstSteps,
  depthFirstSearch,
  depthFirstSteps,
  kahnSteps,
  tarjanSteps,
  type BreadthFirstState,
  type DepthFirstState,
  type EdgeClass,
  type TarjanState,
} from 'aifn/graph/traversal'
import { fromEdges, height, type Graph } from 'aifn/graph'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import type { DiagramNode, ElementState, Tone } from '@lab/diagram'
import { Columns, Figure } from '@lab/layout'
import { GraphView, TreeView } from '@lab/views'
import { Readout } from '@lab/viz'
import { Framed, Sequence } from '../_shared/common'
import { colourState, lettered, placed, stateAt } from '../_shared/data'

// ── Breadth-first against depth-first ───────────────────────────────────────────────────────────────────────────────

/** A directed graph whose depth-first search from a meets all four edge classes. */
const SEARCH = lettered('abcdefgh', ['ab', 'ac', 'af', 'bd', 'be', 'cf', 'da', 'ed', 'fg', 'gc', 'hg', 'he'])
const SEARCH_AT = placed('abcdefgh', {
  a: [0, 1.5],
  b: [2, 0],
  c: [2, 3],
  d: [4, -0.5],
  e: [4, 1.5],
  f: [4, 3.5],
  g: [6, 2.5],
  h: [6, 0.5],
})

const CLASS_TONE: Record<EdgeClass, Tone> = { tree: 0, back: 1, forward: 2, cross: 3 }
const CLASS_LETTER: Record<EdgeClass, string> = { tree: 'T', back: 'B', forward: 'F', cross: 'C' }

const letters = (g: Graph, xs: ArrayLike<number>) => Array.from(xs, (v) => g.labels![v])

function describeBfs(s: BreadthFirstState, g: Graph): string {
  const e = s.edge >= 0 ? g.edges[s.edge] : null
  const name = (v: number) => g.labels![v]
  switch (s.event) {
    case 'start':
      return 'start'
    case 'root':
      return `new root ${name(s.order.data[s.order.shape[0] - 1])}`
    case 'expand':
      return `dequeue ${name(s.current)}`
    case 'discover':
      return `${name(e!.from)}→${name(e!.to)}: discover`
    case 'seen':
      return `${name(e!.from)}→${name(e!.to)}: already seen`
    case 'done':
      return 'done'
  }
}

function describeDfs(s: DepthFirstState, g: Graph): string {
  const e = s.edge >= 0 ? g.edges[s.edge] : null
  const name = (v: number) => g.labels![v]
  if (s.event === 'root') return `new root ${name(s.current)}`
  if (s.event === 'finish') return `finish ${name(s.current)}`
  if (e) return `${name(e.from)}→${name(e.to)}: ${s.event}`
  return s.event
}

export function SearchSpecimen() {
  const bfs = useMemo(() => trace(breadthFirstSteps(SEARCH), undefined, 500), [])
  const dfs = useMemo(() => trace(depthFirstSteps(SEARCH), undefined, 500), [])
  const count = Math.max(bfs.steps.length, dfs.steps.length)
  const [step, setStep] = useState(0)
  const b = stateAt(bfs, step)
  const d = stateAt(dfs, step)
  const bfsTree = new Set(Array.from(b.parentEdge.data).filter((k) => k >= 0))

  return (
    <Figure
      title="Breadth-first against depth-first search"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="step" value={step} onChange={setStep} count={count} defaultSpeed={3} />
        </div>
      }
      readouts={
        <>
          <Readout label="breadth-first" value={describeBfs(b, SEARCH)} />
          <Readout label="depth-first" value={describeDfs(d, SEARCH)} />
        </>
      }
      caption="One step examines one edge (or takes the next node, or finishes one). Nodes are idle until discovered, active while in the queue or on the stack, done once all their edges are examined; the current node is highlighted. Breadth-first notes the depth; depth-first notes discovery/finish times and colours each edge by its class: T tree, B back, F forward, C cross."
    >
      <Columns
        panels={[
          {
            title: 'breadth-first: a FIFO queue',
            body: (
              <GraphView
                graph={SEARCH}
                layout={SEARCH_AT}
                height="fill"
                ariaLabel="Breadth-first search"
                nodeState={(v) => colourState(b.colour.data[v])}
                nodeHighlight={(v) => v === b.current}
                nodeNotes={(v) => (b.depth.data[v] >= 0 ? { n: `$${b.depth.data[v]}$` } : undefined)}
                edgeState={(k) => (k === b.edge ? 'active' : bfsTree.has(k) ? 'done' : 'idle')}
                edgeTone={(k) => (bfsTree.has(k) ? 0 : undefined)}
              />
            ),
            footer: <Sequence label="queue" ends="front first" items={letters(SEARCH, b.queue.data)} />,
          },
          {
            title: 'depth-first: a LIFO stack',
            body: (
              <GraphView
                graph={SEARCH}
                layout={SEARCH_AT}
                height="fill"
                ariaLabel="Depth-first search"
                nodeState={(v) => colourState(d.colour.data[v])}
                nodeHighlight={(v) => v === d.current}
                nodeNotes={(v) =>
                  d.discovery.data[v] >= 0
                    ? { n: `$${d.discovery.data[v]}/${d.finish.data[v] >= 0 ? d.finish.data[v] : '\\cdot'}$` }
                    : undefined
                }
                edgeState={(k) => (k === d.edge ? 'active' : d.edgeClass[k] ? 'done' : 'idle')}
                edgeTone={(k) => (d.edgeClass[k] ? CLASS_TONE[d.edgeClass[k]!] : undefined)}
                edgeLabels={(k) => (d.edgeClass[k] ? CLASS_LETTER[d.edgeClass[k]!] : undefined)}
              />
            ),
            footer: <Sequence label="stack" ends="bottom first" items={letters(SEARCH, d.stack.data)} />,
          },
        ]}
      />
    </Figure>
  )
}

/** The two searches' trees side by side: breadth-first is shallow and wide, depth-first deep and narrow. */
export function SearchTreesSpecimen() {
  const bfs = useMemo(() => trace(breadthFirstSteps(SEARCH), undefined, 500), [])
  const dfs = useMemo(() => trace(depthFirstSteps(SEARCH), undefined, 500), [])
  // The final trees from a (their layout stays put while the player reveals nodes as they are discovered).
  const bfsTree = useMemo(() => breadthFirstSearch(SEARCH, 0).trees[0], [])
  const dfsTree = useMemo(() => depthFirstSearch(SEARCH, 0).trees[0], [])
  const count = Math.max(bfs.steps.length, dfs.steps.length)
  const [chosen, setStep] = useState<number | null>(null)
  const step = chosen ?? count - 1
  const b = stateAt(bfs, step)
  const d = stateAt(dfs, step)
  return (
    <Figure
      title="The breadth-first and depth-first trees"
      description="Each search's parent pointers form a tree over the nodes it reaches. Breadth-first reaches every node by a fewest-edge path, so its tree is as shallow as possible; depth-first follows one path as far as it goes, so its tree is deep."
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="step" value={Math.min(step, count - 1)} onChange={setStep} count={count} defaultSpeed={3} />
        </div>
      }
      readouts={
        <>
          <Readout label="breadth-first height" value={height(bfsTree)} />
          <Readout label="depth-first height" value={height(dfsTree)} />
        </>
      }
      caption="Searches from a, stepped as in the figure above: nodes appear when discovered, active while in the queue or on the stack, done when finished; the edge being examined is highlighted when it is a tree edge. Breadth-first nodes note their depth, depth-first nodes discovery/finish times. Node h is not reachable from a, so neither tree holds it."
    >
      <Columns
        panels={[
          {
            title: 'breadth-first tree',
            body: (
              <TreeView
                tree={bfsTree}
                height="fill"
                ariaLabel="Breadth-first tree"
                hidden={(v) => b.colour.data[bfsTree.nodes[v].vertex] === 0}
                nodeState={(v) => colourState(b.colour.data[bfsTree.nodes[v].vertex])}
                edgeState={(c) => (bfsTree.edges[c]!.edge === b.edge ? 'active' : 'done')}
                nodeNotes={(v) => ({ e: `$${b.depth.data[bfsTree.nodes[v].vertex]}$` })}
              />
            ),
          },
          {
            title: 'depth-first tree',
            body: (
              <TreeView
                tree={dfsTree}
                height="fill"
                ariaLabel="Depth-first tree"
                hidden={(v) => d.colour.data[dfsTree.nodes[v].vertex] === 0}
                nodeState={(v) => colourState(d.colour.data[dfsTree.nodes[v].vertex])}
                edgeState={(c) => (dfsTree.edges[c]!.edge === d.edge ? 'active' : 'done')}
                nodeNotes={(v) => {
                  const x = dfsTree.nodes[v].vertex
                  const f = d.finish.data[x]
                  return { e: `$${d.discovery.data[x]}/${f >= 0 ? f : '\\cdot'}$` }
                }}
              />
            ),
          },
        ]}
      />
    </Figure>
  )
}

// ── Kahn's topological sort ─────────────────────────────────────────────────────────────────────────────────────────

/** CLRS Figure 22.7: getting dressed. */
const CLOTHES = ['undershorts', 'pants', 'belt', 'shirt', 'tie', 'jacket', 'socks', 'shoes', 'watch']
const DRESSING: Graph = (() => {
  const at = (s: string) => CLOTHES.indexOf(s)
  const pairs = [
    ['undershorts', 'pants'],
    ['undershorts', 'shoes'],
    ['pants', 'belt'],
    ['pants', 'shoes'],
    ['belt', 'jacket'],
    ['shirt', 'belt'],
    ['shirt', 'tie'],
    ['tie', 'jacket'],
    ['socks', 'shoes'],
  ]
  return fromEdges(
    CLOTHES.length,
    pairs.map(([a, b]): [number, number] => [at(a), at(b)]),
    { directed: true, labels: CLOTHES },
  )
})()
const PILL = (): Partial<DiagramNode> => ({ shape: 'pill', h: 0.7 })

export function KahnSpecimen() {
  const t = useMemo(() => trace(kahnSteps(DRESSING), undefined, 100), [])
  const [step, setStep] = useState(0)
  const s = stateAt(t, step)
  const output = new Set(Array.from(s.order.data))
  const queued = new Set(Array.from(s.queue.data))
  const decremented = new Set(Array.from(s.decremented.data))
  const nodeState = (v: number): ElementState => (output.has(v) ? 'done' : queued.has(v) ? 'active' : 'idle')
  return (
    <Figure
      title="Kahn's algorithm: getting dressed"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="step" value={step} onChange={setStep} count={t.steps.length} defaultSpeed={1.5} />
        </div>
      }
      readouts={
        <>
          <Readout label="output" value={s.current >= 0 ? CLOTHES[s.current] : '—'} />
          <Readout label="released" value={Array.from(s.released.data, (v) => CLOTHES[v]).join(', ') || '—'} />
        </>
      }
      caption="Each node carries its in-degree counting only edges from nodes not yet output. The queue holds the nodes at 0 (active). Each step outputs the front of the queue (done) and decrements its out-neighbours along the highlighted edges; those reaching 0 join the queue. The order below is a topological order: every edge points forward in it."
    >
      <Framed
        footer={
          <>
            <Sequence label="queue" ends="front first" items={Array.from(s.queue.data, (v) => CLOTHES[v])} />
            <Sequence label="order" items={Array.from(s.order.data, (v) => CLOTHES[v])} />
          </>
        }
      >
        <GraphView
          graph={DRESSING}
          height="fill"
          layered={{ nodeGap: 1.3, layerGap: 3 }}
          ariaLabel="Clothing dependencies (CLRS Figure 22.7)"
          node={PILL}
          nodeState={nodeState}
          nodeHighlight={(v) => v === s.current}
          nodeNotes={(v) => ({ s: `$\\deg^- = ${s.inDegree.data[v]}$` })}
          edgeState={(k) => (decremented.has(k) ? 'active' : output.has(DRESSING.edges[k].from) ? 'done' : 'idle')}
        />
      </Framed>
    </Figure>
  )
}

// ── Tarjan's strongly connected components ──────────────────────────────────────────────────────────────────────────

/** CLRS Figure 22.9. */
const SCC = lettered('abcdefgh', ['ab', 'bc', 'be', 'bf', 'cd', 'cg', 'dc', 'dh', 'ea', 'ef', 'fg', 'gf', 'gh', 'hh'])
const SCC_AT = placed('abcdefgh', {
  a: [0, 0],
  b: [2.5, 0],
  c: [5, 0],
  d: [7.5, 0],
  e: [0, 2.5],
  f: [2.5, 2.5],
  g: [5, 2.5],
  h: [7.5, 2.5],
})

function describeTarjan(s: TarjanState): string {
  const name = (v: number) => SCC.labels![v]
  const e = s.edge >= 0 ? SCC.edges[s.edge] : null
  switch (s.event) {
    case 'root':
      return `start at ${name(s.current)}`
    case 'tree':
      return `${name(e!.from)}→${name(e!.to)}: visit`
    case 'back':
      return `${name(e!.from)}→${name(e!.to)}: on the stack, lower low-link`
    case 'ignore':
      return `${name(e!.from)}→${name(e!.to)}: into a finished component`
    case 'return':
      return `finish ${name(s.current)}: pass its low-link up`
    case 'component':
      return `finish ${name(s.current)}: root of {${Array.from(s.emitted.data, name).join(', ')}}`
    default:
      return s.event
  }
}

export function TarjanSpecimen() {
  const t = useMemo(() => trace(tarjanSteps(SCC), undefined, 500), [])
  const [step, setStep] = useState(0)
  const s = stateAt(t, step)
  const comp = s.component.data
  const nodeState = (v: number): ElementState => (comp[v] >= 0 ? 'done' : s.index.data[v] >= 0 ? 'active' : 'idle')
  return (
    <Figure
      title="Tarjan's strongly connected components"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="step" value={step} onChange={setStep} count={t.steps.length} defaultSpeed={2} />
        </div>
      }
      readouts={
        <>
          <Readout label="step" value={describeTarjan(s)} />
          <Readout label="components" value={s.count} />
        </>
      }
      caption="Each visited node shows index/low-link: its discovery index, and the least index reachable from its subtree through one edge to a node still on Tarjan's stack. Visited nodes are active until their component is complete. When a node finishes with low-link equal to its index it is the root of a component: the stack is popped down to it and the component takes a colour (numbered in reverse topological order)."
    >
      <Framed
        footer={
          <>
            <Sequence label="Tarjan's stack" ends="bottom first" items={letters(SCC, s.stack.data)} />
            <Sequence label="search path" ends="root first" items={letters(SCC, s.path.data)} />
          </>
        }
      >
        <GraphView
          graph={SCC}
          layout={SCC_AT}
          height="fill"
          ariaLabel="CLRS Figure 22.9"
          nodeState={nodeState}
          nodeTone={(v) => (comp[v] >= 0 ? comp[v] % 8 : undefined)}
          nodeHighlight={(v) => v === s.current}
          nodeNotes={(v) => (s.index.data[v] >= 0 ? { n: `$${s.index.data[v]}/${s.lowlink.data[v]}$` } : undefined)}
          edgeState={(k) => {
            const e = SCC.edges[k]
            if (k === s.edge) return 'active'
            return comp[e.from] >= 0 && comp[e.from] === comp[e.to] ? 'done' : 'idle'
          }}
          edgeTone={(k) => {
            const e = SCC.edges[k]
            return comp[e.from] >= 0 && comp[e.from] === comp[e.to] ? comp[e.from] % 8 : undefined
          }}
        />
      </Framed>
    </Figure>
  )
}
