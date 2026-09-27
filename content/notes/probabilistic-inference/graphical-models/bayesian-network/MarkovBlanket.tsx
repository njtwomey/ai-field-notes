import { useState } from 'react'
import { GraphDiagram, Interactive, ParamChoice, Readout, type GraphEdge, type GraphNode } from '@/components/viz'

const NODES = [
  { id: 'a', x: 0, y: 0 },
  { id: 'b', x: 1.5, y: 0 },
  { id: 'd', x: 3.5, y: 0 },
  { id: 'c', x: 0.75, y: 1 },
  { id: 'e', x: 2.75, y: 1 },
  { id: 'g', x: 0, y: 2 },
  { id: 'f', x: 1.75, y: 2 },
  { id: 'h', x: 1.75, y: 3 },
] as const satisfies readonly GraphNode[]

type Id = (typeof NODES)[number]['id']

const EDGES: GraphEdge[] = [
  { source: 'a', target: 'c' },
  { source: 'b', target: 'c' },
  { source: 'd', target: 'e' },
  { source: 'c', target: 'g' },
  { source: 'c', target: 'f' },
  { source: 'e', target: 'f' },
  { source: 'f', target: 'h' },
]

const parents = (i: string) => EDGES.filter((e) => e.target === i).map((e) => e.source)
const children = (i: string) => EDGES.filter((e) => e.source === i).map((e) => e.target)

function blanket(i: string) {
  const ch = children(i)
  const coParents = [...new Set(ch.flatMap(parents))].filter((j) => j !== i)
  return { parents: parents(i), children: ch, coParents }
}

const list = (ids: string[]) => (ids.length ? [...ids].sort().join(', ') : 'none')

/** Pick a node; its parents, children and children's other parents are shaded, as the set to condition on. */
export function MarkovBlanket() {
  const [node, setNode] = useState<Id>('c')
  const mb = blanket(node)
  const inBlanket = new Set([...mb.parents, ...mb.children, ...mb.coParents])
  return (
    <Interactive
      title="The Markov blanket of a node"
      caption="Click a node, or choose it below. Its Markov blanket, the parents, children and children's other parents, is shaded. Given the shaded variables, the chosen node is independent of every unshaded one."
      controls={
        <ParamChoice
          label="node"
          value={node}
          onChange={setNode}
          options={NODES.map((n) => ({ value: n.id, label: n.id }))}
        />
      }
      readout={
        <>
          <Readout label="parents" value={list(mb.parents)} />
          <Readout label="children" value={list(mb.children)} />
          <Readout label="co-parents" value={list(mb.coParents)} />
        </>
      }
    >
      <GraphDiagram
        nodes={NODES.map((n) => ({ ...n, kind: inBlanket.has(n.id) ? 'observed' : 'variable' }))}
        edges={EDGES}
        highlight={[node]}
        height={260}
        onNodeClick={(id) => NODES.some((n) => n.id === id) && setNode(id as Id)}
        ariaLabel={`Directed graph with the Markov blanket of ${node} shaded`}
      />
    </Interactive>
  )
}
