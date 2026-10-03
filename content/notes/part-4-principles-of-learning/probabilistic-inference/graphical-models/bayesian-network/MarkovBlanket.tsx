import { useState } from 'react'
import { Diagram } from 'aifn-render'
import { link, variable } from 'aifn-render'
import { Interactive, ParamChoice, Readout } from 'aifn-render'

const NODES = [
  { id: 'a', x: 0, y: 0 },
  { id: 'b', x: 1.5, y: 0 },
  { id: 'd', x: 3.5, y: 0 },
  { id: 'c', x: 0.75, y: 1 },
  { id: 'e', x: 2.75, y: 1 },
  { id: 'g', x: 0, y: 2 },
  { id: 'f', x: 1.75, y: 2 },
  { id: 'h', x: 1.75, y: 3 },
] as const

type Id = (typeof NODES)[number]['id']

const EDGES: [string, string][] = [
  ['a', 'c'],
  ['b', 'c'],
  ['d', 'e'],
  ['c', 'g'],
  ['c', 'f'],
  ['e', 'f'],
  ['f', 'h'],
]

const parents = (i: string) => EDGES.filter(([, t]) => t === i).map(([s]) => s)
const children = (i: string) => EDGES.filter(([s]) => s === i).map(([, t]) => t)

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
      <Diagram
        spec={{
          unit: 48,
          spread: [1.1, 1.3],
          nodes: NODES.map((n) =>
            variable(n.id, n.x, n.y, `$${n.id}$`, { filled: inBlanket.has(n.id), highlight: n.id === node }),
          ),
          edges: EDGES.map(([s, t]) => link(s, t)),
        }}
        onNodeClick={(id) => NODES.some((n) => n.id === id) && setNode(id as Id)}
        ariaLabel={`Directed graph with the Markov blanket of ${node} shaded`}
      />
    </Interactive>
  )
}
