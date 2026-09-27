import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive, ParamChoice, Readout, type GraphEdge, type GraphNode } from '@/components/viz'

type Graph = { nodes: GraphNode[]; edges: GraphEdge[]; a: string; b: string; separator: string[] }

const link = (source: string, target: string): GraphEdge => ({ source, target, directed: false })

// A chain of five spins, and a 3 × 3 grid numbered row by row.
const GRAPHS: Record<'chain' | 'grid', Graph> = {
  chain: {
    nodes: [1, 2, 3, 4, 5].map((i) => ({ id: `${i}`, label: `x_${i}`, x: i, y: 0 })),
    edges: [1, 2, 3, 4].map((i) => link(`${i}`, `${i + 1}`)),
    a: '1',
    b: '5',
    separator: ['3'],
  },
  grid: {
    nodes: [...Array(9).keys()].map((i) => ({ id: `${i + 1}`, label: `x_${i + 1}`, x: i % 3, y: Math.floor(i / 3) })),
    edges: [...Array(9).keys()].flatMap((i) => [
      ...(i % 3 < 2 ? [link(`${i + 1}`, `${i + 2}`)] : []),
      ...(i < 6 ? [link(`${i + 1}`, `${i + 4}`)] : []),
    ]),
    a: '1',
    b: '9',
    separator: ['3', '5', '7'],
  },
}

/** Nodes reachable from `start` along edges that avoid every node of `blocked`. */
function reachable(g: Graph, start: string, blocked: Set<string>): Set<string> {
  const seen = new Set([start])
  const stack = [start]
  while (stack.length) {
    const u = stack.pop()!
    for (const e of g.edges) {
      const v = e.source === u ? e.target : e.target === u ? e.source : undefined
      if (v && !blocked.has(v) && !seen.has(v)) {
        seen.add(v)
        stack.push(v)
      }
    }
  }
  return seen
}

/** Toggle the observed set S by clicking; the nodes reachable from A without passing through S are coloured. */
export function Separation() {
  const [which, setWhich] = useState<'chain' | 'grid'>('chain')
  const [observed, setObserved] = useState<Record<string, string[]>>({})
  const g = GRAPHS[which]
  const s = new Set(observed[which] ?? g.separator)
  const reach = reachable(g, g.a, s)
  const label = (id: string) => `$${g.nodes.find((n) => n.id === id)!.label!}$`
  const toggle = (id: string) => {
    if (id === g.a || id === g.b) return
    const next = new Set(s)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setObserved({ ...observed, [which]: [...next] })
  }
  return (
    <Interactive
      title="Separation in a Markov random field"
      caption={
        <MathText
          text={`Click a node to add it to the observed set $S$ (shaded) or remove it. The coloured nodes are those reachable from ${label(g.a)} without passing through $S$. $S$ separates ${label(g.a)} from ${label(g.b)} exactly when ${label(g.b)} is not coloured, and then the two are conditionally independent given $S$.`}
        />
      }
      controls={
        <ParamChoice
          label="graph"
          value={which}
          onChange={setWhich}
          options={[
            { value: 'chain', label: 'Ising chain' },
            { value: 'grid', label: '3 × 3 grid' },
          ]}
        />
      }
      readout={
        <>
          <Readout
            label={<MathText text="$S$" />}
            value={<MathText text={[...s].map(label).join(', ') || 'empty'} />}
          />
          <Readout
            label={<MathText text={`$S$ separates ${label(g.a)} from ${label(g.b)}`} />}
            value={reach.has(g.b) ? 'no' : 'yes'}
          />
        </>
      }
    >
      <GraphDiagram
        nodes={g.nodes.map((n) => ({ ...n, kind: s.has(n.id) ? 'observed' : 'variable' }))}
        edges={g.edges}
        highlight={[...reach]}
        height={which === 'chain' ? 110 : 240}
        onNodeClick={toggle}
        ariaLabel={`Undirected ${which} with an observed set`}
      />
    </Interactive>
  )
}
