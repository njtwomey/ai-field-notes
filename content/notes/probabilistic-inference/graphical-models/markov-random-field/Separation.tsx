import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import { Interactive, ParamChoice, Readout } from '@/components/viz'

type Vertex = { id: string; label: string; x: number; y: number }
type Graph = { nodes: Vertex[]; edges: [string, string][]; a: string; b: string; separator: string[] }

// A chain of five spins, and a 3 × 3 grid numbered row by row.
const GRAPHS: Record<'chain' | 'grid', Graph> = {
  chain: {
    nodes: [1, 2, 3, 4, 5].map((i) => ({ id: `${i}`, label: `x_${i}`, x: 1.5 * i, y: 0 })),
    edges: [1, 2, 3, 4].map((i): [string, string] => [`${i}`, `${i + 1}`]),
    a: '1',
    b: '5',
    separator: ['3'],
  },
  grid: {
    nodes: [...Array(9).keys()].map((i) => ({
      id: `${i + 1}`,
      label: `x_${i + 1}`,
      x: 1.5 * (i % 3),
      y: 1.5 * Math.floor(i / 3),
    })),
    edges: [...Array(9).keys()].flatMap((i): [string, string][] => [
      ...(i % 3 < 2 ? [[`${i + 1}`, `${i + 2}`] as [string, string]] : []),
      ...(i < 6 ? [[`${i + 1}`, `${i + 4}`] as [string, string]] : []),
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
    for (const [s, t] of g.edges) {
      const v = s === u ? t : t === u ? s : undefined
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
  const label = (id: string) => `$${g.nodes.find((n) => n.id === id)!.label}$`
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
      <Diagram
        spec={{
          unit: 48,
          nodes: g.nodes.map((n) =>
            variable(n.id, n.x, n.y, `$${n.label}$`, { filled: s.has(n.id), highlight: reach.has(n.id) }),
          ),
          edges: g.edges.map(([a, b]) => link(a, b, false)),
        }}
        onNodeClick={toggle}
        ariaLabel={`Undirected ${which} with an observed set`}
      />
    </Interactive>
  )
}
