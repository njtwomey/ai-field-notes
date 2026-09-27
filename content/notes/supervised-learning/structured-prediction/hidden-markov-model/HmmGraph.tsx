import { useState } from 'react'
import { GraphDiagram, Interactive, ParamSwitch, type GraphEdge, type GraphNode } from '@/components/viz'
import { POSITIONS, at, chainEnds, labelNodes, potentialChain } from '../_shared/chain-graph'

function directedChain(): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const xs: GraphNode[] = POSITIONS.map((p, i) => ({
    id: `x${i}`,
    label: `x_{${p}}`,
    x: at(i),
    y: 1,
    kind: 'observed',
  }))
  const ends = chainEnds(0, 'y0', 'y2', true)
  return {
    nodes: [...labelNodes(0), ...xs, ...ends.nodes],
    edges: [
      { source: 'y0', target: 'y1' },
      { source: 'y1', target: 'y2' },
      ...[0, 1, 2].map((i) => ({ source: `y${i}`, target: `x${i}` })),
      ...ends.edges,
    ],
  }
}

/** The HMM as a directed graph, and the same chain in potential form with the observations absorbed. */
export function HmmGraph() {
  const [potentials, setPotentials] = useState(false)
  const g = potentials ? potentialChain() : directedChain()
  return (
    <Interactive
      title="The HMM as a graph"
      caption={
        potentials
          ? 'Potential form: each square is a factor. The node potential below each state absorbs its emission, and the edge potential between two states is the transition matrix.'
          : 'Each hidden state depends on the one before it and emits one observation. Shaded nodes are observed.'
      }
      controls={<ParamSwitch label="potential form" checked={potentials} onChange={setPotentials} />}
    >
      <GraphDiagram nodes={g.nodes} edges={g.edges} height={190} ariaLabel="Hidden Markov model chain" />
    </Interactive>
  )
}
