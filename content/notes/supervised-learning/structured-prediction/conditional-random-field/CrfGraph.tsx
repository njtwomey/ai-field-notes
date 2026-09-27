import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive, ParamSwitch, type GraphEdge, type GraphNode } from '@/components/viz'
import { at, chainEnds, labelNodes, potentialChain } from '../_shared/chain-graph'

/** Undirected chain of labels, every label also joined to the whole observed sequence x. */
function undirectedChain(): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const ends = chainEnds(0, 'y0', 'y2', false)
  return {
    nodes: [...labelNodes(0), { id: 'x', label: '\\mathbf{x}', x: at(1), y: 1.2, kind: 'observed' }, ...ends.nodes],
    edges: [
      { source: 'y0', target: 'y1', directed: false },
      { source: 'y1', target: 'y2', directed: false },
      ...[0, 1, 2].map((i) => ({ source: 'x', target: `y${i}`, directed: false })),
      ...ends.edges,
    ],
  }
}

/** The linear-chain CRF as an undirected graph conditioned on x, and as a factor graph of its potentials. */
export function CrfGraph() {
  const [factors, setFactors] = useState(false)
  const g = factors ? potentialChain() : undirectedChain()
  return (
    <Interactive
      title="The linear-chain CRF as a graph"
      caption={
        <MathText
          text={
            factors
              ? 'Factor graph: a node potential $\\psivec_n$ on each label and an edge potential $\\Psimat_n$ between neighbours. Both are computed from $\\xvec$, which is fixed.'
              : 'Labels form an undirected chain. The observation sequence $\\xvec$ (shaded) is conditioned on and can reach every position.'
          }
        />
      }
      controls={<ParamSwitch label="factor graph" checked={factors} onChange={setFactors} />}
    >
      <GraphDiagram nodes={g.nodes} edges={g.edges} height={190} ariaLabel="Linear-chain CRF" />
    </Interactive>
  )
}
