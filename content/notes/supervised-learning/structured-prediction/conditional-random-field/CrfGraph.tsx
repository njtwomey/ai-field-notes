import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive, ParamSwitch } from '@/components/viz'
import { at, chainEnds, labelNodes, potentialChain } from '../_shared/chain-graph'

/** Undirected chain of labels, every label also joined to the whole observed sequence x. */
function undirectedChain(): DiagramSpec {
  const ends = chainEnds(0, 'y0', 'y2', false)
  return {
    nodes: [...labelNodes(0), variable('x', at(1), 2, '$\\xvec$', { w: 1.05, h: 1.05, filled: true }), ...ends.nodes],
    edges: [
      link('y0', 'y1', false),
      link('y1', 'y2', false),
      ...[0, 1, 2].map((i) => link('x', `y${i}`, false)),
      ...ends.edges,
    ],
  }
}

const UNDIRECTED = undirectedChain()
const FACTORS = potentialChain()

/** The linear-chain CRF as an undirected graph conditioned on x, and as a factor graph of its potentials. */
export function CrfGraph() {
  const [factors, setFactors] = useState(false)
  const g = factors ? FACTORS : UNDIRECTED
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
      <Diagram spec={{ ...g, unit: 48 }} ariaLabel="Linear-chain CRF" />
    </Interactive>
  )
}
