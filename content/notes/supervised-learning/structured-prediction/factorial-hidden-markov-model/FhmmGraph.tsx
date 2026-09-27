import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive, ParamSwitch, type GraphEdge, type GraphNode } from '@/components/viz'
import { POSITIONS, at } from '../_shared/chain-graph'

const CHAINS = [1, 2, 3]
const ROW = 0.85

const NODES: GraphNode[] = [
  ...CHAINS.flatMap((m, r) =>
    POSITIONS.map((p, i) => ({ id: `y${m}${i}`, label: `y^{(${m})}_{${p}}`, x: at(i), y: r * ROW })),
  ),
  ...POSITIONS.map((p, i) => ({
    id: `x${i}`,
    label: `x_{${p}}`,
    x: at(i) + 0.55,
    y: CHAINS.length * ROW + 0.2,
    kind: 'observed' as const,
  })),
]

const DIRECTED: GraphEdge[] = CHAINS.flatMap((m) => [
  { source: `y${m}0`, target: `y${m}1` },
  { source: `y${m}1`, target: `y${m}2` },
  ...[0, 1, 2].map((i) => ({ source: `y${m}${i}`, target: `x${i}` })),
])

// Moralising at each x_n joins every pair of its parents, the states of all chains at position n.
const MORAL: GraphEdge[] = [0, 1, 2].flatMap((i) =>
  [
    [1, 2],
    [2, 3],
    [1, 3],
  ].map(([a, b]) => ({
    source: `y${a}${i}`,
    target: `y${b}${i}`,
    directed: false,
    dashed: true,
    highlight: true,
    // The edge from chain 1 to chain 3 bends around chain 2.
    curveness: b - a > 1 ? 0.45 : 0,
  })),
)

/** M = 3 hidden chains share each observation; moralising shows how observing x_n couples them. */
export function FhmmGraph() {
  const [moral, setMoral] = useState(false)
  return (
    <Interactive
      title="The factorial HMM as a graph"
      caption={
        <MathText
          text={
            moral
              ? 'The dashed edges join the parents of each observed $x_n$. Conditioning on $x_n$ couples the states $y^{(1)}_n, \\dots, y^{(M)}_n$ of every chain, so the posterior does not factorise over chains.'
              : 'Three hidden chains $\\yvec^{(1)}, \\yvec^{(2)}, \\yvec^{(3)}$ evolve independently. Each observation $x_n$ (shaded) has the states of all chains at position $n$ as parents.'
          }
        />
      }
      controls={<ParamSwitch label="couple chains given x" checked={moral} onChange={setMoral} />}
    >
      <GraphDiagram
        nodes={NODES}
        edges={moral ? [...DIRECTED, ...MORAL] : DIRECTED}
        height={300}
        ariaLabel="Factorial hidden Markov model with three chains"
      />
    </Interactive>
  )
}
