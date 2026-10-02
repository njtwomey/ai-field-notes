import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import type { DiagramEdge, DiagramNode } from '@/components/diagram/types'
import { Interactive, ParamSwitch } from 'aifn-render'
import { POSITIONS } from '../_shared/chain-graph'

const CHAINS = [1, 2, 3]
const ROW = 1.5
const STEP = 3.2
// Each chain sits a little further left than the one above, so that the edges from the upper chains down to x_n
// pass beside the lower chains' nodes rather than through them.
const SHIFT = 0.8

const NODES: DiagramNode[] = [
  ...CHAINS.flatMap((m, r) =>
    POSITIONS.map((p, i) =>
      variable(`y${m}${i}`, STEP * i - SHIFT * r, r * ROW, `$y^{(${m})}_{${p}}$`, { w: 1.15, h: 1.15 }),
    ),
  ),
  ...POSITIONS.map((p, i) =>
    variable(`x${i}`, STEP * i + 0.9, CHAINS.length * ROW + 0.3, `$x_{${p}}$`, { w: 1.15, h: 1.15, filled: true }),
  ),
]

const DIRECTED: DiagramEdge[] = CHAINS.flatMap((m) => [
  link(`y${m}0`, `y${m}1`),
  link(`y${m}1`, `y${m}2`),
  ...[0, 1, 2].map((i) => link(`y${m}${i}`, `x${i}`)),
])

// Moralising at each x_n joins every pair of its parents, the states of all chains at position n.
const MORAL: DiagramEdge[] = [0, 1, 2].flatMap((i) =>
  [
    [1, 2],
    [2, 3],
    [1, 3],
  ].map(([a, b]) =>
    link(`y${a}${i}`, `y${b}${i}`, false, {
      dashed: true,
      highlight: true,
      // The edge from chain 1 to chain 3 bends around chain 2.
      ...(b - a > 1 && { route: 'curve' as const, bend: 1 }),
    }),
  ),
)

const PLAIN = { unit: 44, nodes: NODES, edges: DIRECTED }
const MORALISED = { unit: 44, nodes: NODES, edges: [...DIRECTED, ...MORAL] }

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
      <Diagram spec={moral ? MORALISED : PLAIN} ariaLabel="Factorial hidden Markov model with three chains" />
    </Interactive>
  )
}
