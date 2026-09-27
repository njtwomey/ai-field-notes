import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from '@/components/viz'

const SPEC: DiagramSpec = {
  unit: 40,
  nodes: [
    factor('p0', 0, 0, '$\\Gauss(0, 1)$'),
    factor('p1', 4, 0, '$\\Gauss(0, 1)$'),
    factor('p2', 8, 0, '$\\Gauss(0, 1)$'),
    variable('a', 0, 1.6, '$s_A$'),
    variable('b', 4, 1.6, '$s_B$'),
    variable('c', 8, 1.6, '$s_C$'),
    factor('g1', 2, 3.6, '$f_1 = \\Phi(s_A - s_B)$', 's'),
    factor('g2', 6, 3.6, '$f_2 = \\Phi(s_B - s_C)$', 's'),
  ],
  edges: [
    link('p0', 'a', false),
    link('p1', 'b', false),
    link('p2', 'c', false),
    link('a', 'g1', false, { label: '$m_{1 \\to A}$', labelSide: 'right' }),
    link('b', 'g1', false, { label: '$m_{1 \\to B}$' }),
    link('b', 'g2', false, { label: '$m_{2 \\to B}$', labelSide: 'right' }),
    link('c', 'g2', false, { label: '$m_{2 \\to C}$' }),
  ],
}

/** Three players, two games: A beat B and B beat C. Squares are factors, circles skills. */
export function ComparisonGraph() {
  return (
    <Interactive
      title="Factor graph for two comparisons"
      caption="Each skill has a Gaussian prior factor (top). Each game is a probit factor on the difference of two skills (bottom). EP sends a Gaussian message from each game factor to each of its players; player B receives one from each game, and each depends on the other through B's cavity."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Factor graph: prior factors above skills s_A, s_B, s_C; factor f1 joins s_A and s_B; factor f2 joins s_B and s_C"
      />
    </Interactive>
  )
}
