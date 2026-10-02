import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from 'aifn-render'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 42,
  nodes: [
    factor('pe', 4, -1.3, '$\\Bern(0.5)$', 'e'),
    variable('e', 4, 0, '$e$'),
    { id: 'd1', x: 1.6, y: 1.1, shape: 'dot', w: 0.18, h: 0.18 },
    { id: 'd0', x: 6.4, y: 1.1, shape: 'dot', w: 0.18, h: 0.18 },
    // Gate e = true: separate rates.
    factor('bc', 0, 1.9, '$\\Beta(1,1)$', 'w'),
    variable('pc', 0, 3.1, '$\\rho_c$'),
    factor('bt', 2.6, 1.9, '$\\Beta(1,1)$', 'e'),
    variable('pt', 2.6, 3.1, '$\\rho_t$'),
    factor('fc1', 0, 4.4, '$\\Bern$', 'w'),
    factor('ft1', 2.6, 4.4, '$\\Bern$', 'e'),
    // Gate e = false: one shared rate.
    factor('ba', 6.4, 1.9, '$\\Beta(1,1)$', 'e'),
    variable('pa', 6.4, 3.1, '$\\rho$'),
    factor('fc0', 5.4, 4.4, '$\\Bern$', 'w'),
    factor('ft0', 7.4, 4.4, '$\\Bern$', 'e'),
    variable('xc', 1.3, 6.2, '$x^{c}_i$', { filled: true }),
    variable('xt', 6.4, 6.2, '$x^{t}_j$', { filled: true }),
  ],
  edges: [
    line('pe', 'e'),
    line('e', 'd1', 'evidence'),
    link('e', 'd0', false),
    line('bc', 'pc'),
    line('bt', 'pt'),
    line('pc', 'fc1'),
    line('pt', 'ft1'),
    line('ba', 'pa'),
    line('pa', 'fc0'),
    line('pa', 'ft0'),
    line('fc1', 'xc'),
    line('fc0', 'xc'),
    line('ft1', 'xt'),
    line('ft0', 'xt'),
  ],
  groups: [
    {
      id: 'g1',
      label: '$e = \\text{true}$',
      tone: 0,
      dashed: true,
      around: ['bc', 'pc', 'bt', 'pt', 'fc1', 'ft1', 'd1'],
      pad: 0.5,
      labelAt: 'top-left',
    },
    {
      id: 'g0',
      label: '$e = \\text{false}$',
      tone: 1,
      dashed: true,
      around: ['ba', 'pa', 'fc0', 'ft0', 'd0'],
      pad: 0.5,
      labelAt: 'top-right',
    },
  ],
}

/** The clinical-trial model: a selector variable switching between two gated sub-models of the same data. */
export function TrialGraph() {
  return (
    <Interactive
      title="The clinical trial as a gated factor graph"
      caption="The selector e says whether the treatment has an effect. Inside the gate e = true, the control and treated groups have separate recovery rates; inside e = false, one rate serves both. The observed outcomes connect to a factor in each gate; only the gate matching e is on. The message from each gate to e is that sub-model's evidence."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Gated factor graph: selector e chooses between separate recovery rates and a shared rate, both explaining the observed outcomes"
      />
    </Interactive>
  )
}
