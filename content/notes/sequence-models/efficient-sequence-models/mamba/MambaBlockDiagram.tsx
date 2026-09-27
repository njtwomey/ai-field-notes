import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import { op } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'

const L = 1.6
const R = 4.8
const C = (L + R) / 2

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: C, y: 8, shape: 'text', w: 2.4, label: 'input, width $D$' },
    { id: 'res', x: C, y: 7.3, shape: 'dot' },
    { id: 'norm', x: C, y: 6.6, w: 2, h: 0.6, label: 'norm', tone: 'neutral' },
    { id: 'in', x: C, y: 5.5, w: 4.4, h: 0.7, label: 'linear, expand to $E D$', tone: 'neutral' },
    { id: 'conv', x: L, y: 4.1, w: 2.2, h: 0.7, label: 'causal conv', tone: 0 },
    { id: 's1', x: L, y: 3.1, shape: 'pill', w: 1.2, h: 0.5, label: 'SiLU', tone: 0, small: true },
    { id: 'ssm', x: L, y: 2, w: 2.4, h: 0.8, label: 'selective SSM', tone: 0 },
    { id: 's2', x: R, y: 3.1, shape: 'pill', w: 1.2, h: 0.5, label: 'SiLU', tone: 1, small: true },
    op('mul', C, 0.9, '$\\otimes$'),
    { id: 'outp', x: C, y: -0.1, w: 3.2, h: 0.7, label: 'linear, back to $D$', tone: 'neutral' },
    op('add', C, -1.1, '$\\oplus$'),
    { id: 'y', x: C, y: -2, shape: 'text', w: 1.2, label: 'output' },
  ],
  edges: [
    { from: 'x', to: 'res', arrow: 'none' },
    { from: 'res', to: 'norm' },
    { from: 'norm', to: 'in' },
    { from: 'in:n', to: 'conv:s' },
    { from: 'in:n', to: 's2:s' },
    { from: 'conv', to: 's1' },
    { from: 's1', to: 'ssm' },
    { from: 'ssm:n', to: 'mul:w', via: [[L, 0.9]] },
    { from: 's2:n', to: 'mul:e', via: [[R, 0.9]], label: 'gate', labelRotate: false },
    { from: 'mul', to: 'outp' },
    { from: 'outp', to: 'add' },
    { from: 'add', to: 'y' },
    {
      from: 'res:e',
      to: 'add:e',
      via: [
        [6.8, 7.3],
        [6.8, -1.1],
      ],
      label: 'residual',
    },
  ],
}

/** The Mamba block: one gated branch holding the selective SSM, wrapped in a residual connection. */
export function MambaBlockDiagram() {
  return (
    <Interactive
      title="The Mamba block"
      caption="Read from the bottom. A linear layer widens the input by the factor E into two branches. The left branch mixes along the sequence: a short causal convolution, SiLU, then the selective SSM. The right branch passes through SiLU and gates the left one elementwise. A linear layer returns to width D, and a residual connection wraps the block. There is no separate attention or MLP layer."
    >
      <Diagram
        spec={spec}
        ariaLabel="Mamba block: norm, linear expansion into two branches; one branch causal convolution, SiLU and selective SSM; the other SiLU as a gate; elementwise product, linear projection back, residual addition"
      />
    </Interactive>
  )
}
