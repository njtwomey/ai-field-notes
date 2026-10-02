import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramNode, DiagramSpec } from '@/components/diagram/types'

const IN: [string, number, string][] = [
  ['q', 1, 'Q'],
  ['k', 3.4, 'K'],
  ['v', 5.8, 'V'],
]

const spec: DiagramSpec = {
  nodes: [
    ...IN.flatMap(([id, x, m]): DiagramNode[] => [
      { id: `${id}in`, x, y: 6.2, shape: 'text', w: 0.8, label: `$\\${m}mat$` },
      { id: `${id}p`, x, y: 4.9, shape: 'stack', w: 1.8, h: 0.8, label: `$\\${m}mat \\Wmat_${m}^i$`, tone: 0 },
      { id: `${id}b`, x, y: 3.85, shape: 'dot' },
    ]),
    { id: 'att', x: 3.4, y: 3.3, shape: 'stack', w: 6.6, h: 1.1, label: 'scaled dot-product attention', tone: 1 },
    {
      id: 'heads',
      x: 9.4,
      y: 4.1,
      shape: 'text',
      w: 3,
      small: true,
      label: '$h$ copies, one per head\neach of width $d_k = d/h$',
    },
    {
      id: 'cat',
      x: 3.4,
      y: 1.8,
      w: 4,
      h: 0.7,
      label: 'concatenate $\\text{head}_1, \\dots, \\text{head}_h$',
      tone: 'neutral',
    },
    { id: 'wo', x: 3.4, y: 0.7, w: 2, h: 0.7, label: 'linear $\\Wmat_O$', tone: 0 },
    { id: 'out', x: 3.4, y: -0.3, shape: 'text', w: 3, label: 'output, width $d$' },
  ],
  edges: [
    ...IN.flatMap(([id]) => [
      { from: `${id}in`, to: `${id}p` },
      { from: `${id}p`, to: `${id}b` },
    ]),
    { from: 'att', to: 'cat' },
    { from: 'cat', to: 'wo' },
    { from: 'wo', to: 'out' },
  ],
}

/** Multi-head attention: h projected attentions in parallel, concatenated and mixed by one output projection. */
export function MultiHeadDiagram() {
  return (
    <Interactive
      title="Multi-head attention"
      caption="Read from the bottom. Each head projects the queries, keys and values into its own subspace with its own matrices and runs scaled dot-product attention there, so each head produces its own attention weights. The h head outputs are concatenated back to width h times d_v and mixed by the output projection."
    >
      <Diagram
        spec={spec}
        ariaLabel="Multi-head attention: Q, K and V each projected per head, scaled dot-product attention per head, heads concatenated and multiplied by W_O"
      />
    </Interactive>
  )
}
