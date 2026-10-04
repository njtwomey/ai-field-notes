import { Diagram, Figure, op } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const cols: [string, number, string][] = [
  ['1', 0, '1'],
  ['2', 2, '2'],
  ['S', 4.6, 'S'],
]

const spec: DiagramSpec = {
  nodes: [
    ...cols.flatMap(([k, x, j]) => [
      { id: `x${k}`, x, y: 6.5, shape: 'text' as const, label: `$x_${j}$` },
      { id: `h${k}`, x, y: 5.2, w: 1.1, h: 0.7, label: `$\\hvec_${j}$`, tone: 0 },
      op(`m${k}`, x, 3.8, '$\\otimes$'),
      { id: `a${k}`, x: x + 0.7, y: 3.8, shape: 'text' as const, small: true, label: `$\\alpha_{i${j}}$` },
    ]),
    { id: 'hd', x: 3.3, y: 5.2, shape: 'text', w: 0.5, label: '$\\cdots$' },
    op('sum', 2, 2.2, '$\\oplus$'),
    { id: 'si', x: 2, y: 0.6, w: 1.1, h: 0.7, label: '$\\svec_i$', tone: 1 },
    { id: 'yi', x: 2, y: -0.7, shape: 'text', label: '$y_i$' },
    { id: 'yprev', x: 4.2, y: 0.6, shape: 'text', label: '$y_{i-1}$' },
    { id: 'sprev', x: -3.4, y: 0.6, w: 1.1, h: 0.7, label: '$\\svec_{i-1}$', tone: 1 },
    {
      id: 'align',
      x: -3.4,
      y: 3.8,
      w: 3,
      h: 1.3,
      label: 'alignment model\n$e_{ij} = a(\\svec_{i-1}, \\hvec_j)$\nsoftmax over $j$',
      tone: 3,
    },
  ],
  edges: [
    ...cols.flatMap(([k]) => [
      { from: `x${k}`, to: `h${k}` },
      { from: `h${k}:n`, to: `m${k}:s` },
    ]),
    { from: 'h1', to: 'h2', arrow: 'both' },
    { from: 'h2', to: 'hd', arrow: 'both' },
    { from: 'hd', to: 'hS', arrow: 'both' },
    { from: 'm1:n', to: 'sum:w', via: [[0, 2.2]] },
    { from: 'm2:n', to: 'sum:s' },
    { from: 'mS:n', to: 'sum:e', via: [[4.6, 2.2]] },
    { from: 'sum:n', to: 'si:s', label: '$\\cvec_i$', labelRotate: false },
    { from: 'si', to: 'yi' },
    { from: 'yprev', to: 'si' },
    { from: 'sprev', to: 'si' },
    { from: 'sprev:s', to: 'align:n' },
    { from: 'h1:w', to: 'align:s', via: [[-3.4, 5.2]], label: 'every $\\hvec_j$' },
    { from: 'align:e', to: 'm1:w', dashed: true, label: 'weights' },
  ],
  groups: [
    { id: 'enc', label: 'bidirectional encoder', tone: 0, around: ['h1', 'hS'], pad: 0.35, labelAt: 'bottom-right' },
  ],
}

/** Additive attention: at each decoder step, a fresh weighted average of all encoder annotations. */
export function AttentionDiagram() {
  return (
    <Figure
      title="One decoder step with additive attention"
      caption="The alignment model scores every encoder annotation against the previous decoder state, and a softmax over source positions turns the scores into weights. The context is the weighted sum of the annotations. The decoder state reads the context, its previous state and the previous output token; the weights are recomputed at every step."
    >
      <Diagram
        spec={spec}
        ariaLabel="Bahdanau attention: encoder annotations h_j weighted by alpha_ij and summed into context c_i, which feeds decoder state s_i; weights come from an alignment model reading s_(i-1) and h_j"
      />
    </Figure>
  )
}
