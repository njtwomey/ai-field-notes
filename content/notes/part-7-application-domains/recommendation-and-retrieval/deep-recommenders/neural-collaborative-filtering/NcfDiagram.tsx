import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'u', x: 2.6, y: 8.2, w: 2.6, h: 0.6, label: 'user one-hot', tone: 'neutral' },
    { id: 'i', x: 8.4, y: 8.2, w: 2.6, h: 0.6, label: 'item one-hot', tone: 'neutral' },
    { id: 'pg', x: 1.4, y: 6.8, w: 1.6, h: 0.6, label: '$\\pvec_u^{G}$', tone: 0 },
    { id: 'pm', x: 3.8, y: 6.8, w: 1.6, h: 0.6, label: '$\\pvec_u^{M}$', tone: 1 },
    { id: 'qg', x: 7.2, y: 6.8, w: 1.6, h: 0.6, label: '$\\qvec_i^{G}$', tone: 0 },
    { id: 'qm', x: 9.6, y: 6.8, w: 1.6, h: 0.6, label: '$\\qvec_i^{M}$', tone: 1 },
    { id: 'had', x: 2.6, y: 4.6, shape: 'op', label: '$\\odot$' },
    { id: 'cat', x: 7.9, y: 5.4, w: 2.4, h: 0.6, label: 'concatenate', tone: 'neutral' },
    { id: 'mlp', x: 7.9, y: 3.9, shape: 'stack', w: 2.6, h: 1, label: 'MLP layers', tone: 1 },
    { id: 'gl', x: 1.2, y: 4.6, shape: 'text', small: true, label: 'GMF' },
    { id: 'ml', x: 10.2, y: 3.9, shape: 'text', small: true, label: 'MLP' },
    { id: 'neu', x: 5.2, y: 2.2, w: 3.6, h: 0.7, label: 'NeuMF layer: $\\sigma(\\hvec\\transpose[\\cdot])$', tone: 2 },
    { id: 'y', x: 5.2, y: 0.9, shape: 'text', label: '$\\hat{y}_{ui}$' },
  ],
  edges: [
    { from: 'u:s', to: 'pg:n', via: [[1.4, 7.6]] },
    { from: 'u:s', to: 'pm:n', via: [[3.8, 7.6]] },
    { from: 'i:s', to: 'qg:n', via: [[7.2, 7.6]] },
    { from: 'i:s', to: 'qm:n', via: [[9.6, 7.6]] },
    { from: 'pg:s', to: 'had:w', via: [[1.4, 4.6]] },
    {
      from: 'qg:s',
      to: 'had:e',
      via: [
        [7.2, 6.1],
        [5.6, 6.1],
        [5.6, 4.6],
      ],
    },
    { from: 'pm:s', to: 'cat:w', via: [[3.8, 5.4]] },
    { from: 'qm:s', to: 'cat:e', via: [[9.6, 5.4]] },
    { from: 'cat', to: 'mlp' },
    { from: 'had:s', to: 'neu:w', via: [[2.6, 2.2]] },
    { from: 'mlp:s', to: 'neu:e', via: [[7.9, 2.2]] },
    { from: 'neu', to: 'y' },
  ],
}

/** NeuMF: a GMF branch (element-wise product) and an MLP branch on separate embeddings, fused by a final layer. */
export function NcfDiagram() {
  return (
    <Interactive
      title="Neural matrix factorisation (NeuMF)"
      caption="The GMF branch multiplies a user and an item embedding element-wise; the MLP branch concatenates a second pair of embeddings and passes them through hidden layers. The last layers of both branches are concatenated and mapped to a probability. With the MLP branch removed and the output weights fixed to 1, the model is matrix factorisation."
    >
      <Diagram
        spec={spec}
        ariaLabel="NeuMF: separate GMF and MLP embeddings for user and item, an element-wise product and an MLP, fused by a sigmoid layer"
      />
    </Interactive>
  )
}
