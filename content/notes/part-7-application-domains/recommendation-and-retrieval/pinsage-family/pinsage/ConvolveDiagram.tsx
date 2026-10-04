import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'z1', x: 1, y: 9, w: 1.4, h: 0.6, label: '$\\zvec_{v_1}$', tone: 1 },
    { id: 'z2', x: 3.2, y: 9, w: 1.4, h: 0.6, label: '$\\zvec_{v_2}$', tone: 1 },
    { id: 'z3', x: 5.4, y: 9, w: 1.4, h: 0.6, label: '$\\zvec_{v_T}$', tone: 1 },
    { id: 'dots', x: 4.3, y: 9, shape: 'text', label: '$\\cdots$' },
    { id: 'q1', x: 1, y: 7.4, w: 1.9, h: 0.7, label: 'dense, ReLU', tone: 0, small: true },
    { id: 'q2', x: 3.2, y: 7.4, w: 1.9, h: 0.7, label: 'dense, ReLU', tone: 0, small: true },
    { id: 'q3', x: 5.4, y: 7.4, w: 1.9, h: 0.7, label: 'dense, ReLU', tone: 0, small: true },
    { id: 'qshare', x: 3.2, y: 8.2, shape: 'text', small: true, label: 'shared $\\Qmat, \\qvec$' },
    { id: 'pool', x: 3.2, y: 5.6, shape: 'op', label: '$\\gamma$' },
    {
      id: 'poolt',
      x: 0.2,
      y: 4.6,
      shape: 'text',
      small: true,
      label: 'importance pooling:\nweights $\\alpha_v$ from\nrandom-walk visits',
    },
    { id: 'nu', x: 3.2, y: 4.3, w: 1.2, h: 0.6, label: '$\\nvec_u$', tone: 0 },
    { id: 'zu', x: 7.9, y: 9, w: 1.4, h: 0.6, label: '$\\zvec_u$', tone: 1 },
    { id: 'cat', x: 5.4, y: 3.1, shape: 'op', label: '$\\Vert$' },
    { id: 'catt', x: 6.3, y: 3.7, shape: 'text', small: true, label: 'concat' },
    { id: 'w', x: 5.4, y: 1.9, w: 2.4, h: 0.7, label: 'dense $\\Wmat, \\wvec$, ReLU', tone: 0, small: true },
    { id: 'norm', x: 5.4, y: 0.8, w: 1.9, h: 0.6, label: 'divide by $\\norm{\\cdot}_2$', tone: 'neutral', small: true },
    { id: 'out', x: 5.4, y: -0.3, shape: 'text', label: '$\\zvec_u^{\\mathrm{new}}$' },
  ],
  edges: [
    { from: 'z1', to: 'q1' },
    { from: 'z2', to: 'q2' },
    { from: 'z3', to: 'q3' },
    { from: 'q1:n', to: 'pool:w', via: [[1, 5.6]], label: '$\\alpha_{v_1}$', labelPos: 0.3, labelRotate: false },
    { from: 'q2:n', to: 'pool:s', label: '$\\alpha_{v_2}$', labelRotate: false },
    {
      from: 'q3:n',
      to: 'pool:e',
      via: [[5.4, 5.6]],
      label: '$\\alpha_{v_T}$',
      labelPos: 0.3,
      labelSide: 'right',
      labelRotate: false,
    },
    { from: 'pool', to: 'nu' },
    { from: 'nu:n', to: 'cat:w', via: [[3.2, 3.1]] },
    { from: 'zu:n', to: 'cat:e', via: [[7.9, 3.1]] },
    { from: 'cat', to: 'w' },
    { from: 'w', to: 'norm' },
    { from: 'norm', to: 'out' },
  ],
  groups: [
    { id: 'conv', label: 'convolve, layer k', around: ['z1', 'q1', 'pool', 'nu', 'cat', 'w', 'norm', 'zu'], pad: 0.5 },
  ],
}

/** One PinSage convolution: transform each sampled neighbour, pool with random-walk weights, concatenate, transform. */
export function ConvolveDiagram() {
  return (
    <Figure
      title="One PinSage convolution"
      caption="Each of the T sampled neighbours passes through the same dense layer. The results are averaged with weights from the random-walk visit counts. The pooled vector is concatenated with the node's own vector, passed through a second dense layer and scaled to unit length. Stacking K such modules, each with its own weights, and a final two-layer network gives the embedding."
    >
      <Diagram
        spec={spec}
        ariaLabel="PinSage convolve module: neighbour embeddings through a shared dense layer, importance pooling with random-walk weights, concatenation with the node's embedding, dense layer and L2 normalisation"
      />
    </Figure>
  )
}
