import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'in', x: 0, y: 3, shape: 'text', w: 1.4, label: '$h_{l-1}$' },
    { id: 'split', x: 1.4, y: 3, shape: 'dot' },
    { id: 'conv', x: 4, y: 3, w: 3.4, h: 1.3, label: 'dilated causal conv\n2 taps, dilation $d_l$', tone: 0 },
    { id: 'tanh', x: 7.6, y: 2, shape: 'pill', w: 1.4, h: 0.8, label: 'tanh', tone: 'neutral' },
    { id: 'sig', x: 7.6, y: 4, shape: 'pill', w: 1.4, h: 0.8, label: '$\\sigma$', tone: 'neutral' },
    { id: 'mul', x: 9.4, y: 3, shape: 'op', label: '$\\odot$' },
    { id: 'zsplit', x: 10.3, y: 3, shape: 'dot' },
    { id: 'res', x: 12, y: 3, w: 2, h: 1, label: '$1 \\times 1$ conv', tone: 1 },
    { id: 'add', x: 14, y: 3, shape: 'op', label: '$\\oplus$' },
    { id: 'out', x: 15.5, y: 3, shape: 'text', w: 1.2, label: '$h_l$' },
    { id: 'skip', x: 12, y: 5.4, w: 2, h: 1, label: '$1 \\times 1$ conv', tone: 2 },
    { id: 'sum', x: 2, y: 7.6, shape: 'op', label: '$\\Sigma$' },
    { id: 'relu1', x: 3.9, y: 7.6, shape: 'pill', w: 1.5, h: 0.8, label: 'ReLU', tone: 'neutral' },
    { id: 'head1', x: 6.1, y: 7.6, w: 2, h: 1, label: '$1 \\times 1$ conv', tone: 'neutral' },
    { id: 'relu2', x: 8.3, y: 7.6, shape: 'pill', w: 1.5, h: 0.8, label: 'ReLU', tone: 'neutral' },
    { id: 'head2', x: 10.5, y: 7.6, w: 2, h: 1, label: '$1 \\times 1$ conv', tone: 'neutral' },
    { id: 'soft', x: 13.3, y: 7.6, w: 2.6, h: 1.2, label: 'softmax over\n256 levels', tone: 'neutral' },
  ],
  groups: [
    {
      id: 'block',
      label: 'residual block, repeated for every layer',
      around: ['conv', 'tanh', 'sig', 'mul', 'zsplit', 'res', 'add', 'skip'],
      pad: 0.5,
      dashed: true,
    },
  ],
  edges: [
    { from: 'in', to: 'split', arrow: 'none' },
    { from: 'split', to: 'conv' },
    {
      from: 'conv:e',
      to: 'tanh',
      via: [
        [6.2, 3],
        [6.2, 2],
      ],
      label: 'filter half',
    },
    {
      from: 'conv:e',
      to: 'sig',
      via: [
        [6.2, 3],
        [6.2, 4],
      ],
      label: 'gate half',
      labelSide: 'left',
    },
    { from: 'tanh:e', to: 'mul', via: [[9.4, 2]] },
    { from: 'sig:e', to: 'mul', via: [[9.4, 4]] },
    { from: 'mul', to: 'zsplit', arrow: 'none' },
    { from: 'zsplit', to: 'res' },
    { from: 'res', to: 'add' },
    { from: 'add', to: 'out' },
    { from: 'zsplit:s', to: 'skip', via: [[10.3, 5.4]] },
    {
      from: 'split:n',
      to: 'add:n',
      via: [
        [1.4, 0.9],
        [14, 0.9],
      ],
      label: 'residual',
    },
    {
      from: 'skip:s',
      to: 'sum:n',
      via: [
        [12, 6.6],
        [2, 6.6],
      ],
      label: 'skip outputs of all layers',
    },
    { from: 'sum', to: 'relu1' },
    { from: 'relu1', to: 'head1' },
    { from: 'head1', to: 'relu2' },
    { from: 'relu2', to: 'head2' },
    { from: 'head2', to: 'soft' },
  ],
}

/** One WaveNet residual block and the output head. */
export function WaveNetBlock() {
  return (
    <Figure
      title="Inside a WaveNet"
      hoverReadout={false}
      caption="One residual block of WaveNet (dashed box) and the output head below it. The dilated causal convolution produces two sets of channels. One passes through tanh (the filter), the other through a sigmoid (the gate), and their elementwise product is the block's activation. A 1 × 1 convolution of the activation is added to the block's input and passed to the next block (the residual path); a second 1 × 1 convolution sends a copy to the skip sum. The head adds the skip outputs of every block and maps them, through two ReLU and 1 × 1 convolution stages, to a softmax over the 256 μ-law levels of the next sample. Conditioning inputs, such as a speaker embedding or upsampled text features, are added inside both the filter and the gate."
    >
      <Diagram
        spec={spec}
        ariaLabel="WaveNet block: a dilated causal convolution splits into a tanh filter and a sigmoid gate, multiplied elementwise; a 1 by 1 convolution adds the result to the block input for the residual path, another 1 by 1 convolution sends it to a skip sum; the skip sum of all blocks passes through ReLU, 1 by 1 convolution, ReLU, 1 by 1 convolution and a softmax over 256 levels"
      />
    </Figure>
  )
}
