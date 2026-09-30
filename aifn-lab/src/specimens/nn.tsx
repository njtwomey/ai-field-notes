import type { Specimen } from '../specimen'
import { AttentionQuerySpecimen, ConvolutionSpecimen, MlpTrainingSpecimen, MultiHeadSpecimen } from './_nn/figures'

export const specimens: Specimen[] = [
  {
    module: 'nn',
    title: 'An MLP learning a decision surface',
    description:
      'A tanh MLP trained by Adam on two spirals, XOR or moons (a traceable training loop): the decision surface at any step, the loss trace, and every parameter tensor with its gradient.',
    tags: ['training', 'MLP', 'Adam', 'trace', 'ParamsView', 'decision surface'],
    render: () => <MlpTrainingSpecimen />,
  },
  {
    module: 'nn',
    title: 'Convolution on a small image',
    description:
      'conv2d of a test image with edge, blur and sharpen kernels; stride, padding and dilation change the output size and the receptive field, which follows a draggable output position.',
    tags: ['conv2d', 'CNN', 'receptive field', 'pooling', 'stride', 'dilation'],
    render: () => <ConvolutionSpecimen />,
  },
  {
    module: 'nn',
    title: 'Attention weights',
    description:
      'Scaled dot-product attention of a draggable query over six keys, and the per-head weight matrices of a causal multi-head self-attention layer.',
    tags: ['attention', 'transformer', 'softmax', 'causal mask', 'multi-head'],
    render: () => (
      <>
        <AttentionQuerySpecimen />
        <MultiHeadSpecimen />
      </>
    ),
  },
]
