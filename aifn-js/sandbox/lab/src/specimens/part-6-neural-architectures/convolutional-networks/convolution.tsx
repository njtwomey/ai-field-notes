import type { Specimen } from '@lab/specimen'
import { ConvolutionSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/convolutional-networks',
    title: 'Convolution on a small image',
    description:
      'conv2d of a test image with edge, blur and sharpen kernels; stride, padding and dilation change the output size and the receptive field, which follows a draggable output position.',
    tags: ['conv2d', 'CNN', 'receptive field', 'pooling', 'stride', 'dilation'],
    render: () => <ConvolutionSpecimen />,
  },
]
