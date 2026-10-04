import type { Specimen } from '@lab/specimen'
import { CircularConvolutionFigure, ConvolutionTheoremFigure } from '../_transforms/convolution'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/discrete-transforms',
    title: 'Convolution and the convolution theorem',
    description:
      'Convolution in time against multiplication of DFTs, with a kernel whose taps are dragged; circular against linear convolution as both inputs are zero-padded.',
    tags: ['convolution', 'convolution theorem', 'circular convolution', 'zero-padding', 'kernel', 'convolve', 'fft'],
    render: () => (
      <>
        <ConvolutionTheoremFigure />
        <CircularConvolutionFigure />
      </>
    ),
  },
]
