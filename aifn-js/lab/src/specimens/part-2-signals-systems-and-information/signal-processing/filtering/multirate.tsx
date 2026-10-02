import type { Specimen } from '@lab/specimen'
import { DecimationFigure, InterpolationFigure, PolyphaseFigure, SamplingFigure } from '../_transforms/multirate'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/filtering',
    title: 'Sampling, aliasing and multirate',
    description:
      'Sampling a continuous signal and rebuilding it by sinc interpolation, with the Nyquist frequency dragged below the signal band; decimation with and without the anti-alias filter; interpolation by zero insertion and a low-pass; the polyphase form of a decimating filter.',
    tags: [
      'sampling theorem',
      'aliasing',
      'sinc interpolation',
      'decimation',
      'interpolation',
      'polyphase',
      'resamplePoly',
      'decimateSignal',
      'sincInterpolate',
      'upfirdn',
    ],
    render: () => (
      <>
        <SamplingFigure />
        <DecimationFigure />
        <InterpolationFigure />
        <PolyphaseFigure />
      </>
    ),
  },
]
