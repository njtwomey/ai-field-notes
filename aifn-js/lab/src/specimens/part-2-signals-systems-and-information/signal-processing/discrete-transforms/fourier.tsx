import type { Specimen } from '@lab/specimen'
import {
  AliasingFigure,
  InverseDftFigure,
  LeakageFigure,
  SinusoidsToDftFigure,
  ZeroPaddingFigure,
} from '../_transforms/fourier'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/discrete-transforms',
    title: 'Fourier: from sums of sinusoids to the DFT',
    description:
      'Build a record from sinusoids with draggable amplitude, frequency and phase on its spectrum, and read the DFT: magnitude and phase, leakage between bins and the window, zero-padding (interpolation, not resolution), aliasing as a frequency is dragged past Nyquist, and reconstruction by the inverse DFT.',
    tags: [
      'DFT',
      'FFT',
      'leakage',
      'window',
      'zero-padding',
      'aliasing',
      'inverse DFT',
      'Parseval',
      'fft',
      'rfft',
      'ifft',
    ],
    render: () => (
      <>
        <SinusoidsToDftFigure />
        <LeakageFigure />
        <ZeroPaddingFigure />
        <AliasingFigure />
        <InverseDftFigure />
      </>
    ),
  },
]
