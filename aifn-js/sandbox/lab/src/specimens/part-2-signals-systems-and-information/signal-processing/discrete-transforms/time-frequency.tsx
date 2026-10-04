import type { Specimen } from '@lab/specimen'
import { CrossTermFigure, StftTradeOffFigure, TimeFrequencyViewsFigure } from '../_transforms/time-frequency'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/discrete-transforms',
    title: 'Time–frequency: STFT, wavelets, Wigner–Ville',
    description:
      'One signal (a chirp, two tones and a click) under the STFT with its window-length trade-off, the Morlet CWT, the Wigner–Ville distribution and its smoothed pseudo form, the reassigned spectrogram and the synchrosqueezed STFT; and the cross-term between two draggable atoms.',
    tags: [
      'STFT',
      'spectrogram',
      'uncertainty principle',
      'CWT',
      'Wigner–Ville',
      'cross-terms',
      'Cohen class',
      'reassignment',
      'synchrosqueezing',
      'wignerVille',
      'synchrosqueeze',
    ],
    render: () => (
      <>
        <StftTradeOffFigure />
        <TimeFrequencyViewsFigure />
        <CrossTermFigure />
      </>
    ),
  },
]
