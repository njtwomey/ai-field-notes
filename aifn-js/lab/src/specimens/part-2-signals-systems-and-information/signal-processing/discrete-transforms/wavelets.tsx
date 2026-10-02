import type { Specimen } from '@lab/specimen'
import { ScalogramSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/discrete-transforms',
    title: 'Wavelet scalogram',
    description: 'The Morlet continuous wavelet transform of a frequency switch, a chirp and an impulse.',
    tags: ['wavelet', 'CWT', 'Morlet', 'time–frequency'],
    render: () => <ScalogramSpecimen />,
  },
]
