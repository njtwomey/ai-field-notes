import type { Specimen } from '../../specimen'
import { ScalogramSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'signal/wavelets',
    title: 'Wavelet scalogram',
    description: 'The Morlet continuous wavelet transform of a frequency switch, a chirp and an impulse.',
    tags: ['wavelet', 'CWT', 'Morlet', 'time–frequency'],
    render: () => <ScalogramSpecimen />,
  },
]
