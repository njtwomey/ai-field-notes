import type { Specimen } from '../specimen'
import { BodeMarginsSpecimen } from './_systems/figures'

export const specimens: Specimen[] = [
  {
    module: 'systems',
    title: 'Bode plot and margins',
    description: 'Magnitude and phase of an open loop with its gain, phase and delay margins marked.',
    tags: ['Bode', 'frequency response', 'gain margin', 'phase margin', 'delay'],
    render: () => <BodeMarginsSpecimen />,
  },
]
