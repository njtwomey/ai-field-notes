import type { Specimen } from '../../specimen'
import { SiftSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'signal/decompositions',
    title: 'Empirical mode decomposition',
    description: 'Sifting one intrinsic mode function, step by step, with its envelopes.',
    tags: ['EMD', 'sifting', 'trace', 'Hilbert–Huang'],
    render: () => <SiftSpecimen />,
  },
]
