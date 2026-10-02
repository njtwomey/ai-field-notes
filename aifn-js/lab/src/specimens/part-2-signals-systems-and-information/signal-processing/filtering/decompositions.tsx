import type { Specimen } from '@lab/specimen'
import { SiftSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/filtering',
    title: 'Empirical mode decomposition',
    description: 'Sifting one intrinsic mode function, step by step, with its envelopes.',
    tags: ['EMD', 'sifting', 'trace', 'Hilbert–Huang'],
    render: () => <SiftSpecimen />,
  },
]
