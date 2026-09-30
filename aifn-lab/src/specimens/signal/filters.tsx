import type { Specimen } from '../../specimen'
import { FilterSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'signal/filters',
    title: 'Filter responses and filtering',
    description:
      'Butterworth, Chebyshev and FIR low-pass filters: magnitude responses and causal against zero-phase filtering.',
    tags: ['Butterworth', 'Chebyshev', 'FIR', 'lfilter', 'filtfilt', 'freqz'],
    render: () => <FilterSpecimen />,
  },
]
