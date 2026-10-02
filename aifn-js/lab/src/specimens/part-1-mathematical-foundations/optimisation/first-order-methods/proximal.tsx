import type { Specimen } from '@lab/specimen'
import { ProximalSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/optimisation/first-order-methods',
    title: 'ISTA and FISTA',
    description: 'Proximal gradient with soft thresholding on a lasso, with and without Nesterov acceleration.',
    tags: ['proximal', 'lasso', 'L1', 'acceleration'],
    render: () => <ProximalSpecimen />,
  },
]
