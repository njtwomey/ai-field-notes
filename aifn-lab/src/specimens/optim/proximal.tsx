import type { Specimen } from '../../specimen'
import { ProximalSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'optim/proximal',
    title: 'ISTA and FISTA',
    description: 'Proximal gradient with soft thresholding on a lasso, with and without Nesterov acceleration.',
    tags: ['proximal', 'lasso', 'L1', 'acceleration'],
    render: () => <ProximalSpecimen />,
  },
]
