import type { Specimen } from '@lab/specimen'
import { PcaSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/unsupervised-learning/dimensionality-reduction',
    title: 'PCA axes on 2-D data',
    description:
      'pca: principal axes, explained variance and the one-component reconstruction of a draggable query point.',
    tags: ['pca', 'explained variance', 'reconstruction'],
    render: () => <PcaSpecimen />,
  },
]
