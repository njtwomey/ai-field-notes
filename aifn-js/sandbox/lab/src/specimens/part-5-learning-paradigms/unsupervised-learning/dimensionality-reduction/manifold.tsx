import type { Specimen } from '@lab/specimen'
import { ManifoldSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/unsupervised-learning/dimensionality-reduction',
    title: 'Manifold learning on a Swiss roll',
    description: 'PCA, classical and metric MDS, kernel PCA, Isomap, Laplacian eigenmaps and LLE on the same roll.',
    tags: ['isomap', 'laplacianEigenmaps', 'locallyLinearEmbedding', 'classicalMds', 'metricMds', 'kernelPca'],
    render: () => <ManifoldSpecimen />,
  },
]
