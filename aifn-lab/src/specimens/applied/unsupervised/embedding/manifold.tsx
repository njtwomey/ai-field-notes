import type { Specimen } from '../../../../specimen'
import { ManifoldSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/unsupervised/embedding/manifold',
    title: 'Manifold learning on a Swiss roll',
    description: 'PCA, classical and metric MDS, kernel PCA, Isomap, Laplacian eigenmaps and LLE on the same roll.',
    tags: ['isomap', 'laplacianEigenmaps', 'locallyLinearEmbedding', 'classicalMds', 'metricMds', 'kernelPca'],
    render: () => <ManifoldSpecimen />,
  },
]
