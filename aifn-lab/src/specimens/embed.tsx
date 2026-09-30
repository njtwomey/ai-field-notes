import type { Specimen } from '../specimen'
import { ManifoldSpecimen, NeighbourEmbeddingSpecimen, PcaSpecimen } from './_embed/figures'

export const specimens: Specimen[] = [
  {
    module: 'embed',
    title: 'PCA axes on 2-D data',
    description:
      'pca: principal axes, explained variance and the one-component reconstruction of a draggable query point.',
    tags: ['pca', 'explained variance', 'reconstruction'],
    render: () => <PcaSpecimen />,
  },
  {
    module: 'embed',
    title: 'Manifold learning on a Swiss roll',
    description: 'PCA, classical and metric MDS, kernel PCA, Isomap, Laplacian eigenmaps and LLE on the same roll.',
    tags: ['isomap', 'laplacianEigenmaps', 'locallyLinearEmbedding', 'classicalMds', 'metricMds', 'kernelPca'],
    render: () => <ManifoldSpecimen />,
  },
  {
    module: 'embed',
    title: 't-SNE and UMAP layouts stepping',
    description: 'tsneSteps and umapSteps traced on noisy digit glyphs, scrubbed iteration by iteration.',
    tags: ['tsne', 'umap', 'perplexityCalibration', 'fuzzyGraph', 'trace'],
    render: () => <NeighbourEmbeddingSpecimen />,
  },
]
