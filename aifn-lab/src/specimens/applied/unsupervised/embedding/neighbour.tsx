import type { Specimen } from '../../../../specimen'
import { NeighbourEmbeddingSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/unsupervised/embedding/neighbour',
    title: 't-SNE and UMAP layouts stepping',
    description: 'tsneSteps and umapSteps traced on noisy digit glyphs, scrubbed iteration by iteration.',
    tags: ['tsne', 'umap', 'perplexityCalibration', 'fuzzyGraph', 'trace'],
    render: () => <NeighbourEmbeddingSpecimen />,
  },
]
