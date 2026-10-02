import type { Specimen } from '../../specimen'
import { RepresentationsShowcase } from './_representations/showcase'

export const specimens: Specimen[] = [
  {
    module: 'text/representations',
    title: 'Showcase: classical word representations, from one-hot to SVD',
    description:
      'No learning, only counting and linear algebra: one corpus as one-hot vectors, a bag of words, n-grams, shingles, a TF-IDF term–document matrix, a PPMI co-occurrence matrix or random index vectors; the matrix, its scree, and the words mapped at SVD rank k with cosine neighbours; shingle sets, MinHash and LSH banding; analogies by vector offset.',
    tags: [
      'showcase',
      'one-hot',
      'bag of words',
      'TF-IDF',
      'co-occurrence',
      'PPMI',
      'LSA',
      'SVD',
      'random indexing',
      'shingles',
      'MinHash',
      'LSH',
      'analogies',
      'distributional semantics',
    ],
    render: () => <RepresentationsShowcase />,
  },
]
