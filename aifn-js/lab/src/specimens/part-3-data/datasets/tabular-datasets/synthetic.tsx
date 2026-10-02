import type { Specimen } from '@lab/specimen'
import {
  GallerySpecimen,
  ImagesSpecimen,
  ManifoldSpecimen,
  RecommendationSpecimen,
  RegressionSpecimen,
  SequencesSpecimen,
} from './_shared/figures'
import { RecipeSpecimen } from './_shared/recipe'

export const specimens: Specimen[] = [
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Dataset recipes',
    description:
      'A dataset from plain parameters: base generator, prevalence, separation, label noise, outliers, nuisance features and missing values, with the Bayes boundary and error where the truth is known.',
    tags: ['recipe', 'prevalence', 'label noise', 'Bayes error', 'Bayes posterior', 'missing values', 'outliers'],
    render: () => <RecipeSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Gallery of 2-D datasets',
    description: 'Moons, circles, rings, spirals, XOR, checkerboard and blobs, each drawn from a named stream.',
    tags: ['moons', 'spirals', 'XOR', 'blobs', 'stream', 'DatasetView'],
    render: () => <GallerySpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Manifolds in three dimensions',
    description: 'The Swiss roll and S-curve, coloured by the coordinate along the sheet.',
    tags: ['Swiss roll', 'S-curve', 'manifold learning'],
    render: () => <ManifoldSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Regression functions',
    description: 'Named test functions with homoscedastic or growing noise.',
    tags: ['regression', 'heteroscedastic', 'Doppler'],
    render: () => <RegressionSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Sequences and time series',
    description: 'The occasionally dishonest casino, an AR(2) series and a seasonal series with trend.',
    tags: ['HMM', 'casino', 'AR', 'seasonal'],
    render: () => <SequencesSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Recommendation data',
    description: 'Zipf popularity and click logs under the position-based model.',
    tags: ['Zipf', 'popularity', 'click model', 'position bias'],
    render: () => <RecommendationSpecimen />,
  },
  {
    module: 'part-3-data/datasets/tabular-datasets',
    title: 'Test images',
    description: 'The shapes image, checkerboard, gradient and digit glyphs.',
    tags: ['images', 'digits'],
    render: () => <ImagesSpecimen />,
  },
]
