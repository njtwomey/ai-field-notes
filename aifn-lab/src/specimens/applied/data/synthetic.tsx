import type { Specimen } from '../../../specimen'
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
    module: 'applied/data/synthetic',
    title: 'Dataset recipes',
    description:
      'A dataset from plain parameters: base generator, prevalence, separation, label noise, outliers, nuisance features and missing values, with the Bayes boundary and error where the truth is known.',
    tags: ['recipe', 'prevalence', 'label noise', 'Bayes error', 'Bayes posterior', 'missing values', 'outliers'],
    render: () => <RecipeSpecimen />,
  },
  {
    module: 'applied/data/synthetic',
    title: 'Gallery of 2-D datasets',
    description: 'Moons, circles, rings, spirals, XOR, checkerboard and blobs, each drawn from a named stream.',
    tags: ['moons', 'spirals', 'XOR', 'blobs', 'stream', 'DatasetView'],
    render: () => <GallerySpecimen />,
  },
  {
    module: 'applied/data/synthetic',
    title: 'Manifolds in three dimensions',
    description: 'The Swiss roll and S-curve, coloured by the coordinate along the sheet.',
    tags: ['Swiss roll', 'S-curve', 'manifold learning'],
    render: () => <ManifoldSpecimen />,
  },
  {
    module: 'applied/data/synthetic',
    title: 'Regression functions',
    description: 'Named test functions with homoscedastic or growing noise.',
    tags: ['regression', 'heteroscedastic', 'Doppler'],
    render: () => <RegressionSpecimen />,
  },
  {
    module: 'applied/data/synthetic',
    title: 'Sequences and time series',
    description: 'The occasionally dishonest casino, an AR(2) series and a seasonal series with trend.',
    tags: ['HMM', 'casino', 'AR', 'seasonal'],
    render: () => <SequencesSpecimen />,
  },
  {
    module: 'applied/data/synthetic',
    title: 'Recommendation data',
    description: 'Zipf popularity and click logs under the position-based model.',
    tags: ['Zipf', 'popularity', 'click model', 'position bias'],
    render: () => <RecommendationSpecimen />,
  },
  {
    module: 'applied/data/synthetic',
    title: 'Test images',
    description: 'The shapes image, checkerboard, gradient and digit glyphs.',
    tags: ['images', 'digits'],
    render: () => <ImagesSpecimen />,
  },
]
