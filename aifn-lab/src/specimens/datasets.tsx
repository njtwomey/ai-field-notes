import type { Specimen } from '../specimen'
import {
  GallerySpecimen,
  ImagesSpecimen,
  ManifoldSpecimen,
  RealDataSpecimen,
  RecommendationSpecimen,
  RegressionSpecimen,
  SequencesSpecimen,
} from './_datasets/figures'
import { RecipeSpecimen } from './_datasets/recipe'
import { PairPlotSpecimen, ParallelCoordinatesSpecimen } from './_datasets/views'

export const specimens: Specimen[] = [
  {
    module: 'datasets',
    title: 'Dataset recipes',
    description:
      'A dataset from plain parameters: base generator, prevalence, separation, label noise, outliers, nuisance features and missing values, with the Bayes boundary and error where the truth is known.',
    tags: ['recipe', 'prevalence', 'label noise', 'Bayes error', 'Bayes posterior', 'missing values', 'outliers'],
    render: () => <RecipeSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Gallery of 2-D datasets',
    description: 'Moons, circles, rings, spirals, XOR, checkerboard and blobs, each drawn from a named stream.',
    tags: ['moons', 'spirals', 'XOR', 'blobs', 'stream', 'DatasetView'],
    render: () => <GallerySpecimen />,
  },
  {
    module: 'datasets',
    title: 'Pair plot',
    description:
      'Every pair of features in one grid: per-class histograms or KDE curves on the diagonal, scatters below, correlations above, with linked brushing and hover across panels.',
    tags: ['pair plot', 'scatter matrix', 'brushing', 'correlation', 'KDE', 'PairPlotView'],
    render: () => <PairPlotSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Parallel coordinates',
    description:
      'One axis per feature and one line per row, coloured by class: brush axes to filter, drag to reorder, flip, and compare with Andrews curves.',
    tags: ['parallel coordinates', 'Andrews curves', 'brushing', 'high-dimensional', 'ParallelCoordinatesView'],
    render: () => <ParallelCoordinatesSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Manifolds in three dimensions',
    description: 'The Swiss roll and S-curve, coloured by the coordinate along the sheet.',
    tags: ['Swiss roll', 'S-curve', 'manifold learning'],
    render: () => <ManifoldSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Real datasets',
    description: "Iris, Old Faithful and Anscombe's quartet, embedded.",
    tags: ['Iris', 'Old Faithful', 'Anscombe'],
    render: () => <RealDataSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Regression functions',
    description: 'Named test functions with homoscedastic or growing noise.',
    tags: ['regression', 'heteroscedastic', 'Doppler'],
    render: () => <RegressionSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Sequences and time series',
    description: 'The occasionally dishonest casino, an AR(2) series and a seasonal series with trend.',
    tags: ['HMM', 'casino', 'AR', 'seasonal'],
    render: () => <SequencesSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Recommendation data',
    description: 'Zipf popularity and click logs under the position-based model.',
    tags: ['Zipf', 'popularity', 'click model', 'position bias'],
    render: () => <RecommendationSpecimen />,
  },
  {
    module: 'datasets',
    title: 'Test images',
    description: 'The shapes image, checkerboard, gradient and digit glyphs.',
    tags: ['images', 'digits'],
    render: () => <ImagesSpecimen />,
  },
]
