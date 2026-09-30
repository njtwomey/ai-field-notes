/**
 * `aifn-applied/data`: datasets: seeded synthetic generators and recipes, small embedded real datasets, test
 * objectives, target densities, environments for bandits and reinforcement learning, and test signals. The shared
 * layer holds the `Dataset` shape, ground truth and sizes.
 */

export {
  classificationTruth,
  regressionTruth,
  twoGaussianBayesError,
  type ClassModel,
  type ClassificationTruth,
  type LabelOp,
  type Reference,
  type RegressionModel,
  type RegressionTruth,
  type Row,
  type Truth,
} from './truth'
export { classCounts, type ClassSizeOptions, type ClassSizes } from './sizes'
export { type Dataset, type DatasetMeta, type Recipe, type RecipeStep } from './types'
export { blobs, moons } from './synthetic'
export { gridworld } from './environments'
