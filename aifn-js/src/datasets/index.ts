/**
 * `aifn/datasets`: seeded generators and small embedded real datasets, with fixed conventions. Every generator takes a
 * stream first (`aifn/random`), so figures, tests and replicates agree, and returns tensors plus metadata (feature and
 * label names, a caption-ready description, a source).
 *
 * - Point clouds (`Dataset`: x n × d, y int32 labels): `blobs`, `moons`, `circles`, `rings`, `spirals`, `xor`,
 *   `checkerboard`, `gaussians`, `anisotropicBlobs`; manifolds with a coordinate `t`: `swissRoll`, `sCurve`;
 *   `shuffleDataset`.
 * - Regression (`y` real, `f` noise-free): `regression1d` (named functions), `linearRegression`, `friedman1`.
 * - Sequences: `hmmSample`, `casino` (the occasionally dishonest casino); time series `arSeries`, `seasonalSeries`,
 *   `randomWalk`.
 * - Recommendation: `zipfWeights`, `zipfCatalogue`, `ratings` (low-rank ratings matrix with train/test entries),
 *   `clickLog` (position-based and cascade click models).
 * - Images and patterns: `checkerboardImage`, `gradientImage`, `shapesImage`, `digitGlyphs`, `digits`,
 *   `barsAndStripes`.
 * - Real data: `iris`, `oldFaithful`, `anscombe`.
 * - Optimisation test surfaces with gradients, Hessians and known minima: `rosenbrock`, `himmelblau`, `beale`,
 *   `quadraticBowl`, `rastrigin`.
 *
 * - Parametric control: every labelled generator takes `ClassSizeOptions` (a total `n` with `prevalence` or
 *   `classWeights`, or per-class counts; exact counts by largest remainder, `classCounts`). Modifiers change one thing
 *   and record themselves in `meta.recipe`: `withLabelNoise` (+ `flippedMask`), `withPrevalence`, `withLabelShift`,
 *   `withOutliers`, `withNuisanceFeatures`, `withTransform` (+ `rotation2d`, `shear2d`), `withMissing`,
 *   `withCovariateShift` and `split`. `recipe` builds a dataset from a JSON-serialisable `DatasetRecipe`;
 *   `describeRecipe`, `encodeRecipe` and `decodeRecipe` caption it and put it in a URL.
 * - Known truth: `meta.truth` holds the generating process where it has a closed form. Classification
 *   (`ClassificationTruth`): the Bayes posterior `posterior(x)`, `probability(x)` = P(y = 1 | x), the Bayes-optimal
 *   `score(x)` (log odds) for ROC and PR curves, and `bayesError` (closed form for two Gaussians with a shared
 *   covariance, Monte Carlo otherwise). Regression (`RegressionTruth`): `mean(x)`, `noiseSd` and `bayesRisk`. Label
 *   noise, prevalence changes, outliers, nuisance features, invertible maps and covariate shift update it.
 *
 * Class overlap, per generator: `blobs` and `gaussians` take `separation` (neighbouring centres in blob sds; d′ in
 * Mahalanobis units, Bayes error Φ(−d′/2) for two equal classes); `moons` (noise vs a gap of 1), `circles` (noise vs
 * 1 − factor), `rings` (radial noise vs the gap between radii), `spirals` (noise vs 1/(arms · turns)) and Gaussian
 * `xor` (sd vs 2) blur with their noise; uniform `xor` and `checkerboard` are separable (Bayes error 0) and blur only
 * with label noise.
 *
 * Conventions: points are grouped by class in label order; generators draw from named substreams (`points`, `noise`,
 * …) so that changing one part of a recipe (say the noise) leaves the other draws unchanged. Class 1 is the positive
 * class of a binary problem.
 */

export type { Dataset, DatasetMeta, RecipeStep } from './types'
export { classCounts, type ClassSizeOptions, type ClassSizes } from './sizes'
export type { ClassificationTruth, ClassModel, LabelOp, Reference, RegressionTruth, Row, Truth } from './truth'
export { twoGaussianBayesError } from './truth'
export {
  flippedMask,
  rotation2d,
  shear2d,
  split,
  symmetricNoise,
  withCovariateShift,
  withLabelNoise,
  withLabelShift,
  withMissing,
  withNuisanceFeatures,
  withOutliers,
  withPrevalence,
  withTransform,
  type CovariateShiftOptions,
  type LabelNoiseOptions,
  type MissingMechanism,
  type MissingOptions,
  type NuisanceOptions,
  type OutlierOptions,
  type PrevalenceOptions,
  type SplitOptions,
} from './modifiers'
export {
  RECIPE_BASES,
  decodeRecipe,
  describeRecipe,
  encodeRecipe,
  isClassificationBase,
  parseRecipe,
  recipe,
  type ClassificationBase,
  type DatasetRecipe,
  type Json,
  type RecipeBase,
  type RegressionBase,
} from './recipe'
export {
  anisotropicBlobs,
  blobs,
  checkerboard,
  circles,
  gaussians,
  moons,
  rings,
  sCurve,
  shuffleDataset,
  spirals,
  swissRoll,
  xor,
  type AnisotropicOptions,
  type BlobsOptions,
  type CheckerboardOptions,
  type CirclesOptions,
  type GaussiansOptions,
  type ManifoldOptions,
  type MoonsOptions,
  type RingsOptions,
  type SpiralsOptions,
  type XorOptions,
} from './points'
export {
  friedman1,
  linearRegression,
  regression1d,
  type LinearRegressionOptions,
  type Regression1dOptions,
  type RegressionFunction,
} from './regression'
export {
  arSeries,
  casino,
  hmmSample,
  randomWalk,
  seasonalSeries,
  type ArOptions,
  type CasinoOptions,
  type DiscreteHmm,
  type HmmSample,
  type SeasonalOptions,
  type TimeSeries,
} from './sequences'
export {
  clickLog,
  ratings,
  zipfCatalogue,
  zipfWeights,
  type ClickLog,
  type ClickLogOptions,
  type Ratings,
  type RatingsOptions,
  type ZipfCatalogue,
} from './recsys'
export { barsAndStripes, checkerboardImage, digitGlyphs, digits, gradientImage, shapesImage } from './images'
export { anscombe, iris, oldFaithful } from './real'
export { beale, himmelblau, quadraticBowl, rastrigin, rosenbrock, type TestFunction } from './surfaces'
