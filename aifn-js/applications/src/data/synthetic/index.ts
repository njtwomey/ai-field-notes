/**
 * `aifn-applied/data/synthetic`: seeded synthetic datasets: points (blobs, moons, circles, spirals, …), regression,
 * sequences, piecewise series with known changepoints, images and recommender interactions; and modifiers (noise,
 * outliers, shifts, missingness, linear maps). Recipes, which replay generators and modifiers by key, are in
 * `aifn-applied/data`.
 */

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
  ADDITIVE_SHAPES,
  additiveData,
  type AdditiveFamily,
  type AdditiveOptions,
  type AdditiveShape,
} from './additive'
export { CURVE1D_CASES, curve1d, type Curve1dCase, type Curve1dOptions } from './curves'
export {
  friedman1,
  linearRegressionData,
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
  arRegimes,
  meanShifts,
  poissonShifts,
  varianceShifts,
  type ArRegimeOptions,
  type MeanShiftOptions,
  type PoissonShiftOptions,
  type SegmentOptions,
  type VarianceShiftOptions,
} from './changepoints'
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
  type TransformOptions,
} from './modifiers'
