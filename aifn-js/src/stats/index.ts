/**
 * Statistics on plain numeric arrays (`ArrayLike<number>`) and `aifn/tensor` tensors: descriptive statistics,
 * quantiles, ranks and rank correlations, histograms with an explicit bin rule, the empirical CDF, Gaussian KDE,
 * running moments, autocovariance and cross-correlation (by FFT for long sequences: the one autocorrelation in aifn),
 * and resampling (bootstrap, permutation tests, and Kish's importance ESS, the one ESS of weights in aifn).
 *
 * Conventions: variances are population (÷ n) unless `{ sample: true }`; ranks average ties; histograms count
 * values equal to the last edge in the last bin and report dropped values; randomness comes from an `aifn/random`
 * stream, first argument. Reductions take `{ axis, keepDims }` to reduce a tensor along one axis (a tensor result);
 * without an axis a tensor of any rank reduces over every element. Sequence and pairwise functions need rank-1
 * tensors. Array results are rank-1 tensors (float64, or int32 for indices and lags), except `histogram`, `kde` and
 * `zScores`, which still return typed arrays for now.
 */
export {
  correlation,
  covariance,
  extent,
  kurtosis,
  max,
  mean,
  min,
  mode,
  range,
  skewness,
  standardDeviation,
  standardise,
  sum,
  variance,
  weightedMean,
  weightedVariance,
  zScores,
  type Along,
  type KurtosisOptions,
  type SampleOption,
  type Whole,
} from './descriptive'
export {
  interquartileRange,
  median,
  quantile,
  quantileMethods,
  sorted,
  type QuantileMethod,
  type QuantileOptions,
} from './quantile'
export { argsort, kendallTau, ranks, spearman, type TiePolicy } from './ranks'
export { ecdf, ecdfAt, histogram, kde, kdeBandwidth, type BandwidthRule, type BinRule, type Histogram } from './density'
export {
  autocorrelation,
  autocovariance,
  crossCorrelation,
  crossCovariance,
  emptyMoments,
  momentsMerge,
  momentsPush,
  momentsVariance,
  runningMean,
  runningVariance,
  type LagOptions,
  type Moments,
} from './sequence'
export {
  bootstrap,
  bootstrapInterval,
  importanceEffectiveSampleSize,
  permutationTest,
  resampleIndices,
  shuffled,
  type Bootstrap,
  type PermutationTest,
} from './resampling'
export type { AxisOption, Data } from './input'
