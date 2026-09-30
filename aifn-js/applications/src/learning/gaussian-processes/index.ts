/**
 * `aifn-applied/learning/gaussian-processes`: Gaussian processes on the kernels of `aifn/learning/kernels`: regression
 * with the marginal likelihood and fitting, sparse approximations, and classification.
 */

export {
  fitGp,
  gaussianProcessRegressor,
  gpPosterior,
  gpPrior,
  kernelLogVector,
  logMarginalLikelihood,
  logMarginalLikelihoodGradient,
  samplePrior,
  type Draws,
  type FitGpOptions,
  type GaussianProcessRegressionModel,
  type GaussianProcessRegressorParams,
  type GpFit,
  type GpOptions,
  type GpPosterior,
  type LogMarginal,
  type LogMarginalGradient,
  type Prediction,
} from './regression'
export {
  fitSparseGp,
  sparseGaussianProcessRegressor,
  sparseGp,
  sparseLogMarginal,
  type FitSparseGpOptions,
  type SparseGp,
  type SparseGpFit,
  type SparseGaussianProcessRegressionModel,
  type SparseGaussianProcessRegressorParams,
  type SparseGpOptions,
  type SparseMethod,
} from './sparse'
export {
  gpClassifier,
  laplaceMode,
  type ClassificationLikelihood,
  type GpClassifierModel,
  type GpClassifierParams,
  type LaplaceProblem,
  type LaplaceState,
} from './classification'
