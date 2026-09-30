/**
 * `aifn/gp`: Gaussian processes on `aifn/kernels`.
 *
 * ```ts
 * const post = gpPosterior(rbf({ lengthscale: 0.3 }), x, y, { noiseVariance: 0.01 })
 * post.predict(xs) // { mean, variance }
 * post.sample(stream(1), xs, 5).draws // [5, m]
 * fitGp(rbf(), x, y).kernel // type-II maximum likelihood
 * ```
 *
 * - Exact regression: `gpPrior`, `samplePrior`, `gpPosterior` (Cholesky with jitter reported; `predict`, `latent`,
 *   `sample`), `logMarginalLikelihood` (differentiable, split into data fit, complexity and constant),
 *   `logMarginalLikelihoodGradient` (in the hyperparameters and in their logs), `fitGp` (L-BFGS in log space, with
 *   restarts), and the estimator `gaussianProcessRegressor`.
 * - Sparse regression with inducing inputs: `sparseGp` (`vfe` Titsias bound, `fitc`, `dtc`, `sor`),
 *   `sparseLogMarginal` (differentiable in Z), `fitSparseGp` (hyperparameters, noise and inducing inputs).
 * - Classification by the Laplace approximation: `laplaceMode` (Newton's method as a traceable `Algorithm`) and the
 *   estimator `gpClassifier` (logistic or probit likelihood).
 *
 * Posterior and prior draws use fixed standard normals from an `aifn/random` stream, so they move continuously as the
 * data or hyperparameters change.
 */

export {
  fitGp,
  gaussianProcessRegressor,
  gpPosterior,
  gpPrior,
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
  sparseGp,
  sparseLogMarginal,
  type FitSparseGpOptions,
  type SparseGp,
  type SparseGpFit,
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
