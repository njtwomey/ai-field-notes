/**
 * `aifn/timeseries`: models of one series observed in time. Dynamics without noise live in `aifn/ode` and friends;
 * the linear-Gaussian filter lives here.
 *
 * - Autocorrelation and AR estimation: `sampleAcf` (with white-noise and Bartlett bands), `samplePacf`,
 *   `levinsonDurbin`, `yuleWalker`, `burg`.
 * - ARMA: `armaRoots`, `isStationary`, `isInvertible`, `psiWeights`, `armaAutocovariance`, `armaAutocorrelation`,
 *   `simulateArma` (never clips; reports `stationary` and `diverged`), `armaResiduals`, `armaLogLikelihood` (exact,
 *   by the Kalman filter), `armaFitter` / `fitArma` (CSS or exact), `forecastArma`.
 * - Transformations: `difference`, `undifference`, `classicalDecomposition`, `stl`.
 * - State space: `simulateStateSpace`, `kalmanFilter` (Joseph form, missing values, singular noise allowed),
 *   `rtsSmoother`, `steadyStateKalman`, `stateSpaceEm` (traceable EM), `extendedKalmanFilter`,
 *   `unscentedKalmanFilter`.
 * - Exponential smoothing: `exponentialSmoothing` (simple, Holt, damped, Holt–Winters additive and multiplicative,
 *   with intervals), `smoothingFitter`.
 * - Volatility: `simulateGarch`, `garchLogLikelihood`, `garchFitter`, `garchForecast`, `garchProperties`.
 *
 * Every fitter is an `Algorithm` (see `aifn/trace`) whose state carries the current parameters; samplers take the
 * stream first.
 */

export {
  burg,
  levinsonDurbin,
  sampleAcf,
  samplePacf,
  yuleWalker,
  type AutoregressiveFit,
  type LevinsonDurbin,
  type SampleAcf,
} from './autoregression'
export {
  armaAutocorrelation,
  armaAutocovariance,
  armaFitter,
  armaLogLikelihood,
  armaResiduals,
  armaRoots,
  fitArma,
  forecastArma,
  isInvertible,
  isStationary,
  psiWeights,
  simulateArma,
  type ArmaFit,
  type ArmaFitOptions,
  type ArmaLikelihood,
  type ArmaSimulation,
  type ArmaSpec,
  type Forecast,
  type LagRoots,
} from './arma'
export { classicalDecomposition, difference, stl, undifference, type Decomposition, type StlOptions } from './decompose'
export {
  kalmanFilter,
  rtsSmoother,
  simulateStateSpace,
  stateSpaceEm,
  steadyStateKalman,
  type EmEstimate,
  type KalmanFilterResult,
  type SmootherResult,
  type StateSpaceEmState,
  type StateSpaceModel,
} from './kalman'
export {
  extendedKalmanFilter,
  unscentedKalmanFilter,
  type NonlinearStateSpaceModel,
  type UnscentedOptions,
} from './nonlinear'
export {
  exponentialSmoothing,
  smoothingFitter,
  type SmoothingResult,
  type SmoothingSpec,
  type SmoothingStructure,
} from './smoothing'
export { garchFitter, garchForecast, garchLogLikelihood, garchProperties, simulateGarch, type GarchSpec } from './garch'
export type { FitState } from './fit'
