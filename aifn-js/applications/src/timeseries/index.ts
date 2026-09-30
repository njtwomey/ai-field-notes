/**
 * `aifn-applied/timeseries`: models of one series observed in time: ARMA fitting and forecasting, GARCH, exponential
 * smoothing, classical and STL decomposition, and EM for linear-Gaussian state-space models.
 */

export {
  armaAutocorrelation,
  armaAutocovariance,
  armaFitSteps,
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
export {
  classicalDecomposition,
  difference,
  stl,
  undifference,
  type SeasonalDecomposition,
  type StlOptions,
} from './decompose'
export {
  exponentialSmoothing,
  exponentialSmoothingFitSteps,
  type SmoothingResult,
  type SmoothingSpec,
  type SmoothingStructure,
} from './smoothing'
export {
  fitGarch,
  garchFitSteps,
  garchForecast,
  garchLogLikelihood,
  garchProperties,
  simulateGarch,
  type GarchSpec,
} from './garch'
export type { FitState } from './fit'
export { stateSpaceEm, type EmEstimate, type StateSpaceEmState } from './state-space'
