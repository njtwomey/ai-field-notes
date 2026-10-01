/**
 * `aifn/learning/calibration`: maps from scores to calibrated probabilities. Today: isotonic regression by pool
 * adjacent violators (`poolAdjacentViolatorsSteps`, the step-through form; `isotonicRegression` runs it, pooling ties
 * in x first). Calibration error metrics are in `aifn/learning/metrics`.
 */

export {
  isotonicRegression,
  poolAdjacentViolatorsSteps,
  type IsotonicFit,
  type IsotonicOptions,
  type PavEvent,
  type PavState,
} from './isotonic'
export { calibrationAlgorithms, calibrationFunctions } from './registry'
