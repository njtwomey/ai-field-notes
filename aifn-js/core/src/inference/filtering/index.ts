/**
 * `aifn/inference/filtering`: state-space filtering: linear-Gaussian models, simulation, the Kalman filter, the
 * Rauch–Tung–Striebel smoother, the steady-state filter, and the extended and unscented Kalman filters.
 */

export {
  kalmanFilter,
  rtsSmoother,
  simulateStateSpace,
  steadyStateKalman,
  type KalmanFilterResult,
  type SmootherResult,
  type StateSpaceModel,
} from './kalman'
// The working form the filter, smoother and EM (`aifn-applied/timeseries`) share: plain rows of numbers.
export {
  filterArrays,
  modelTensors,
  packFilter,
  parseModel,
  smootherArrays,
  type FilterArrays,
  type Model,
  type SmootherArrays,
} from './kalman'
export {
  extendedKalmanFilter,
  unscentedKalmanFilter,
  type NonlinearStateSpaceModel,
  type UnscentedOptions,
} from './nonlinear'
