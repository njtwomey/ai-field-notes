/**
 * `aifn/optim`: optimisation, as torch.optim, optax and scipy.optimize. The shared layer holds the objective and option types, the
 * start point, the standard stopping and runner options, and step-size schedules; the first-order update rules work
 * on parameter pytrees for training loops. Children: line-search, first-order, second-order, proximal, derivative-free,
 * programming, minimize.
 */

export {
  type Evaluation,
  type Hessian,
  type IterateState,
  type MatrixLike,
  type ObjectiveFn,
  type Schedule,
  type StoppingOptions,
  type ValueFunction,
  type VectorLike,
} from 'aifn/foundation/contracts'
export { objectiveFn, type RunOptions, type StartOptions } from './options'
export { exponentialDecay, inverseSqrtDecay, inverseTimeDecay } from './schedules'
export { minimize } from './minimize'
export {
  adam,
  adamRule,
  applyUpdates,
  gradientDescent,
  momentum,
  sgdRule,
  type StepSize,
  type UpdateRule,
} from './first-order'
export { newton, lbfgs } from './second-order'
export { nelderMead, minimizeScalar } from './derivative-free'
export { linprog } from './programming'
