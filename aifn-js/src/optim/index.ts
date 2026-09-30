/**
 * `aifn/optim`: traceable optimisers. Every method is an `Algorithm` (see `aifn/trace`) built by a factory from the
 * objective and its options, `trace(adam(f, { lr: 0.05 }), { x0 }, 500)`, whose states expose the method's internals
 * (iterate, value, gradient, step size, direction, line-search trials, curvature pairs, simplex, population…);
 * `minimize(f, x0, { method })` runs one by name.
 *
 * - Objectives supply their own gradients, `(x) => ({ value, grad })`; derivative-free methods accept `(x) => number`.
 * - First order: `gradientDescent` (fixed, scheduled or line-searched step), `momentum`, `nesterov`, `adagrad`,
 *   `rmsprop`, `adam`, `adamw`; schedules `inverseTimeDecay`, `exponentialDecay`, `inverseSqrtDecay`.
 * - Line searches: `backtracking` (Armijo), `strongWolfe`, with every trial recorded.
 * - Second order and quasi-Newton: `newton` (damped, Hessian supplied), `trustRegion` (dogleg), `bfgs`, `lbfgs`,
 *   `conjugateGradient` (Fletcher–Reeves, Polak–Ribière); `linearConjugateGradient` and `solveConjugateGradient` for
 *   symmetric positive definite systems.
 * - Least squares: `gaussNewton`, `levenbergMarquardt`, `leastSquares`.
 * - Composite and constrained: `proximalGradient`, `ista`, `fista`, `projectedGradient`; prox operators `proxL1`,
 *   `proxL2`, `proxSquaredL2`, `proxBox`, `proxNonnegative`, `proxZero`; projections `projectBox`,
 *   `projectNonnegative`, `projectBall`, `projectSimplex`.
 * - Derivative free: `nelderMead`, `coordinateDescent` (gradient, Newton or exact coordinate steps),
 *   `simulatedAnnealing`, `cmaEs` (both draw from the stream given to `init`).
 * - One variable: `minimizeScalar` (Brent's method or golden-section search, on a bracket or bounds).
 *
 * Test surfaces (`rosenbrock`, `himmelblau`, …) are data and live in `aifn/datasets`.
 */

export type {
  Evaluation,
  Hessian,
  IterateState,
  MatrixLike,
  Objective,
  Schedule,
  StoppingOptions,
  ValueFunction,
  VectorLike,
} from './types'
export {
  backtracking,
  strongWolfe,
  type BacktrackingOptions,
  type LineSearchResult,
  type LineSearchTrial,
  type StrongWolfeOptions,
} from './lineSearch'
export {
  adagrad,
  adam,
  adamw,
  exponentialDecay,
  gradientDescent,
  inverseSqrtDecay,
  inverseTimeDecay,
  momentum,
  nesterov,
  rmsprop,
  type AdamOptions,
  type AdaptiveOptions,
  type FirstOrderOptions,
  type FirstOrderState,
  type GradientDescentOptions,
  type MomentumOptions,
  type RmspropOptions,
  type StartOptions,
} from './firstOrder'
export {
  newton,
  trustRegion,
  type DoglegKind,
  type NewtonOptions,
  type NewtonState,
  type TrustRegionOptions,
  type TrustRegionState,
} from './newton'
export {
  bfgs,
  lbfgs,
  type BfgsState,
  type CurvaturePair,
  type LbfgsOptions,
  type LbfgsState,
  type QuasiNewtonOptions,
} from './quasiNewton'
export {
  conjugateGradient,
  linearConjugateGradient,
  solveConjugateGradient,
  type ConjugateGradientOptions,
  type ConjugateGradientSolution,
  type ConjugateGradientState,
  type ConjugateGradientVariant,
  type LinearConjugateGradientOptions,
  type LinearConjugateGradientState,
  type LinearOperator,
} from './conjugateGradient'
export {
  gaussNewton,
  leastSquares,
  levenbergMarquardt,
  type GaussNewtonOptions,
  type GaussNewtonState,
  type LeastSquaresResult,
  type LeastSquaresState,
  type LevenbergMarquardtOptions,
  type LevenbergMarquardtState,
  type ResidualFunction,
} from './leastSquares'
export {
  nelderMead,
  type NelderMeadOperation,
  type NelderMeadOptions,
  type NelderMeadState,
  type NelderMeadTrial,
} from './nelderMead'
export {
  coordinateDescent,
  type CoordinateDescentOptions,
  type CoordinateDescentState,
  type CoordinateRule,
} from './coordinateDescent'
export {
  fista,
  ista,
  projectBall,
  projectBox,
  projectNonnegative,
  projectSimplex,
  projectedGradient,
  proximalGradient,
  proxBox,
  proxL1,
  proxL2,
  proxNonnegative,
  proxSquaredL2,
  proxZero,
  type Bound,
  type Prox,
  type ProximalGradientOptions,
  type ProximalGradientState,
} from './proximal'
export {
  cmaEs,
  simulatedAnnealing,
  type CmaEsOptions,
  type CmaEsState,
  type SimulatedAnnealingOptions,
  type SimulatedAnnealingState,
} from './stochastic'
export { minimizeScalar, type MinimizeScalarOptions, type MinimizeScalarResult } from './scalar'
export { minimize, type Method, type MethodOptions, type MinimizeResult } from './minimize'
