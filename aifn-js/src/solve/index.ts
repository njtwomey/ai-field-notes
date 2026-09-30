/**
 * `aifn/solve`: roots of nonlinear equations, every iterative method a traceable `Algorithm` (see `aifn/trace`) whose
 * state records the estimate, the bracket or Jacobian, and the kind of step taken; failure is reported in `failure`.
 *
 * - Scalar equations f(x) = 0: `bisection`, `regulaFalsi` (Illinois), `brent`, `secant`, `newtonRoot` (optionally
 *   damped); `findRoot` runs Brent on a bracket.
 * - Systems F(x) = 0: `newtonSystem` (optionally damped), `broyden`, `fixedPoint` (with contraction and error-bound
 *   diagnostics), `continuation` along a homotopy (`newtonHomotopy`); `solveSystem` runs one by name.
 * - Polynomials: `polynomialRoots` (companion-matrix eigenvalues, as numpy's `roots`), `polynomialValue`.
 */

export {
  bisection,
  brent,
  newtonRoot,
  regulaFalsi,
  secant,
  type BracketOptions,
  type BracketState,
  type BrentState,
  type NewtonRootState,
  type RegulaFalsiState,
  type RootState,
  type RootTolerance,
  type ScalarFunction,
  type ScalarWithDerivative,
  type SecantState,
} from './scalar'
export {
  broyden,
  continuation,
  fixedPoint,
  newtonHomotopy,
  newtonSystem,
  type BroydenState,
  type ContinuationOptions,
  type ContinuationState,
  type FixedPointState,
  type Homotopy,
  type NewtonSystemState,
  type SystemFunction,
  type SystemState,
  type SystemTolerance,
  type SystemWithJacobian,
} from './systems'
export { polynomialRoots, polynomialValue, type PolynomialRoots } from './polynomial'
export { findRoot, solveSystem, type RootResult, type SystemResult } from './convenience'
export type { MatrixLike, VectorLike } from './vector'
