/** Types shared by every initial-value solver in `aifn/ode`. */

import type { Tensor, Value, Vector } from 'aifn/tensor'
import type { MatrixLike, VectorLike } from './vector'

/**
 * The right-hand side of x′ = f(t, x): time t and state x (a rank-1 tensor of length n) to the derivative (length
 * n). Solvers that need the Jacobian ∂f/∂x by automatic differentiation call f with a traced x, so f must then be
 * written with `aifn/tensor` primitives (`get`, `stack`, `mul`, `sin`, …) rather than by reading `x.data`. The result
 * may be a tensor, a plain array or (from primitives) a `Value`; outside autodiff it must be a length-n vector.
 */
export type Rhs = (t: number, x: Tensor) => VectorLike | Value

/** How an implicit solver obtains ∂f/∂x: by `aifn/autodiff` (default), by forward differences, or a given function. */
export type JacobianOption = 'autodiff' | 'finite-difference' | ((t: number, x: Tensor) => MatrixLike)

/** The initial value passed to `init`: x(t₀) = x₀. */
export type InitialValue = { x0: VectorLike; t0?: number }

/** Fields every solver state carries. */
export type OdeState = {
  /** The current time. */
  t: number
  /** The current state x(t) (length n). */
  x: Vector
  /** Accepted steps so far. */
  step: number
  /** The size of the last accepted step (0 at t₀). */
  h: number
  /** The estimated local error of the last step in the solver's error norm, or NaN for methods without an estimate. */
  error: number
  /** Evaluations of f so far. */
  evaluations: number
  /** Evaluations of the Jacobian ∂f/∂x so far (implicit methods). */
  jacobianEvaluations: number
  /** Step attempts rejected by error control so far (adaptive methods). */
  rejected: number
  /** True when the state stopped being finite or a step failed (see `failure`). */
  diverged: boolean
  /** Why the solver cannot continue, or null: `'not finite'`, `'newton failed'`, `'step size underflow'`. */
  failure: string | null
}

/** Options common to the fixed-step solvers. */
export type FixedStepOptions = {
  /** The step size (negative integrates backwards in time). */
  h: number
  /** Stop on reaching this time; the last step is shortened to land on it exactly. Default: never (run n steps). */
  tEnd?: number
}
