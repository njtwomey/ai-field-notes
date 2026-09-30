import type { Matrix, Tensor, Vector } from 'aifn/tensor'

/** A vector argument: a rank-1 tensor (any strides) or a plain array of numbers. Always copied on the way in. */
export type VectorLike = Tensor | ArrayLike<number>

/** A matrix argument: a rank-2 tensor or rows of numbers. Always copied on the way in. */
export type MatrixLike = Tensor | readonly ArrayLike<number>[]

/** What a differentiable objective returns at a point: its value and its gradient (same length as the point). */
export type Evaluation = { value: number; grad: VectorLike }

/**
 * A differentiable objective f: ℝⁿ → ℝ supplying its own gradient, `(x) => ({ value, grad })`. The point `x` is a
 * float64 vector of length n and must not be mutated.
 */
export type Objective = (x: Vector) => Evaluation

/** A function to minimise without derivatives: returns f(x), or an `Evaluation` whose gradient is ignored. */
export type ValueFunction = (x: Vector) => number | { value: number }

/** The Hessian ∇²f(x) as an n×n matrix (symmetric; only the values matter, not the strides). */
export type Hessian = (x: Vector) => MatrixLike

/** A step-size schedule: the step size (learning rate) used on step t = 0, 1, 2, … */
export type Schedule = (t: number) => number

/**
 * Fields every optimiser state carries. `x` is the current iterate and `value` is f(x); `evaluations` counts calls of
 * the objective so far (including the initial one). `converged` is set when the method's stopping test passes and
 * `diverged` when the value or iterate is not finite or exceeds the divergence threshold; either stops the runners.
 */
export type IterateState = {
  /** Steps taken (0 in the initial state). */
  t: number
  x: Vector
  value: number
  evaluations: number
  converged: boolean
  diverged: boolean
}

/** Options shared by the gradient-based methods. */
export type StoppingOptions = {
  /** Stop (converged) when ‖∇f(x)‖₂ ≤ `tolerance`. Default 1e-6. */
  tolerance?: number
  /** Flag divergence when |f(x)| exceeds this (or anything is not finite). Default 1e15. */
  divergeAbove?: number
}

export type { Matrix, Vector }
