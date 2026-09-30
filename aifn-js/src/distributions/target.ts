/**
 * The protocol for an unnormalised log-density: what samplers (`aifn/mcmc`), variational inference (`aifn/vi`) and
 * other inference code need from a target distribution. It sits here, below all of them, so that none imports another
 * for the type.
 */

import type { Value, Vector, VectorLike } from 'aifn/tensor'

/**
 * A target distribution known up to a constant: `logDensity(θ)` is log π(θ) + const for θ a float64 vector of length
 * `dim` (−Infinity outside the support). `grad(θ)` is ∇ log π(θ); when it is omitted, code that needs it
 * differentiates `logDensity` with `aifn/autodiff`, which then receives a traced vector and must be written with aifn
 * primitives (`sum`, `mul`, `get`, …) rather than `toFlat`.
 */
export type Target = {
  dim: number
  logDensity: (theta: Vector) => number | Value
  grad?: (theta: Vector) => VectorLike
  name?: string
}
