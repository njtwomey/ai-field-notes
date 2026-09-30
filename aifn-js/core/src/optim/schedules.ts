/**
 * Step-size schedules t ↦ η_t (the family's shared layer): for the first-order methods and update rules, and for
 * anything else that decays a step size over steps (stochastic-gradient Langevin dynamics, annealing). Robbins &
 * Monro (1951), "A stochastic approximation method": Σ η_t = ∞ and Σ η_t² < ∞ hold for `inverseTimeDecay`.
 */

import type { Scalar, Schedule } from 'aifn/foundation/contracts'

/** The schedule η_t = η₀ / (1 + kt). */
export const inverseTimeDecay =
  (initial: Scalar, k: Scalar): Schedule =>
  (t) =>
    initial / (1 + k * t)

/** The schedule η_t = η₀γᵗ. */
export const exponentialDecay =
  (initial: Scalar, gamma: Scalar): Schedule =>
  (t) =>
    initial * gamma ** t

/** The schedule η_t = η₀ / √(t + 1). */
export const inverseSqrtDecay =
  (initial: Scalar): Schedule =>
  (t) =>
    initial / Math.sqrt(t + 1)
