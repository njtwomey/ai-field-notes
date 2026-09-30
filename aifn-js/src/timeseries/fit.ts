/**
 * Private: maximum-likelihood and least-squares fits as traceable algorithms. Each fitter maps unconstrained
 * coordinates u to valid parameters (a logistic map to (0, 1), partial autocorrelations to a stationary AR polynomial,
 * …) and runs Nelder–Mead from `aifn/optim` on the objective in u, so every step of a fit can be traced.
 */

import { nelderMead, type NelderMeadState } from 'aifn/optim'
import { toFlat } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'

/** The state of a fitter: the optimiser's state and the parameters its best vertex decodes to. */
export type FitState<P> = {
  /** Steps taken. */
  t: number
  /** The parameters at the best vertex. */
  params: P
  /** The objective at the best vertex (a negative log-likelihood or a sum of squares). */
  objective: number
  evaluations: number
  converged: boolean
  diverged: boolean
  /** The Nelder–Mead state in the unconstrained coordinates. */
  optimiser: NelderMeadState
}

/** Objective values for invalid parameters: finite, so the simplex simply moves away from them. */
export const INVALID = 1e300

export function simplexFit<P>(
  name: string,
  objective: (u: number[]) => number,
  decode: (u: number[]) => P,
  u0: number[],
  tolerance = 1e-9,
): Algorithm<Record<string, never>, FitState<P>> {
  const f = (u: number[]) => {
    const v = objective(u)
    return Number.isFinite(v) ? v : INVALID
  }
  const nm = nelderMead((x) => f(toFlat(x)), { xTolerance: tolerance, fTolerance: tolerance, divergeAbove: Infinity })
  const wrap = (s: NelderMeadState): FitState<P> => ({
    t: s.t,
    params: decode(toFlat(s.x)),
    objective: s.value,
    evaluations: s.evaluations,
    converged: s.converged,
    diverged: s.diverged,
    optimiser: s,
  })
  return {
    name,
    init: () => wrap(nm.init({ x0: u0 })),
    step: (s) => wrap(nm.step(s.optimiser)),
    done: (s) => s.converged || s.diverged,
  }
}

export const logistic = (u: number): number => 1 / (1 + Math.exp(-u))
export const logit = (p: number): number => Math.log(p / (1 - p))
