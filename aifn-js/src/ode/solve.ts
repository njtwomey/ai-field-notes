/** `solveIvp`: one call that runs any solver over an interval and returns the solution as tensors. */

import { fromData, toFlat, type Matrix, type Vector } from 'aifn/tensor'
import { trace, type Algorithm, type StopReason, type Trace } from 'aifn/trace'
import { dormandPrince } from './adaptive'
import { withEvents, type EventHit, type OdeEvent } from './events'
import { rungeKutta } from './explicit'
import { bdf, implicitEuler, trapezoid } from './implicit'
import type { InitialValue, JacobianOption, OdeState, Rhs } from './types'
import type { VectorLike } from './vector'

/** The solvers `solveIvp` can run by name. */
export type OdeMethod =
  'euler' | 'heun' | 'midpoint' | 'rk4' | 'dormand-prince' | 'implicit-euler' | 'trapezoid' | 'bdf1' | 'bdf2' | 'bdf3'

/** Options for `solveIvp`. */
export type SolveIvpOptions = {
  /** Default `'dormand-prince'`. */
  method?: OdeMethod
  /** The step size of a fixed-step method. Default (t₁ − t₀)/100. */
  h?: number
  /** Tolerances of the adaptive method. */
  rtol?: number
  atol?: number
  /** The Jacobian option of the implicit methods. */
  jacobian?: JacobianOption
  /** Events to locate (a terminal event stops the run). */
  events?: readonly OdeEvent[]
  /** The most steps to take. Default 100 000. */
  maxSteps?: number
}

/** The solution returned by `solveIvp`. */
export type OdeSolution = {
  /** The times of the accepted steps, t₀ first (length m). */
  t: Vector
  /** The states at those times (m × n). */
  x: Matrix
  /** The step sizes (length m; 0 at t₀). */
  h: Vector
  /** The local error estimates (length m; NaN where the method has none). */
  error: Vector
  evaluations: number
  jacobianEvaluations: number
  rejected: number
  events: EventHit[]
  /** `done` on reaching t₁ (or a terminal event), `limit` after `maxSteps`, `diverged` on failure. */
  stopped: StopReason
  failure: string | null
  /** The full trace, for figures that want every state. */
  trace: Trace<OdeState>
}

/** The named solver for [t₀, t₁]. */
export function solverFor(f: Rhs, t1: number, t0: number, options: SolveIvpOptions): Algorithm<InitialValue, OdeState> {
  const { method = 'dormand-prince', h = (t1 - t0) / 100, rtol, atol, jacobian } = options
  switch (method) {
    case 'dormand-prince':
      return dormandPrince(f, { tEnd: t1, rtol, atol })
    case 'implicit-euler':
      return implicitEuler(f, { h, tEnd: t1, jacobian })
    case 'trapezoid':
      return trapezoid(f, { h, tEnd: t1, jacobian })
    case 'bdf1':
    case 'bdf2':
    case 'bdf3':
      return bdf(f, Number(method[3]) as 1 | 2 | 3, { h, tEnd: t1, jacobian })
    default:
      return rungeKutta(f, method, { h, tEnd: t1 })
  }
}

/**
 * Solves x′ = f(t, x), x(t₀) = x₀ on [t₀, t₁] with the named method (default Dormand–Prince with rtol 1e-3 and atol
 * 1e-6, as scipy's `solve_ivp`) and returns every accepted step, the step sizes, error estimates, work counts and
 * located events.
 */
export function solveIvp(
  f: Rhs,
  [t0, t1]: readonly [number, number],
  x0: VectorLike,
  options: SolveIvpOptions = {},
): OdeSolution {
  const base = solverFor(f, t1, t0, options)
  const alg = (options.events?.length ? withEvents(base, f, options.events) : base) as Algorithm<InitialValue, OdeState>
  const tr = trace(alg, { x0, t0 }, options.maxSteps ?? 100_000, { stopOnNonFinite: false })
  const last = tr.steps[tr.steps.length - 1]
  const n = tr.steps[0].x.shape[0]
  const X = new Float64Array(tr.steps.length * n)
  tr.steps.forEach((s, k) => X.set(toFlat(s.x), k * n))
  return {
    t: fromData(
      Float64Array.from(tr.steps, (s) => s.t),
      [tr.steps.length],
    ),
    x: fromData(X, [tr.steps.length, n]),
    h: fromData(
      Float64Array.from(tr.steps, (s) => s.h),
      [tr.steps.length],
    ),
    error: fromData(
      Float64Array.from(tr.steps, (s) => s.error),
      [tr.steps.length],
    ),
    evaluations: last.evaluations,
    jacobianEvaluations: last.jacobianEvaluations,
    rejected: last.rejected,
    events: (last as OdeState & { events?: EventHit[] }).events ?? [],
    stopped: tr.meta.stopped,
    failure: last.failure,
    trace: tr,
  }
}
