/**
 * Event detection: find the times at which scalar functions g(t, x(t)) cross zero along a solution, as scipy's
 * `solve_ivp(events=…)` does. A sign change of g between two accepted steps brackets a crossing; the crossing is then
 * located by Brent's method on the cubic Hermite interpolant of the step (Hairer, Nørsett & Wanner, 1993, §II.6,
 * "Dense output" and "Discontinuities"; Shampine & Thompson, 2000).
 */

import { findRoot } from 'aifn/solve'
import { fromData, type Tensor, type Vector } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { evaluate } from './explicit'
import type { OdeState, Rhs } from './types'
import type { F64 } from './vector'

/** An event: the zero of g(t, x). */
export type OdeEvent = {
  name?: string
  g: (t: number, x: Tensor) => number
  /** Stop the integration at the first crossing. Default false. */
  terminal?: boolean
  /** Only crossings in this direction: +1 (g increasing), −1 (decreasing) or 0 (both, the default). */
  direction?: -1 | 0 | 1
}

/** A located event. */
export type EventHit = {
  /** The position of the event in the list passed to `withEvents`. */
  event: number
  name: string
  t: number
  x: Vector
  /** +1 if g increased through zero, −1 if it decreased. */
  direction: 1 | -1
}

/** A solver state with the events found so far. */
export type EventState<S extends OdeState> = S & {
  /** Every crossing found so far, in time order. */
  events: EventHit[]
  /** g at the current point, one per event. */
  eventValues: number[]
  /** True once a terminal event has stopped the run. */
  terminated: boolean
}

/**
 * The cubic Hermite interpolant on [t₀, t₁] through (t₀, x₀) and (t₁, x₁) with slopes f₀ and f₁: third-order accurate,
 * which is enough to locate events to well within a step's own error.
 */
export function hermite(t0: number, x0: F64, f0: F64, t1: number, x1: F64, f1: F64): (t: number) => F64 {
  const h = t1 - t0
  return (t) => {
    const s = (t - t0) / h
    const h00 = (1 + 2 * s) * (1 - s) ** 2
    const h10 = s * (1 - s) ** 2
    const h01 = s * s * (3 - 2 * s)
    const h11 = s * s * (s - 1)
    return Float64Array.from(x0, (v, i) => h00 * v + h10 * h * f0[i] + h01 * x1[i] + h11 * h * f1[i])
  }
}

/**
 * Wraps a solver so that it detects the zeros of the given event functions. After each step, every event whose g
 * changes sign (in its `direction`) is located by Brent's method on the step's Hermite interpolant, which costs two
 * extra evaluations of f per step with a crossing. A terminal event ends the run at the event: the state is moved to
 * (t_e, x(t_e)) and `terminated` is set. Works with any solver whose state is an `OdeState`.
 */
export function withEvents<Opts, S extends OdeState>(
  solver: Algorithm<Opts, S>,
  f: Rhs,
  events: readonly OdeEvent[],
): Algorithm<Opts, EventState<S>> {
  const values = (t: number, x: Vector) => events.map((e) => e.g(t, x))
  return {
    name: `${solver.name}+events`,
    init: (opts, s) => {
      const state = solver.init(opts, s)
      return { ...state, events: [], eventValues: values(state.t, state.x), terminated: false }
    },
    step: (s) => {
      const next = solver.step(s)
      const g1 = values(next.t, next.x)
      const hits: EventHit[] = []
      let evaluations = 0
      if (!next.diverged) {
        const x0 = s.x.data as F64
        const x1 = next.x.data as F64
        let interpolant: ((t: number) => F64) | null = null
        events.forEach((e, k) => {
          const a = s.eventValues[k]
          const b = g1[k]
          const up = a < 0 && b >= 0
          const down = a > 0 && b <= 0
          if (!(up || down)) return
          if ((e.direction ?? 0) === 1 && !up) return
          if ((e.direction ?? 0) === -1 && !down) return
          if (!interpolant) {
            const f0 = evaluate(f, s.t, x0, 'withEvents')
            const f1 = evaluate(f, next.t, x1, 'withEvents')
            evaluations += 2
            interpolant = hermite(s.t, x0, f0, next.t, x1, f1)
          }
          const at = interpolant
          const lo = Math.min(s.t, next.t)
          const hi = Math.max(s.t, next.t)
          const r =
            b === 0 ? { x: next.t } : findRoot((t) => e.g(t, fromData(at(t), [x0.length])), [lo, hi], { xtol: 1e-14 })
          const xe = at(r.x)
          hits.push({
            event: k,
            name: e.name ?? `event ${k}`,
            t: r.x,
            x: fromData(xe, [xe.length]),
            direction: up ? 1 : -1,
          })
        })
      }
      // Keep hits in the order the solution meets them.
      hits.sort((u, v) => (next.t >= s.t ? u.t - v.t : v.t - u.t))
      const firstTerminal = hits.find((hit) => events[hit.event].terminal)
      if (firstTerminal) {
        const kept = hits.filter((hit) => (next.t >= s.t ? hit.t <= firstTerminal.t : hit.t >= firstTerminal.t))
        return {
          ...next,
          t: firstTerminal.t,
          x: firstTerminal.x,
          h: firstTerminal.t - s.t,
          evaluations: next.evaluations + evaluations,
          events: [...s.events, ...kept],
          eventValues: values(firstTerminal.t, firstTerminal.x),
          terminated: true,
        }
      }
      return {
        ...next,
        evaluations: next.evaluations + evaluations,
        events: [...s.events, ...hits],
        eventValues: g1,
        terminated: false,
      }
    },
    done: (s) => s.terminated || (solver.done?.(s) ?? false),
  }
}
