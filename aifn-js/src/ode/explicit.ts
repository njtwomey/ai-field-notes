/**
 * Explicit Runge–Kutta methods from their Butcher tableaux: explicit Euler, Heun, the midpoint rule and the classical
 * fourth-order method (Butcher, 2016, "Numerical Methods for Ordinary Differential Equations", 3rd ed., §23; Hairer,
 * Nørsett & Wanner, 1993, "Solving Ordinary Differential Equations I", §II.1).
 */

import { fromData } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import type { FixedStepOptions, InitialValue, OdeState, Rhs } from './types'
import { allFinite, toF64, type F64 } from './vector'

/**
 * A Butcher tableau: stage times c, stage coefficients a (strictly lower triangular for an explicit method, rows of
 * length s), weights b, and optionally the weights `bHat` of an embedded lower-order solution for error estimates.
 */
export type ButcherTableau = {
  name: string
  /** The classical order of the method. */
  order: number
  c: readonly number[]
  a: readonly (readonly number[])[]
  b: readonly number[]
  bHat?: readonly number[]
}

/** Explicit Euler: x ← x + h f(t, x). Order 1. */
export const EULER: ButcherTableau = { name: 'euler', order: 1, c: [0], a: [[]], b: [1] }

/** Heun's method (the explicit trapezoid rule): average the slopes at both ends of an Euler step. Order 2. */
export const HEUN: ButcherTableau = { name: 'heun', order: 2, c: [0, 1], a: [[], [1]], b: [0.5, 0.5] }

/** The explicit midpoint rule: the slope at the midpoint of a half Euler step. Order 2. */
export const MIDPOINT: ButcherTableau = { name: 'midpoint', order: 2, c: [0, 0.5], a: [[], [0.5]], b: [0, 1] }

/** The classical fourth-order Runge–Kutta method (Kutta, 1901). Order 4. */
export const RK4: ButcherTableau = {
  name: 'rk4',
  order: 4,
  c: [0, 0.5, 0.5, 1],
  a: [[], [0.5], [0, 0.5], [0, 0, 1]],
  b: [1 / 6, 1 / 3, 1 / 3, 1 / 6],
}

/** The explicit tableaux by name. */
export const TABLEAUX = { euler: EULER, heun: HEUN, midpoint: MIDPOINT, rk4: RK4 } as const

/** Evaluates f and checks the length of its result. */
export function evaluate(f: Rhs, t: number, x: F64, where: string): F64 {
  const k = toF64(f(t, fromData(x, [x.length])), where)
  if (k.length !== x.length)
    throw new Error(`${where}: f returned ${k.length} values for a state of length ${x.length}`)
  return k
}

/** The stages k_i = f(t + c_i h, x + h Σ_j a_ij k_j) of an explicit tableau; `k0` reuses a known f(t, x). */
export function stages(f: Rhs, tab: ButcherTableau, t: number, x: F64, h: number, where: string, k0?: F64): F64[] {
  const k: F64[] = []
  for (let i = 0; i < tab.b.length; i++) {
    if (i === 0 && k0) {
      k.push(k0)
      continue
    }
    const y = Float64Array.from(x)
    const row = tab.a[i]
    for (let j = 0; j < row.length; j++) {
      const aij = row[j]
      if (aij === 0) continue
      for (let d = 0; d < y.length; d++) y[d] += h * aij * k[j][d]
    }
    k.push(evaluate(f, t + tab.c[i] * h, y, where))
  }
  return k
}

/** x + h Σ w_i k_i. */
export function combine(x: F64, h: number, w: readonly number[], k: F64[]): F64 {
  const y = Float64Array.from(x)
  for (let i = 0; i < w.length; i++) {
    if (w[i] === 0) continue
    for (let d = 0; d < y.length; d++) y[d] += h * w[i] * k[i][d]
  }
  return y
}

/** The initial state every solver starts from. */
export function initialState(x0: F64, t0: number): OdeState {
  return {
    t: t0,
    x: fromData(x0, [x0.length]),
    step: 0,
    h: 0,
    error: NaN,
    evaluations: 0,
    jacobianEvaluations: 0,
    rejected: 0,
    diverged: !allFinite(x0),
    failure: allFinite(x0) ? null : 'not finite',
  }
}

/** The step to take from t: h, shortened so as not to pass `tEnd`. */
export function nextStep(t: number, h: number, tEnd: number | undefined): number {
  if (tEnd === undefined) return h
  const left = tEnd - t
  return Math.abs(left) < Math.abs(h) ? left : h
}

/** True when a solver has reached `tEnd` (to rounding). */
export function reached(t: number, h: number, tEnd: number | undefined): boolean {
  if (tEnd === undefined) return false
  return Math.sign(h) * (tEnd - t) <= 1e-12 * Math.max(1, Math.abs(tEnd))
}

/**
 * A fixed-step explicit Runge–Kutta solver for x′ = f(t, x) from a Butcher tableau or the name of one (`'euler'`,
 * `'heun'`, `'midpoint'`, `'rk4'`). Each step costs one evaluation of f per stage; the global error is O(h^order).
 * `init` takes `{ x0, t0 }`; the run stops at `tEnd` when given. A non-finite state sets `diverged`.
 *
 * @example run(rungeKutta((t, x) => neg(x), 'rk4', { h: 0.1, tEnd: 1 }), { x0: [1] }, 100).x // ≈ e⁻¹
 */
export function rungeKutta(
  f: Rhs,
  method: keyof typeof TABLEAUX | ButcherTableau,
  { h, tEnd }: FixedStepOptions,
): Algorithm<InitialValue, OdeState> {
  const tab = typeof method === 'string' ? TABLEAUX[method] : method
  if (!tab) throw new Error(`rungeKutta: unknown method ${String(method)}`)
  if (!(h !== 0 && Number.isFinite(h))) throw new Error('rungeKutta: the step size h must be finite and non-zero')
  const name = tab.name
  return {
    name,
    init: ({ x0, t0 = 0 }) => initialState(toF64(x0, name), t0),
    step: (s) => {
      const x = s.x.data as F64
      const hk = nextStep(s.t, h, tEnd)
      const k = stages(f, tab, s.t, x, hk, name)
      const y = combine(x, hk, tab.b, k)
      const finite = allFinite(y)
      return {
        ...s,
        t: s.t + hk,
        x: fromData(y, [y.length]),
        step: s.step + 1,
        h: hk,
        evaluations: s.evaluations + tab.b.length,
        diverged: !finite,
        failure: finite ? null : 'not finite',
      }
    },
    done: (s) => reached(s.t, h, tEnd),
  }
}
