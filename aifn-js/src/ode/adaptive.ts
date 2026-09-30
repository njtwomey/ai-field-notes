/**
 * Adaptive Runge–Kutta: the Dormand–Prince 5(4) pair with error control, the first-same-as-last property, Hairer's
 * starting step size and a record of every attempted step (Dormand & Prince, 1980, "A family of embedded Runge–Kutta
 * formulae", J. Comput. Appl. Math. 6; Hairer, Nørsett & Wanner, 1993, §II.4, as in scipy's `RK45`).
 */

import { fromData, type Vector } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { combine, evaluate, initialState, stages, type ButcherTableau } from './explicit'
import type { InitialValue, OdeState, Rhs } from './types'
import { allFinite, toF64, type F64 } from './vector'

/** The Dormand–Prince 5(4) tableau: b gives the fifth-order solution, bHat the embedded fourth-order one. */
export const DORMAND_PRINCE: ButcherTableau = {
  name: 'dormand-prince',
  order: 5,
  c: [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1],
  a: [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
  ],
  b: [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
  bHat: [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
}

/** One attempted step of an adaptive solver. */
export type StepAttempt = { h: number; error: number; accepted: boolean }

/** The state of `dormandPrince`. */
export type AdaptiveState = OdeState & {
  /** The step size proposed for the next step. */
  hNext: number
  /** f(t, x) at the current point (reused as the first stage of the next step: first same as last). */
  derivative: Vector
  /** The attempts made to take the last step, in order; the last one was accepted. */
  attempts: StepAttempt[]
}

/** Options for `dormandPrince`. */
export type AdaptiveOptions = {
  /** The end time (required: it fixes the direction and bounds the last step). */
  tEnd: number
  /** Relative tolerance. Default 1e-3 (scipy's default). */
  rtol?: number
  /** Absolute tolerance. Default 1e-6. */
  atol?: number
  /** The first step size; default chosen by Hairer's algorithm. */
  h0?: number
  /** The largest step size allowed (magnitude). Default ∞. */
  hMax?: number
  /** The smallest step size (magnitude) before the run fails with `'step size underflow'`. Default 1e-12·|t|. */
  hMin?: number
}

const SAFETY = 0.9
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10

/** The weighted RMS norm ‖e‖ = √(mean((e_i / (atol + rtol·max(|x_i|, |y_i|)))²)). */
function errorNorm(e: F64, x: F64, y: F64, rtol: number, atol: number): number {
  let s = 0
  for (let i = 0; i < e.length; i++) {
    const sc = atol + rtol * Math.max(Math.abs(x[i]), Math.abs(y[i]))
    s += (e[i] / sc) ** 2
  }
  return Math.sqrt(s / Math.max(1, e.length))
}

/**
 * The starting step size of Hairer, Nørsett & Wanner (1993, §II.4, "Starting step size"): balance h so that an Euler
 * step's change and the estimated second derivative are both small relative to the tolerance. Costs one evaluation.
 */
function startingStep(f: Rhs, t0: number, x0: F64, f0: F64, dir: number, rtol: number, atol: number, order: number) {
  const scaled = (v: F64) => {
    let s = 0
    for (let i = 0; i < v.length; i++) s += (v[i] / (atol + Math.abs(x0[i]) * rtol)) ** 2
    return Math.sqrt(s / Math.max(1, v.length))
  }
  const d0 = scaled(x0)
  const d1 = scaled(f0)
  const h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : (0.01 * d0) / d1
  const x1 = Float64Array.from(x0, (v, i) => v + dir * h0 * f0[i])
  const f1 = evaluate(f, t0 + dir * h0, x1, 'dormandPrince')
  const d2 = scaled(Float64Array.from(f1, (v, i) => v - f0[i])) / h0
  const h1 = Math.max(d1, d2) <= 1e-15 ? Math.max(1e-6, h0 * 1e-3) : (0.01 / Math.max(d1, d2)) ** (1 / (order + 1))
  return Math.min(100 * h0, h1)
}

/**
 * The Dormand–Prince 5(4) adaptive solver for x′ = f(t, x) on [t₀, tEnd] (tEnd < t₀ integrates backwards). Each
 * step estimates its local error as the difference between the fifth- and fourth-order solutions, measured in the
 * weighted RMS norm with tolerances `rtol` and `atol`; a step with error norm above 1 is rejected and retried smaller.
 * The next step size is h·min(10, max(0.2, 0.9·err^{−1/5})) and the solution advances with the fifth-order result
 * (local extrapolation). Six evaluations per attempt, thanks to first-same-as-last. Each `step` of the algorithm is
 * one accepted step; its state lists the attempts it took (`attempts`), so a trace records the step-size history.
 */
export function dormandPrince(f: Rhs, options: AdaptiveOptions): Algorithm<InitialValue, AdaptiveState> {
  const { tEnd, rtol = 1e-3, atol = 1e-6, hMax = Infinity } = options
  if (!Number.isFinite(tEnd)) throw new Error('dormandPrince: tEnd must be finite')
  const tab = DORMAND_PRINCE
  const name = tab.name
  const errW = tab.b.map((b, i) => b - tab.bHat![i])
  return {
    name,
    init: ({ x0, t0 = 0 }) => {
      const x = toF64(x0, name)
      const base = initialState(x, t0)
      const dir = Math.sign(tEnd - t0) || 1
      const f0 = evaluate(f, t0, x, name)
      let evaluations = 1
      let h = options.h0
      if (h === undefined) {
        h = startingStep(f, t0, x, f0, dir, rtol, atol, 4)
        evaluations++
      }
      h = dir * Math.min(Math.abs(h), hMax)
      return { ...base, evaluations, hNext: h, derivative: fromData(f0, [f0.length]), attempts: [] }
    },
    step: (s) => {
      const x = s.x.data as F64
      const k0 = s.derivative.data as F64
      const dir = Math.sign(s.hNext) || 1
      const hMin = options.hMin ?? 1e-12 * Math.max(1, Math.abs(s.t))
      let h = s.hNext
      let evaluations = 0
      const attempts: StepAttempt[] = []
      for (;;) {
        // Do not step past tEnd.
        if (dir * (s.t + h - tEnd) > 0) h = tEnd - s.t
        if (Math.abs(h) < hMin) {
          return {
            ...s,
            attempts,
            evaluations: s.evaluations + evaluations,
            diverged: true,
            failure: 'step size underflow',
          }
        }
        const k = stages(f, tab, s.t, x, h, name, k0)
        evaluations += tab.b.length - 1
        const y = combine(x, h, tab.b, k)
        const e = combine(new Float64Array(x.length), h, errW, k)
        const err = errorNorm(e, x, y, rtol, atol)
        if (!allFinite(y) || !Number.isFinite(err)) {
          attempts.push({ h, error: err, accepted: false })
          h *= MIN_FACTOR
          continue
        }
        const factor = err === 0 ? MAX_FACTOR : Math.min(MAX_FACTOR, Math.max(MIN_FACTOR, SAFETY * err ** (-1 / 5)))
        if (err <= 1) {
          attempts.push({ h, error: err, accepted: true })
          // After a rejection in this step, do not grow the step again at once (Hairer et al., §II.4).
          const grow = attempts.length > 1 ? Math.min(1, factor) : factor
          const hNext = dir * Math.min(Math.abs(h * grow), hMax)
          // The last stage is evaluated at (t + h, y): first same as last.
          const fy = k[k.length - 1]
          return {
            t: s.t + h,
            x: fromData(y, [y.length]),
            step: s.step + 1,
            h,
            error: err,
            evaluations: s.evaluations + evaluations,
            jacobianEvaluations: 0,
            rejected: s.rejected + attempts.length - 1,
            diverged: false,
            failure: null,
            hNext,
            derivative: fromData(fy, [fy.length]),
            attempts,
          }
        }
        attempts.push({ h, error: err, accepted: false })
        h *= factor
      }
    },
    done: (s) => s.failure !== null || Math.abs(tEnd - s.t) <= 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}
