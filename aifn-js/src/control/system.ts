/**
 * Linear time-invariant state-space systems x′ = Ax + Bu, y = Cx + Du (continuous) or x_{k+1} = Ax_k + Bu_k,
 * y_k = Cx_k + Du_k (discrete), their poles and stability, discretisation, and simulation as a traceable algorithm.
 */

import { eig, expm } from 'aifn/ode'
import { fromData, toFlat, type Matrix, type Vector } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { add, asM, block, eyeM, hcat, invM, mul, scaleM, sub, toT, vcat, zerosM, type M } from './dense'
import type { MatrixLike, VectorLike } from './vector'

/** A linear time-invariant system in state-space form. */
export type StateSpace = {
  /** State matrix, n×n. */
  A: Matrix
  /** Input matrix, n×m. */
  B: Matrix
  /** Output matrix, p×n. */
  C: Matrix
  /** Feedthrough matrix, p×m. */
  D: Matrix
  /** Sampling interval of a discrete system, or null for a continuous one. */
  dt: number | null
  /** State dimension n, inputs m, outputs p. */
  states: number
  inputs: number
  outputs: number
}

/** The arguments of `stateSpace`: `B` may be a vector (one input), `C` defaults to I (the state is the output). */
export type StateSpaceInput = {
  A: MatrixLike
  B: MatrixLike | VectorLike
  C?: MatrixLike | VectorLike
  D?: MatrixLike | VectorLike | number
  /** Sampling interval for a discrete system; omitted or null for continuous time. */
  dt?: number | null
}

/**
 * A state-space system from its matrices, with shapes checked. A vector `B` is one input (a column); a vector `C` is
 * one output (a row); `C` defaults to the identity and `D` to zeros.
 *
 * @example stateSpace({ A: [[0, 1], [0, 0]], B: [0, 1], C: [1, 0] }) // the double integrator, position measured
 */
export function stateSpace({ A, B, C, D, dt = null }: StateSpaceInput): StateSpace {
  const a = asM(A, 'stateSpace A')
  const n = a.r
  if (a.c !== n) throw new Error(`stateSpace: A must be square, got ${a.r}×${a.c}`)
  const b = asM(B, 'stateSpace B')
  if (b.r !== n) throw new Error(`stateSpace: B has ${b.r} rows, A is ${n}×${n}`)
  let c = C === undefined ? eyeM(n) : asM(C, 'stateSpace C')
  if (c.c === 1 && c.r === n && n !== 1) c = { d: c.d, r: 1, c: n } // a vector C is one output row
  if (c.c !== n) throw new Error(`stateSpace: C has ${c.c} columns, A is ${n}×${n}`)
  let d = D === undefined ? zerosM(c.r, b.c) : asM(D, 'stateSpace D')
  if (d.r * d.c === c.r * b.c) d = { d: d.d, r: c.r, c: b.c }
  else throw new Error(`stateSpace: D must be ${c.r}×${b.c}`)
  if (dt !== null && !(dt > 0)) throw new Error('stateSpace: dt must be positive (or null for continuous time)')
  return { A: toT(a), B: toT(b), C: toT(c), D: toT(d), dt, states: n, inputs: b.c, outputs: c.r }
}

/** The matrices of a system as working arrays. */
export function parts(sys: StateSpace): { A: M; B: M; C: M; D: M } {
  return { A: asM(sys.A, 'A'), B: asM(sys.B, 'B'), C: asM(sys.C, 'C'), D: asM(sys.D, 'D') }
}

/** Poles and the stability verdict of a system. */
export type Poles = {
  /** Real and imaginary parts of the eigenvalues of A (most unstable first). */
  real: Vector
  imag: Vector
  /**
   * Continuous: every Re λ < 0. Discrete: every |λ| < 1. Poles within 1e-12 of the boundary count as not stable
   * (marginal).
   */
  stable: boolean
  /** The largest Re λ (continuous) or |λ| (discrete): the stability margin to the boundary 0 or 1. */
  abscissa: number
  converged: boolean
}

/**
 * The poles of a system (the eigenvalues of A) and whether it is asymptotically stable: in the open left half-plane
 * for continuous time, inside the unit circle for discrete time. Pass a matrix with `{ discrete }` to test a bare A.
 */
export function poles(sys: StateSpace | MatrixLike, { discrete }: { discrete?: boolean } = {}): Poles {
  const isSys = typeof sys === 'object' && sys !== null && 'states' in sys
  const A = isSys ? (sys as StateSpace).A : (sys as MatrixLike)
  const disc = discrete ?? (isSys ? (sys as StateSpace).dt !== null : false)
  const e = eig(A, { vectors: false })
  const re = toFlat(e.real)
  const im = toFlat(e.imag)
  const abscissa = re.length ? Math.max(...re.map((r, i) => (disc ? Math.hypot(r, im[i]) : r))) : disc ? 0 : -Infinity
  return { real: e.real, imag: e.imag, stable: abscissa < (disc ? 1 : 0) - 1e-12, abscissa, converged: e.converged }
}

/** The system with state feedback u = −Kx + v: A ← A − BK, D and C unchanged. K is m×n. */
export function stateFeedback(sys: StateSpace, K: MatrixLike | VectorLike): StateSpace {
  const { A, B, C, D } = parts(sys)
  let k = asM(K, 'stateFeedback K')
  if (k.c === 1 && k.r === sys.states && sys.inputs === 1) k = { d: k.d, r: 1, c: sys.states }
  if (k.r !== sys.inputs || k.c !== sys.states) throw new Error(`stateFeedback: K must be ${sys.inputs}×${sys.states}`)
  return stateSpace({ A: toT(sub(A, mul(B, k))), B: toT(B), C: toT(sub(C, mul(D, k))), D: toT(D), dt: sys.dt })
}

/** Methods for `discretise`. */
export type DiscretisationMethod = 'zoh' | 'euler' | 'tustin'

/**
 * A continuous system sampled every `dt`:
 * - `zoh` (zero-order hold, exact for piecewise-constant inputs): exp([[A, B], [0, 0]]·dt) = [[A_d, B_d], [0, I]], so
 *   A_d = e^{A dt} and B_d = ∫₀^dt e^{As} ds B (Van Loan, 1978, "Computing integrals involving the matrix
 *   exponential"); C and D unchanged.
 * - `euler` (forward difference): A_d = I + A dt, B_d = B dt.
 * - `tustin` (bilinear, s ≈ (2/dt)(z − 1)/(z + 1)): with W = (I − A dt/2)⁻¹, A_d = W(I + A dt/2), B_d = W B dt,
 *   C_d = C W, D_d = D + C W B dt/2 (as scipy's `cont2discrete` with `bilinear`). Maps the left half-plane onto the
 *   unit disc, so stability is preserved.
 */
export function discretise(sys: StateSpace, dt: number, method: DiscretisationMethod = 'zoh'): StateSpace {
  if (sys.dt !== null) throw new Error('discretise: the system is already discrete')
  if (!(dt > 0)) throw new Error('discretise: dt must be positive')
  const { A, B, C, D } = parts(sys)
  const n = sys.states
  const m = sys.inputs
  if (method === 'euler')
    return stateSpace({ A: toT(add(eyeM(n), scaleM(A, dt))), B: toT(scaleM(B, dt)), C: toT(C), D: toT(D), dt })
  if (method === 'tustin') {
    const W = invM(sub(eyeM(n), scaleM(A, dt / 2)))
    if (!W) throw new Error('discretise: I − A dt/2 is singular (an eigenvalue of A at 2/dt)')
    const Bd = scaleM(mul(W, B), dt)
    return stateSpace({
      A: toT(mul(W, add(eyeM(n), scaleM(A, dt / 2)))),
      B: toT(Bd),
      C: toT(mul(C, W)),
      D: toT(add(D, scaleM(mul(C, Bd), 0.5))),
      dt,
    })
  }
  const big = vcat([hcat([A, B]), zerosM(m, n + m)])
  const E = asM(expm(toT(scaleM(big, dt))).value, 'discretise')
  return stateSpace({ A: toT(block(E, 0, n, 0, n)), B: toT(block(E, 0, n, n, n + m)), C: toT(C), D: toT(D), dt })
}

// ---------------------------------------------------------------------------------------------------------------------
// Simulation

/** The input to a simulation: a constant, or u(t, x) (which may be state feedback, e.g. u = −Kx). */
export type Input = number | VectorLike | ((t: number, x: Vector) => number | VectorLike)

/** The state of `simulate`. */
export type SimulationState = {
  /** Step number k and time t = k·dt. */
  step: number
  t: number
  /** The state x(t). */
  x: Vector
  /** The input applied from t to t + dt (held constant over the step). */
  u: Vector
  /** The output y(t) = C x(t) + D u(t). */
  y: Vector
  /** True once some component of x is not finite. */
  diverged: boolean
}

/** Options for `simulate`. */
export type SimulationOptions = {
  /** Time step. Required for a continuous system; a discrete system steps by its own `dt`. */
  dt?: number
  /** Stop at this time (inclusive). Omitted, the run lasts as many steps as the runner asks for. */
  tEnd?: number
  /** Discrete-time input index instead of a sampled time: an impulse at k = 0 of size 1/dt for a continuous system. */
  impulse?: boolean
}

function evalInput(u: Input, t: number, x: Vector, m: number): Float64Array<ArrayBuffer> {
  const v = typeof u === 'function' ? u(t, x) : u
  const out = typeof v === 'number' ? new Float64Array(m).fill(v) : Float64Array.from(asM(v, 'simulate input').d)
  if (out.length !== m) throw new Error(`simulate: the input has ${out.length} components, the system ${m}`)
  return out
}

/**
 * Simulates a system driven by `input` as a traceable algorithm. A continuous system is advanced exactly for an input
 * held constant over each step (zero-order hold: x_{k+1} = A_d x_k + B_d u_k with the `zoh` discretisation), so the
 * only approximation is the hold itself; a discrete system steps by its own recursion. `init` takes `{ x0 }`
 * (default zeros). The input is evaluated at the start of each step from (t, x), so state feedback is closed exactly
 * at the sampling instants.
 */
export function simulate(
  sys: StateSpace,
  input: Input = 0,
  options: SimulationOptions = {},
): Algorithm<{ x0?: VectorLike }, SimulationState> {
  const dt = sys.dt ?? options.dt
  if (dt === undefined || !(dt > 0)) throw new Error('simulate: a continuous system needs a positive dt')
  const d = sys.dt === null ? discretise(sys, dt, 'zoh') : sys
  const { A, B, C, D } = parts(d)
  const n = sys.states
  const m = sys.inputs
  const tEnd = options.tEnd ?? Infinity
  const output = (x: M, u: M) => add(mul(C, x), mul(D, u))
  const make = (step: number, x: M): SimulationState => {
    const t = step * dt
    const xv = fromData(Float64Array.from(x.d), [n])
    const raw = evalInput(input, t, xv, m)
    // An impulse is the input over the first step only; a continuous one has height 1/dt (exact as dt → 0).
    const scale = sys.dt === null ? 1 / dt : 1
    const u: M = { d: options.impulse ? raw.map((v) => (step === 0 ? v * scale : 0)) : raw, r: m, c: 1 }
    return {
      step,
      t,
      x: xv,
      u: fromData(Float64Array.from(u.d), [m]),
      y: fromData(Float64Array.from(output(x, u).d), [sys.outputs]),
      diverged: !x.d.every(Number.isFinite),
    }
  }
  return {
    name: 'lti-simulation',
    init: ({ x0 } = {}) => {
      const x = x0 === undefined ? zerosM(n, 1) : asM(x0, 'simulate x0')
      if (x.r * x.c !== n) throw new Error(`simulate: x0 must have ${n} components`)
      return make(0, { d: x.d, r: n, c: 1 })
    },
    step: (s) => {
      const x = { d: Float64Array.from(toFlat(s.x)), r: n, c: 1 }
      const u = { d: Float64Array.from(toFlat(s.u)), r: m, c: 1 }
      return make(s.step + 1, add(mul(A, x), mul(B, u)))
    },
    done: (s) => s.t >= tEnd - 1e-9 * dt,
  }
}

/** A sampled response: times, outputs, states and inputs stacked over time. */
export type Response = {
  /** Times, length T. */
  t: Vector
  /** Outputs, T×p. */
  y: Matrix
  /** States, T×n. */
  x: Matrix
  /** Inputs, T×m. */
  u: Matrix
  diverged: boolean
}

/** Runs `simulate` to `tEnd` and stacks the states. */
export function respond(
  sys: StateSpace,
  input: Input,
  { dt, tEnd, x0, impulse }: { dt?: number; tEnd: number; x0?: VectorLike; impulse?: boolean },
): Response {
  const step = sys.dt ?? dt
  if (step === undefined) throw new Error('respond: a continuous system needs dt')
  const alg = simulate(sys, input, { dt: step, tEnd, impulse })
  const T = Math.round(tEnd / step) + 1
  const ts = new Float64Array(T)
  const ys = new Float64Array(T * sys.outputs)
  const xs = new Float64Array(T * sys.states)
  const us = new Float64Array(T * sys.inputs)
  let s = alg.init({ x0 })
  let k = 0
  for (; k < T; k++) {
    ts[k] = s.t
    ys.set(toFlat(s.y), k * sys.outputs)
    xs.set(toFlat(s.x), k * sys.states)
    us.set(toFlat(s.u), k * sys.inputs)
    if (s.diverged || k === T - 1) break
    s = alg.step(s)
  }
  const kept = Math.min(k + 1, T)
  return {
    t: fromData(ts.slice(0, kept), [kept]),
    y: fromData(ys.slice(0, kept * sys.outputs), [kept, sys.outputs]),
    x: fromData(xs.slice(0, kept * sys.states), [kept, sys.states]),
    u: fromData(us.slice(0, kept * sys.inputs), [kept, sys.inputs]),
    diverged: s.diverged,
  }
}

/** The unit-step response from rest on input `input` (default 0): u_j(t) = 1 for t ≥ 0. */
export function stepResponse(
  sys: StateSpace,
  { tEnd, dt, input = 0 }: { tEnd: number; dt?: number; input?: number },
): Response {
  const u = new Float64Array(sys.inputs)
  u[input] = 1
  return respond(sys, u, { dt, tEnd })
}

/**
 * The unit-impulse response from rest on input `input`. Continuous: x(0⁺) = B e_j and u = 0 afterwards (exact; the
 * D δ(t) term is omitted). Discrete: u_0 = e_j, then 0.
 */
export function impulseResponse(
  sys: StateSpace,
  { tEnd, dt, input = 0 }: { tEnd: number; dt?: number; input?: number },
): Response {
  if (sys.dt !== null) {
    const u = new Float64Array(sys.inputs)
    u[input] = 1
    return respond(sys, u, { tEnd, impulse: true })
  }
  const B = asM(sys.B, 'B')
  const x0 = Array.from({ length: sys.states }, (_, i) => B.d[i * sys.inputs + input])
  return respond(sys, 0, { dt, tEnd, x0 })
}

/** The free response from `x0` with zero input. */
export function initialResponse(
  sys: StateSpace,
  x0: VectorLike,
  { tEnd, dt }: { tEnd: number; dt?: number },
): Response {
  return respond(sys, 0, { dt, tEnd, x0 })
}
