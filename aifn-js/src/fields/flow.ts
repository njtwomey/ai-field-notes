/**
 * Flows of autonomous fields x′ = f(x): flow maps and trajectories, streamlines, Poincaré sections and limit cycles
 * (Strogatz, 2015, "Nonlinear Dynamics and Chaos", 2nd ed., §6–8; Guckenheimer & Holmes, 1983, §1.5), and the
 * transport of a density along the flow by Liouville's equation.
 */

import { dormandPrince, rungeKutta, withEvents, type Rhs } from 'aifn/ode'
import { fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn/tensor'
import { run, trace } from 'aifn/trace'
import { divergence } from './calculus'
import type { VectorField } from './grid'
import { allFinite, toF64, toMatrixF64, type F64, type MatrixLike, type VectorLike } from './vector'

const autonomous =
  (f: VectorField): Rhs =>
  (_t, x) =>
    f(x)

/** Options for integrating a flow with fixed RK4 steps. */
export type FlowOptions = {
  /** The number of RK4 steps over the time span. Default 100. */
  steps?: number
}

/** The flow map φ_t(x₀): where the trajectory from x₀ is after time t (negative t runs backwards), by RK4. */
export function flowMap(f: VectorField, x0: VectorLike, t: number, { steps = 100 }: FlowOptions = {}): Vector {
  if (t === 0) return fromData(toF64(x0, 'flowMap'), [toF64(x0, 'flowMap').length])
  return run(rungeKutta(autonomous(f), 'rk4', { h: t / steps, tEnd: t }), { x0 }, steps + 1).x
}

/** The trajectory from x₀ over [0, t] by RK4: `t` (steps + 1 times) and `x` ((steps + 1) × n states). */
export function trajectory(
  f: VectorField,
  x0: VectorLike,
  t: number,
  { steps = 100 }: FlowOptions = {},
): { t: Vector; x: Matrix; diverged: boolean } {
  const tr = trace(rungeKutta(autonomous(f), 'rk4', { h: t / steps, tEnd: t }), { x0 }, steps + 1, {
    stopOnNonFinite: false,
  })
  const n = tr.steps[0].x.shape[0]
  const X = new Float64Array(tr.steps.length * n)
  tr.steps.forEach((s, k) => X.set(toFlat(s.x), k * n))
  return {
    t: fromData(
      Float64Array.from(tr.steps, (s) => s.t),
      [tr.steps.length],
    ),
    x: fromData(X, [tr.steps.length, n]),
    diverged: tr.meta.stopped === 'diverged',
  }
}

/** A box [lo_i, hi_i] per coordinate. */
export type Box = readonly (readonly [number, number])[]

/** Options for `streamline`. */
export type StreamlineOptions = {
  /** The time to follow the flow in each direction. Default 10. */
  t?: number
  /** RK4 steps per direction. Default 200. */
  steps?: number
  /** `'forward'`, `'backward'` or `'both'` (default): both joins the backward part, reversed, to the forward part. */
  direction?: 'forward' | 'backward' | 'both'
  /** Stop when the curve leaves this box (the last point is the first one outside). */
  bounds?: Box
  /** Stop when the speed ‖f‖ falls below this (the curve has reached a fixed point). Default 1e-9. */
  minSpeed?: number
}

function follow(f: VectorField, x0: F64, h: number, steps: number, bounds: Box | undefined, minSpeed: number): F64[] {
  const pts: F64[] = [x0]
  const rk = rungeKutta(autonomous(f), 'rk4', { h })
  let s = rk.init({ x0 })
  const inside = (x: F64) => !bounds || bounds.every(([lo, hi], i) => x[i] >= lo && x[i] <= hi)
  for (let k = 0; k < steps; k++) {
    const speed = Math.hypot(...toF64(f(s.x), 'streamline'))
    if (!(speed > minSpeed)) break
    s = rk.step(s)
    const x = s.x.data as F64
    if (!allFinite(x)) break
    pts.push(x)
    if (!inside(x)) break
  }
  return pts
}

/**
 * A streamline (trajectory curve) of the flow through `seed`, followed by RK4 forwards, backwards or both, stopping on
 * leaving `bounds`, on reaching a fixed point or on a non-finite state. Returns the points (m × n) in time order.
 */
export function streamline(f: VectorField, seed: VectorLike, options: StreamlineOptions = {}): Matrix {
  const { t = 10, steps = 200, direction = 'both', bounds, minSpeed = 1e-9 } = options
  const x0 = toF64(seed, 'streamline')
  const h = t / steps
  const forward = direction === 'backward' ? [x0] : follow(f, x0, h, steps, bounds, minSpeed)
  const backward = direction === 'forward' ? [] : follow(f, x0, -h, steps, bounds, minSpeed).slice(1).reverse()
  const pts = [...backward, ...forward]
  const n = x0.length
  const out = new Float64Array(pts.length * n)
  pts.forEach((p, k) => out.set(p, k * n))
  return fromData(out, [pts.length, n])
}

/** Streamlines through every row of `seeds` (k × n). */
export function streamlines(f: VectorField, seeds: MatrixLike, options: StreamlineOptions = {}): Matrix[] {
  const { data, m, n } = toMatrixF64(seeds, 'streamlines')
  return Array.from({ length: m }, (_, i) => streamline(f, data.subarray(i * n, (i + 1) * n), options))
}

/** A Poincaré section: the hyperplane {x : n·(x − p) = 0}, crossed in the direction of n. */
export type Section = { point: VectorLike; normal: VectorLike }

/** Options for Poincaré sections and limit cycles. */
export type SectionOptions = {
  /** Stop after this many crossings. Default 20. */
  crossings?: number
  /** Give up after this much time. Default 1000. */
  tMax?: number
  /** Tolerances of the Dormand–Prince integration. Defaults 1e-9 and 1e-11. */
  rtol?: number
  atol?: number
  /** Integrate backwards in time (finds unstable cycles). Default false. */
  reverse?: boolean
}

/**
 * The successive crossings of a trajectory from x₀ with a Poincaré section, in the direction of its normal (with
 * `reverse`, of the time-reversed flow): the orbit of the first-return map P. Located by event detection on an
 * adaptive Dormand–Prince integration. Returns the crossing times and points (k × n) and the return times.
 */
export function poincareSection(
  f: VectorField,
  x0: VectorLike,
  section: Section,
  options: SectionOptions = {},
): { t: Vector; points: Matrix; returnTimes: Vector } {
  const { crossings = 20, tMax = 1000, rtol = 1e-9, atol = 1e-11, reverse = false } = options
  const p = toF64(section.point, 'poincareSection')
  const nrm = toF64(section.normal, 'poincareSection')
  const sign = reverse ? -1 : 1
  const rhs: Rhs = reverse ? (_t, x) => toF64(f(x), 'poincareSection').map((v) => -v) : autonomous(f)
  const g = (_t: number, x: Tensor) => {
    const d = toFlat(x)
    let s = 0
    for (let i = 0; i < d.length; i++) s += nrm[i] * (d[i] - p[i])
    return s
  }
  const alg = withEvents(dormandPrince(rhs, { tEnd: tMax, rtol, atol }), rhs, [{ g, direction: 1 }])
  let s = alg.init({ x0 })
  let guard = 0
  while (s.events.length < crossings && !alg.done!(s) && !s.diverged && guard++ < 1_000_000) s = alg.step(s)
  const hits = s.events.slice(0, crossings)
  const n = p.length
  const pts = new Float64Array(hits.length * n)
  hits.forEach((e, k) => pts.set(toFlat(e.x), k * n))
  const times = Float64Array.from(hits, (e) => sign * e.t)
  const returns = Float64Array.from({ length: Math.max(0, hits.length - 1) }, (_, k) =>
    Math.abs(times[k + 1] - times[k]),
  )
  return {
    t: fromData(times, [hits.length]),
    points: fromData(pts, [hits.length, n]),
    returnTimes: fromData(returns, [returns.length]),
  }
}

/** A limit cycle found by iterating the first-return map. */
export type LimitCycle = {
  /** Whether successive returns agreed to `tol`. */
  converged: boolean
  /** The point where the cycle crosses the section. */
  point: Vector
  /** The period (time of one return). */
  period: number
  /** One period of the orbit (m × n), from `point`. */
  orbit: Matrix
  /**
   * For a planar flow, the derivative of the return map along the section (the nontrivial Floquet multiplier): the
   * cycle is stable when |multiplier| < 1. NaN in higher dimensions.
   */
  multiplier: number
  /** Distances between successive returns, which shrink geometrically by |multiplier| near a cycle. */
  residuals: Vector
}

/**
 * A limit cycle through a Poincaré section, found by iterating the return map P from x₀ until successive crossings
 * agree to `tol` (default 1e-8). This finds attracting cycles; with `reverse: true` it iterates the return map of
 * the time-reversed flow and so finds repelling ones. The Floquet multiplier of a planar cycle is P′ along the
 * section, estimated by central differences on the return map.
 */
export function limitCycle(
  f: VectorField,
  x0: VectorLike,
  section: Section,
  options: SectionOptions & { tol?: number; maxReturns?: number; orbitPoints?: number } = {},
): LimitCycle {
  const { tol = 1e-8, maxReturns = 200, orbitPoints = 200 } = options
  const once = (x: VectorLike) => {
    // The start is on the section already, so the first crossing is the return.
    const r = poincareSection(f, x, section, { ...options, crossings: 2 })
    const pts = toFlat(r.points)
    const n = r.points.shape[1] ?? 0
    if (r.points.shape[0] === 0) return null
    // A start exactly on the section may be reported as a crossing at t ≈ 0; skip it.
    const k = Math.abs(toFlat(r.t)[0]) < 1e-9 && r.points.shape[0] > 1 ? 1 : 0
    return { x: Float64Array.from(pts.slice(k * n, (k + 1) * n)), t: Math.abs(toFlat(r.t)[k]) }
  }
  const first = poincareSection(f, x0, section, { ...options, crossings: 1 })
  if (first.points.shape[0] === 0) throw new Error('limitCycle: the trajectory never crosses the section')
  let x = Float64Array.from(toFlat(first.points))
  let period = NaN
  let converged = false
  const residuals: number[] = []
  for (let k = 0; k < maxReturns; k++) {
    const next = once(x)
    if (!next) break
    const d = Math.hypot(...next.x.map((v, i) => v - x[i]))
    residuals.push(d)
    x = next.x
    period = next.t
    if (d <= tol * (1 + Math.hypot(...x))) {
      converged = true
      break
    }
  }
  let multiplier = NaN
  const nrm = toF64(section.normal, 'limitCycle')
  if (x.length === 2 && converged) {
    const tangent = [-nrm[1], nrm[0]].map((v) => v / Math.hypot(nrm[0], nrm[1]))
    const eps = 1e-5 * (1 + Math.hypot(...x))
    const along = (s: number) => {
      const r = once(x.map((v, i) => v + s * eps * tangent[i]))
      return r ? (r.x[0] - x[0]) * tangent[0] + (r.x[1] - x[1]) * tangent[1] : NaN
    }
    multiplier = (along(1) - along(-1)) / (2 * eps)
    if (options.reverse) multiplier = 1 / multiplier
  }
  const orbit = Number.isFinite(period)
    ? trajectory(f, x, options.reverse ? -period : period, { steps: orbitPoints }).x
    : fromData(Float64Array.from(x), [1, x.length])
  return {
    converged,
    point: fromData(x, [x.length]),
    period,
    orbit,
    multiplier,
    residuals: fromData(Float64Array.from(residuals), [residuals.length]),
  }
}

/** Options for `transportDensity`. */
export type TransportOptions = {
  /** RK4 steps along each characteristic. Default 50. */
  steps?: number
  /** ∇·f in closed form, if known (otherwise by autodiff, which is much slower). */
  divergence?: (x: Tensor) => number
}

/**
 * The density ρ(x, t) at the given points (k × n) of an initial density ρ₀ transported by the flow x′ = f(x):
 * the continuity (Liouville) equation ∂ρ/∂t + ∇·(ρf) = 0. Along a trajectory dρ/dt = −ρ ∇·f, so
 * ρ(x, t) = ρ₀(φ₋ₜ(x)) · exp(−∫₀ᵗ ∇·f(φ₋ₛ(x)) ds): each point is followed backwards to its origin while the divergence
 * is accumulated (the method of characteristics). Returns ρ at every point (length k).
 */
export function transportDensity(
  f: VectorField,
  rho0: (x: Tensor) => number,
  points: MatrixLike,
  t: number,
  { steps = 50, divergence: div }: TransportOptions = {},
): Vector {
  const m = toMatrixF64(points, 'transportDensity').m
  const frames = transportDensityFrames(f, rho0, points, t, { frames: 1, steps, divergence: div })
  return fromData(Float64Array.from((frames.data as F64).subarray(m)), [m])
}

/** The flow with its divergence integrated alongside, z = (x, L): x′ = sign·f(x), L′ = ∇·f(x). */
function withDivergence(f: VectorField, n: number, sign: 1 | -1, divAt: (x: Tensor) => number, where: string): Rhs {
  return (_t, z) => {
    const d = toFlat(z)
    const x = fromData(Float64Array.from(d.slice(0, n)), [n])
    const out = new Float64Array(n + 1)
    const fx = toF64(f(x), where)
    for (let i = 0; i < n; i++) out[i] = sign * fx[i]
    out[n] = divAt(x)
    return fromData(out, [n + 1])
  }
}

/** The states of `rhs` from z₀ at `frames` + 1 equally spaced times over [0, t], `steps` RK4 steps per frame. */
function framesOf(rhs: Rhs, z0: F64, t: number, frames: number, steps: number): F64[] {
  if (t === 0) return Array.from({ length: frames + 1 }, () => z0)
  const kept = trace(rungeKutta(rhs, 'rk4', { h: t / (frames * steps), tEnd: t }), { x0: z0 }, frames * steps + 1, {
    every: steps,
    stopOnNonFinite: false,
  }).steps
  return Array.from({ length: frames + 1 }, (_, k) => kept[Math.min(k, kept.length - 1)].x.data as F64)
}

/** Options for `transportDensityFrames` and `pushForwardFrames`. */
export type TransportFrameOptions = TransportOptions & {
  /** The number of equal time intervals over [0, t]; frames + 1 times are returned. Default 40. */
  frames?: number
  /** RK4 steps per frame. Default 1. */
  steps?: number
}

/**
 * `transportDensity` at `frames` + 1 equally spaced times over [0, t] (for playing the transport): each point's
 * characteristic is followed backwards once, over the whole span, and ρ is read off at every frame. Returns a
 * (frames + 1) × k matrix, row j the density at time j·t/frames.
 */
export function transportDensityFrames(
  f: VectorField,
  rho0: (x: Tensor) => number,
  points: MatrixLike,
  t: number,
  { frames = 40, steps = 1, divergence: div }: TransportFrameOptions = {},
): Matrix {
  const { data, m, n } = toMatrixF64(points, 'transportDensityFrames')
  const back = withDivergence(f, n, -1, div ?? ((x: Tensor) => divergence(f, x)), 'transportDensityFrames')
  const out = new Float64Array((frames + 1) * m)
  for (let k = 0; k < m; k++) {
    const z0 = new Float64Array(n + 1)
    z0.set(data.subarray(k * n, (k + 1) * n))
    framesOf(back, z0, t, frames, steps).forEach((z, j) => {
      out[j * m + k] = rho0(fromData(Float64Array.from(z.subarray(0, n)), [n])) * Math.exp(-z[n])
    })
  }
  return fromData(out, [frames + 1, m])
}

/**
 * Samples (rows of a k × n matrix) moved along the flow, at `frames` + 1 equally spaced times over [0, t], with the
 * log-volume change ∫₀ᵗ ∇·f ds accumulated along each path (the log of the local area ratio, so ρ falls by its
 * exponential along the path). Returns `x` ((frames + 1) × k × n) and `logVolume` ((frames + 1) × k).
 */
export function pushForwardFrames(
  f: VectorField,
  samples: MatrixLike,
  t: number,
  { frames = 40, steps = 1, divergence: div }: TransportFrameOptions = {},
): { x: Tensor; logVolume: Matrix } {
  const { data, m, n } = toMatrixF64(samples, 'pushForwardFrames')
  const fwd = withDivergence(f, n, 1, div ?? ((x: Tensor) => divergence(f, x)), 'pushForwardFrames')
  const x = new Float64Array((frames + 1) * m * n)
  const logVolume = new Float64Array((frames + 1) * m)
  for (let k = 0; k < m; k++) {
    const z0 = new Float64Array(n + 1)
    z0.set(data.subarray(k * n, (k + 1) * n))
    framesOf(fwd, z0, t, frames, steps).forEach((z, j) => {
      x.set(z.subarray(0, n), (j * m + k) * n)
      logVolume[j * m + k] = z[n]
    })
  }
  return { x: fromData(x, [frames + 1, m, n]), logVolume: fromData(logVolume, [frames + 1, m]) }
}

/** Moves every sample (row of a k × n matrix) along the flow for time t: the push-forward of an empirical density. */
export function pushForward(f: VectorField, samples: MatrixLike, t: number, options: FlowOptions = {}): Matrix {
  const { data, m, n } = toMatrixF64(samples, 'pushForward')
  const out = new Float64Array(m * n)
  for (let k = 0; k < m; k++) out.set(toFlat(flowMap(f, data.subarray(k * n, (k + 1) * n), t, options)), k * n)
  return fromData(out, [m, n])
}
