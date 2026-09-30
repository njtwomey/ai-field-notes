/**
 * A primal–dual interior-point method for linear programs: Mehrotra's predictor–corrector (Mehrotra, 1992, "On the
 * implementation of a primal-dual interior point method", SIAM J. Optimization 2(4)), as in Nocedal and Wright, 2006,
 * "Numerical Optimization", Algorithm 14.3, with the starting point of §14.2. It works on the standard form
 * min cᵀz s.t. Az = b, z ≥ 0 and follows the central path z∘s = μ1 towards μ = 0. Every iterate is recorded, and
 * `lpCentralPath` computes points on the exact central path for comparison.
 */

import type { Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { run } from 'aifn/trace'
import { dot, independentRows, intVector, matrix, norm2, normInf, solveSystem, vector } from './dense'
import {
  dualityReport,
  parseLP,
  recoverDuals,
  standardForm,
  toOriginal,
  type LinearProgram,
  type StandardForm,
} from './lp'
import { unsolved, type LinearProgramResult } from './simplex'

/** Options for `interiorPoint`. */
export interface InteriorPointOptions {
  problem: LinearProgram
  /** Stop when the relative primal and dual residuals and the relative duality gap are all below this (default 1e-9). */
  tol?: number
  /** Fraction of the step to the boundary taken (default 0.99). */
  stepFraction?: number
}

/** One iterate of the interior-point method. */
export interface InteriorPointState {
  /** The primal iterate in standard form, length N (strictly positive). */
  z: Tensor
  /** Dual variables of the standard-form rows, length m. */
  y: Tensor
  /** Dual slacks (reduced costs), length N (strictly positive). */
  s: Tensor
  /** The primal iterate in the original variables, length n: the point drawn on the central path. */
  x: Tensor
  /** The duality measure μ = zᵀs / N. */
  mu: number
  /** The centring parameter σ of the last step (Mehrotra's (μ_aff/μ)³). */
  sigma: number
  /** Primal and dual step lengths of the last step. */
  alphaPrimal: number
  alphaDual: number
  /** ‖Az − b‖ / (1 + ‖b‖). */
  primalResidual: number
  /** ‖Aᵀy + s − c‖ / (1 + ‖c‖). */
  dualResidual: number
  /** Relative duality gap |cᵀz − bᵀy| / (1 + |cᵀz|). */
  gap: number
  /** cᵀx of the current iterate. */
  objective: number
  iteration: number
  converged: boolean
  /** True when the normal equations became singular or the iterates grew without bound (infeasible or unbounded). */
  diverged: boolean
  /** Rows of the standard form kept after removing linearly dependent ones. */
  rowsKept: Tensor
  standard: StandardForm
  tol: number
  stepFraction: number
}

/** The reduced standard form: rows made linearly independent. */
type Reduced = { m: number; N: number; A: Float64Array; b: Float64Array; c: Float64Array; kept: number[] }

function reduce(sf: StandardForm): { reduced: Reduced; inconsistent: boolean } {
  const { rows, inconsistent } = independentRows({ m: sf.m, n: sf.N, a: sf.A }, sf.b)
  const A = new Float64Array(rows.length * sf.N)
  rows.forEach((r, i) => A.set(sf.A.subarray(r * sf.N, (r + 1) * sf.N), i * sf.N))
  return {
    reduced: { m: rows.length, N: sf.N, A, b: Float64Array.from(rows, (r) => sf.b[r]), c: sf.c, kept: rows },
    inconsistent,
  }
}

/** A D Aᵀ for a diagonal D. */
function normalMatrix(R: Reduced, d: Float64Array): Float64Array {
  const { m, N, A } = R
  const M = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let k = i; k < m; k++) {
      let s = 0
      for (let j = 0; j < N; j++) s += A[i * N + j] * d[j] * A[k * N + j]
      M[i * m + k] = s
      M[k * m + i] = s
    }
  return M
}

const Av = (R: Reduced, v: ArrayLike<number>) => {
  const out = new Float64Array(R.m)
  for (let i = 0; i < R.m; i++) {
    let s = 0
    for (let j = 0; j < R.N; j++) s += R.A[i * R.N + j] * v[j]
    out[i] = s
  }
  return out
}
const ATv = (R: Reduced, v: ArrayLike<number>) => {
  const out = new Float64Array(R.N)
  for (let i = 0; i < R.m; i++) for (let j = 0; j < R.N; j++) out[j] += R.A[i * R.N + j] * v[i]
  return out
}

/**
 * Solve the Newton system [0 Aᵀ I; A 0 0; S 0 Z] [Δz; Δy; Δs] = [−r_c; −r_b; −r_zs] by the normal equations
 * A D Aᵀ Δy = −r_b + A S⁻¹ r_zs − A D r_c with D = Z S⁻¹.
 */
function newton(
  R: Reduced,
  z: Float64Array,
  s: Float64Array,
  rb: Float64Array,
  rc: Float64Array,
  rzs: Float64Array,
): { dz: Float64Array; dy: Float64Array; ds: Float64Array; singular: boolean } {
  const d = z.map((zi, j) => zi / s[j])
  const t = new Float64Array(R.N)
  for (let j = 0; j < R.N; j++) t[j] = rzs[j] / s[j] - d[j] * rc[j]
  const rhs = Av(R, t)
  for (let i = 0; i < R.m; i++) rhs[i] -= rb[i]
  const { x: dy, singular } = solveSystem(normalMatrix(R, d), R.m, rhs)
  const Aty = ATv(R, dy)
  const ds = new Float64Array(R.N)
  const dz = new Float64Array(R.N)
  for (let j = 0; j < R.N; j++) {
    ds[j] = -rc[j] - Aty[j]
    dz[j] = (-rzs[j] - z[j] * ds[j]) / s[j]
  }
  return { dz, dy, ds, singular }
}

/** The largest α ≤ 1 with v + α dv ≥ 0. */
function maxStep(v: Float64Array, dv: Float64Array): number {
  let a = 1
  for (let j = 0; j < v.length; j++) if (dv[j] < 0) a = Math.min(a, -v[j] / dv[j])
  return a
}

function residuals(R: Reduced, z: Float64Array, y: Float64Array, s: Float64Array) {
  const rb = Av(R, z)
  for (let i = 0; i < R.m; i++) rb[i] -= R.b[i]
  const rc = ATv(R, y)
  for (let j = 0; j < R.N; j++) rc[j] += s[j] - R.c[j]
  return { rb, rc }
}

/** Assemble a state and its convergence measures. */
function makeState(
  base: Pick<InteriorPointState, 'standard' | 'tol' | 'stepFraction' | 'rowsKept'>,
  R: Reduced,
  z: Float64Array,
  y: Float64Array,
  s: Float64Array,
  extra: Pick<InteriorPointState, 'sigma' | 'alphaPrimal' | 'alphaDual' | 'iteration'> & { singular?: boolean },
): InteriorPointState {
  const { rb, rc } = residuals(R, z, y, s)
  const primalResidual = norm2(rb) / (1 + norm2(R.b))
  const dualResidual = norm2(rc) / (1 + norm2(R.c))
  const cz = dot(R.c, z)
  const gap = Math.abs(cz - dot(R.b, y)) / (1 + Math.abs(cz))
  const x = toOriginal(base.standard, z)
  const tol = base.tol
  const converged = primalResidual < tol && dualResidual < tol && gap < tol
  const big = Math.max(normInf(z), normInf(y), normInf(s))
  const diverged =
    !converged &&
    (extra.singular === true || !(big < 1e12) || [primalResidual, dualResidual, gap].some((v) => !Number.isFinite(v)))
  let objective = 0
  for (let j = 0; j < x.length; j++) objective += base.standard.lp.c[j] * x[j]
  return {
    ...base,
    sigma: extra.sigma,
    alphaPrimal: extra.alphaPrimal,
    alphaDual: extra.alphaDual,
    iteration: extra.iteration,
    z: vector(z),
    y: vector(y),
    s: vector(s),
    x: vector(x),
    mu: R.N ? dot(z, s) / R.N : 0,
    primalResidual,
    dualResidual,
    gap,
    objective,
    converged,
    diverged,
  }
}

/** Mehrotra's starting point (Nocedal and Wright, 2006, §14.2): least-norm z and least-squares y, shifted inside. */
function startingPoint(R: Reduced): { z: Float64Array; y: Float64Array; s: Float64Array } {
  const ones = new Float64Array(R.N).fill(1)
  const AAt = normalMatrix(R, ones)
  const { x: w, singular: s1 } = solveSystem(AAt, R.m, R.b)
  const { x: y, singular: s2 } = solveSystem(AAt, R.m, Av(R, R.c))
  if (s1 || s2) return { z: ones.slice(), y: new Float64Array(R.m), s: ones.slice() }
  const z = ATv(R, w)
  const Aty = ATv(R, y)
  const s = R.c.map((cj, j) => cj - Aty[j])
  const dz = Math.max(-1.5 * Math.min(...z, Infinity), 0)
  const ds = Math.max(-1.5 * Math.min(...s, Infinity), 0)
  for (let j = 0; j < R.N; j++) {
    z[j] += dz
    s[j] += ds
  }
  const zs = dot(z, s)
  const sumZ = z.reduce((a, b) => a + b, 0)
  const sumS = s.reduce((a, b) => a + b, 0)
  const dz2 = sumS > 0 ? (0.5 * zs) / sumS : 0
  const ds2 = sumZ > 0 ? (0.5 * zs) / sumZ : 0
  for (let j = 0; j < R.N; j++) {
    z[j] = z[j] + dz2 || 1
    s[j] = s[j] + ds2 || 1
  }
  // Keep strictly positive even for degenerate data (e.g. b = 0 and c = 0).
  for (let j = 0; j < R.N; j++) {
    if (!(z[j] > 0)) z[j] = 1
    if (!(s[j] > 0)) s[j] = 1
  }
  return { z, y, s }
}

/** The reduced problem is held by the state implicitly; rebuild it from the standard form and the kept rows. */
function reducedOf(state: InteriorPointState): Reduced {
  const sf = state.standard
  const kept = Array.from(state.rowsKept.data)
  const A = new Float64Array(kept.length * sf.N)
  kept.forEach((r, i) => A.set(sf.A.subarray(r * sf.N, (r + 1) * sf.N), i * sf.N))
  return { m: kept.length, N: sf.N, A, b: Float64Array.from(kept, (r) => sf.b[r]), c: sf.c, kept }
}

/**
 * The primal–dual interior-point method (Mehrotra's predictor–corrector) for linear programs as a traceable
 * algorithm. Options: `{ problem, tol?, stepFraction? }`. Each step is one predictor–corrector iteration; `x` records
 * the iterate in the original variables. The run is done when `converged`; it stops `diverged` when the problem is
 * infeasible or unbounded (the iterates grow without bound; use the simplex method to tell which) or inconsistent
 * equality constraints are found.
 */
export const interiorPoint: Algorithm<InteriorPointOptions, InteriorPointState> = {
  name: 'interior-point',
  init: (opts) => {
    const sf = standardForm(parseLP(opts.problem))
    const { reduced: R, inconsistent } = reduce(sf)
    const base = {
      standard: sf,
      tol: opts.tol ?? 1e-9,
      stepFraction: opts.stepFraction ?? 0.99,
      rowsKept: intVector(R.kept),
    }
    const { z, y, s } = startingPoint(R)
    return makeState(base, R, z, y, s, {
      sigma: NaN,
      alphaPrimal: 0,
      alphaDual: 0,
      iteration: 0,
      singular: inconsistent,
    })
  },
  step: (state) => {
    if (state.converged || state.diverged) return state
    const R = reducedOf(state)
    const z = Float64Array.from(state.z.data)
    const y = Float64Array.from(state.y.data)
    const s = Float64Array.from(state.s.data)
    const N = R.N
    const { rb, rc } = residuals(R, z, y, s)
    const mu = N ? dot(z, s) / N : 0
    // Predictor: the affine-scaling direction (σ = 0).
    const rzs = z.map((zj, j) => zj * s[j])
    const aff = newton(R, z, s, rb, rc, rzs)
    const ap = maxStep(z, aff.dz)
    const ad = maxStep(s, aff.ds)
    let muAff = 0
    for (let j = 0; j < N; j++) muAff += (z[j] + ap * aff.dz[j]) * (s[j] + ad * aff.ds[j])
    muAff = N ? muAff / N : 0
    const sigma = mu > 0 ? Math.min(1, (muAff / mu) ** 3) : 0
    // Corrector: re-centre towards σμ and correct for the second-order term Δz_aff ∘ Δs_aff.
    for (let j = 0; j < N; j++) rzs[j] = z[j] * s[j] + aff.dz[j] * aff.ds[j] - sigma * mu
    const dir = newton(R, z, s, rb, rc, rzs)
    const eta = state.stepFraction
    const alphaPrimal = Math.min(1, eta * maxStep(z, dir.dz))
    const alphaDual = Math.min(1, eta * maxStep(s, dir.ds))
    for (let j = 0; j < N; j++) {
      z[j] += alphaPrimal * dir.dz[j]
      s[j] += alphaDual * dir.ds[j]
    }
    for (let i = 0; i < R.m; i++) y[i] += alphaDual * dir.dy[i]
    return makeState(state, R, z, y, s, {
      sigma,
      alphaPrimal,
      alphaDual,
      iteration: state.iteration + 1,
      singular: aff.singular || dir.singular,
    })
  },
  done: (s) => s.converged,
}

/** Solve a linear program by the interior-point method (a `run` of `interiorPoint` to convergence). */
export function interiorPointSolve(
  problem: LinearProgram,
  options: { tol?: number; maxIterations?: number } = {},
): LinearProgramResult {
  const s = run(interiorPoint, { problem, tol: options.tol }, options.maxIterations ?? 200)
  if (!s.converged) return unsolved(s.standard.lp.n, s.diverged ? 'diverged' : 'limit', s.iteration, 'interior-point')
  const duals = recoverDuals(s.standard, s.y.data, s.rowsKept.data)
  return {
    status: 'optimal',
    x: s.x,
    objective: s.objective,
    iterations: s.iteration,
    method: 'interior-point',
    report: dualityReport(s.standard.lp, s.x, duals),
    ray: null,
  }
}

/** Points on the central path of a linear program. */
export interface CentralPath {
  /** The barrier parameters, as given. */
  mu: Tensor
  /** The central point x(μ) in the original variables, one row per μ: shape [k, n]. */
  x: Tensor
  /** cᵀx(μ). */
  objective: Tensor
  /** True for each μ whose Newton iteration met the tolerance, int32 (0 or 1). */
  converged: Tensor
}

/**
 * The exact central path of a linear program: for each μ, the solution of Az = b, Aᵀy + s = c, z∘s = μ1, z, s > 0
 * (the minimiser of cᵀz − μ Σ log zⱼ subject to Az = b; Nocedal and Wright, 2006, §14.1). Each point is found by
 * damped Newton iterations warm-started from the previous one, so give `mu` in decreasing order. The path exists
 * when the primal and dual problems are both strictly feasible.
 */
export function lpCentralPath(problem: LinearProgram, mu: readonly number[] | Tensor, tol = 1e-10): CentralPath {
  const sf = standardForm(parseLP(problem))
  const { reduced: R } = reduce(sf)
  const mus = Array.from(Array.isArray(mu) ? mu : (mu as Tensor).data)
  let { z, y, s } = startingPoint(R)
  const n = sf.lp.n
  const xs = new Float64Array(mus.length * n)
  const objective = new Float64Array(mus.length)
  const converged = new Int32Array(mus.length)
  mus.forEach((target, k) => {
    for (let it = 0; it < 100; it++) {
      const { rb, rc } = residuals(R, z, y, s)
      const rzs = z.map((zj, j) => zj * s[j] - target)
      const size = Math.max(normInf(rb), normInf(rc), normInf(rzs) / Math.max(target, 1e-300))
      if (size < tol) {
        converged[k] = 1
        break
      }
      const dir = newton(R, z, s, rb, rc, rzs)
      if (dir.singular) break
      const a = Math.min(1, 0.995 * Math.min(maxStep(z, dir.dz), maxStep(s, dir.ds)))
      z = z.map((v, j) => v + a * dir.dz[j])
      s = s.map((v, j) => v + a * dir.ds[j])
      y = y.map((v, i) => v + a * dir.dy[i])
    }
    const x = toOriginal(sf, z)
    xs.set(x, k * n)
    objective[k] = dot(sf.lp.c, x)
  })
  return {
    mu: vector(mus),
    x: matrix(xs, mus.length, n),
    objective: vector(objective),
    converged: intVector(converged),
  }
}
