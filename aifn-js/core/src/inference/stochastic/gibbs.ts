/**
 * Gibbs sampling (Geman & Geman, 1984) with systematic or random scan over user-supplied full conditionals, the
 * Gaussian full conditionals, and univariate slice sampling (Neal, 2003) as a Gibbs step for any target.
 */

import { inverse } from 'aifn/numerics/linalg'
import { child, integers, normal, uniform, type Stream } from 'aifn/foundation/random'
import { fromData, tensor, type Matrix, type Tensor, type Vector } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import { badLogDensity } from './metropolis'
import type { ChainStart, ChainState, LogDensity, VectorLike } from './types'
import { allFinite, data, logDensityAt, mat, toF64, vec, type F64 } from './util'

/** A full conditional: a draw of coordinate i given the current point x (whose coordinate i is ignored). */
export type Conditional = (x: Vector, s: Stream) => number

/** The state of `gibbs`. `logDensity` is NaN: Gibbs never evaluates the joint. */
export type GibbsState = ChainState & {
  /** The path within the last step: the point before and after each coordinate update, (updates + 1)×d. */
  moves: Matrix
  /** The coordinate updated at each move of the last step (int32). */
  coordinates: Tensor
  /** Coordinate updates so far. */
  updates: number
}

/** Options for `gibbs`. */
export type GibbsOptions = {
  /** `systematic` updates coordinates 0, 1, …, d − 1 in order; `random` picks each uniformly. Default systematic. */
  scan?: 'systematic' | 'random'
  /** Coordinate updates per step (a sweep). Default d. */
  updatesPerStep?: number
}

/**
 * Gibbs sampling (Geman & Geman, 1984): replace one coordinate at a time by a draw from its full conditional
 * π(xᵢ | x₋ᵢ). Each update leaves π invariant, so both scans do; the random scan is also reversible. One step is one
 * sweep of `updatesPerStep` updates, and `moves` holds the axis-parallel zig-zag it traced. Update k of step t draws
 * from `child(ctx.stream, k)`.
 */
export function gibbs(
  conditionals: readonly Conditional[],
  options: GibbsOptions = {},
): Algorithm<ChainStart, GibbsState> {
  const name = 'gibbs'
  const d = conditionals.length
  const { scan = 'systematic', updatesPerStep = d } = options
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new Error(`${name}: x0 has ${x.length} values for ${d} conditionals`)
      return {
        t: 0,
        x: vec(x),
        logDensity: NaN,
        moves: mat(Float64Array.from(x), 1, d),
        coordinates: fromData(new Int32Array(0), [0]),
        updates: 0,
        diverged: !allFinite(x),
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      let x = data(s.x)
      const moves = new Float64Array((updatesPerStep + 1) * d)
      moves.set(x, 0)
      const coords = new Int32Array(updatesPerStep)
      for (let k = 0; k < updatesPerStep; k++) {
        const u = child(draws, k)
        const i = scan === 'random' ? integers(child(u, 'coordinate'), d) : (s.updates + k) % d
        const next = Float64Array.from(x)
        next[i] = conditionals[i](vec(x), u)
        x = next
        coords[k] = i
        moves.set(x, (k + 1) * d)
      }
      return {
        ...s,
        t: s.t + 1,
        x: vec(x),
        moves: mat(moves, updatesPerStep + 1, d),
        coordinates: fromData(coords, [updatesPerStep]),
        updates: s.updates + updatesPerStep,
        diverged: !allFinite(x),
      }
    },
  }
}

/**
 * The full conditionals of N(μ, Σ): xᵢ | x₋ᵢ ~ N(μᵢ − (1/Λᵢᵢ) Σ_{j≠i} Λᵢⱼ(xⱼ − μⱼ), 1/Λᵢᵢ) with Λ = Σ⁻¹ (Bishop, 2006,
 * eq. 2.75 in precision form). For a bivariate Gaussian with correlation ρ and unit variances this is
 * x₁ | x₂ ~ N(ρx₂, 1 − ρ²): the closer |ρ| is to 1, the shorter each zig-zag step and the slower the chain.
 */
export function gaussianConditionals(
  mean: VectorLike,
  covariance: Tensor | readonly (readonly number[])[],
): Conditional[] {
  const mu = toF64(mean, 'gaussianConditionals')
  const d = mu.length
  const cov = Array.isArray(covariance) ? tensor(covariance as number[][]) : (covariance as Tensor)
  const P = data(inverse(cov))
  return Array.from({ length: d }, (_, i) => (x: Vector, s: Stream) => {
    const xs = data(x)
    let shift = 0
    for (let j = 0; j < d; j++) if (j !== i) shift += P[i * d + j] * (xs[j] - mu[j])
    const precision = P[i * d + i]
    return normal(s, mu[i] - shift / precision, 1 / Math.sqrt(precision))
  })
}

/** The full conditionals of the standard bivariate Gaussian with correlation ρ (unit variances, zero means). */
export function bivariateGaussianConditionals(rho: number): Conditional[] {
  if (!(Math.abs(rho) < 1)) throw new Error('bivariateGaussianConditionals: |ρ| must be below 1')
  const sd = Math.sqrt(1 - rho * rho)
  return [(x, s) => normal(s, rho * data(x)[1], sd), (x, s) => normal(s, rho * data(x)[0], sd)]
}

// ---------------------------------------------------------------------------------------------------------------------
// Slice sampling.

/** The state of `sliceSampler`. */
export type SliceState = ChainState & {
  /** log y, the slice level drawn for each coordinate on the last sweep. */
  levels: Vector
  /** The final bracket [L, R] (after stepping out and shrinking) for each coordinate on the last sweep, d×2. */
  intervals: Matrix
  /** The bracket after stepping out and before shrinking, d×2. */
  steppedOut: Matrix
  /** Shrinkage steps on the last sweep, per coordinate. */
  shrinks: Vector
  /** Log-density evaluations so far. */
  evaluations: number
}

/** Options for `sliceSampler`. */
export type SliceOptions = {
  /** Initial bracket width w (one number or one per coordinate). Default 1. */
  width?: number | ArrayLike<number>
  /** Largest number of stepping-out steps m (Neal, 2003, Fig. 3). Default 32. */
  maxSteps?: number
}

/**
 * Univariate slice sampling in turn along each coordinate (Neal, 2003, §4, stepping out and shrinkage, Figs. 3 and 5):
 * draw a level log y = log π(x) + log u, place a bracket of width w at random around xᵢ, step it out until both ends
 * leave the slice, then draw uniformly in it, shrinking towards xᵢ on each rejection. It needs no step-size tuning
 * beyond w and is exact. One step is one sweep; sweep t, coordinate i draws from `child(ctx.stream, i)`.
 */
export function sliceSampler(target: LogDensity, options: SliceOptions = {}): Algorithm<ChainStart, SliceState> {
  const name = 'slice'
  const d = target.dim
  const { maxSteps = 32 } = options
  const width =
    typeof options.width === 'number' || options.width === undefined ? null : Float64Array.from(options.width)
  const w = (i: number) => (width ? width[i] : ((options.width as number | undefined) ?? 1))
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      if (x.length !== d) throw new Error(`${name}: x0 has ${x.length} values for dimension ${d}`)
      const logDensity = logDensityAt(target, x)
      if (logDensity === -Infinity) throw new Error(`${name}: x0 is outside the support`)
      return {
        t: 0,
        x: vec(x),
        logDensity,
        levels: vec(new Float64Array(d).fill(NaN)),
        intervals: mat(new Float64Array(2 * d).fill(NaN), d, 2),
        steppedOut: mat(new Float64Array(2 * d).fill(NaN), d, 2),
        shrinks: vec(new Float64Array(d)),
        evaluations: 1,
        diverged: badLogDensity(logDensity) || !allFinite(x),
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const x: F64 = Float64Array.from(data(s.x))
      let logDensity = s.logDensity
      let evaluations = s.evaluations
      const levels = new Float64Array(d)
      const intervals = new Float64Array(2 * d)
      const steppedOut = new Float64Array(2 * d)
      const shrinks = new Float64Array(d)
      const at = (i: number, v: number) => {
        const y = Float64Array.from(x)
        y[i] = v
        evaluations++
        return logDensityAt(target, y)
      }
      for (let i = 0; i < d; i++) {
        const u = child(draws, i)
        const level = logDensity + Math.log(uniform(u))
        const wi = w(i)
        const x0 = x[i]
        let L = x0 - wi * uniform(u)
        let R = L + wi
        // Stepping out, with at most maxSteps steps split at random between the ends.
        let J = Math.floor(maxSteps * uniform(u))
        let K = maxSteps - 1 - J
        while (J > 0 && at(i, L) > level) {
          L -= wi
          J--
        }
        while (K > 0 && at(i, R) > level) {
          R += wi
          K--
        }
        steppedOut[2 * i] = L
        steppedOut[2 * i + 1] = R
        let count = 0
        for (;;) {
          const x1 = L + uniform(u) * (R - L)
          const logX1 = at(i, x1)
          if (logX1 > level) {
            x[i] = x1
            logDensity = logX1
            break
          }
          count++
          if (x1 < x0) L = x1
          else R = x1
          if (R - L < 1e-12 * (1 + Math.abs(x0))) break // the bracket has collapsed onto x0: keep it
        }
        levels[i] = level
        intervals[2 * i] = L
        intervals[2 * i + 1] = R
        shrinks[i] = count
      }
      return {
        ...s,
        t: s.t + 1,
        x: vec(x),
        logDensity,
        levels: vec(levels),
        intervals: mat(intervals, d, 2),
        steppedOut: mat(steppedOut, d, 2),
        shrinks: vec(shrinks),
        evaluations,
        diverged: badLogDensity(logDensity) || !allFinite(x),
      }
    },
  }
}
