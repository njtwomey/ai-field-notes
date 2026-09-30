/**
 * Internal helpers shared by the distribution families: shapes, raw (untraced) evaluation, masks, a few elementwise
 * primitives that `aifn/special` does not have, numerical inversion of cdfs, and the builder that turns a family's
 * specification into a `Univariate` object.
 */

import type { Stream } from 'aifn/random'
import {
  broadcastShapes,
  broadcastTo,
  defineUnary,
  fromData,
  isTensor,
  isTraced,
  item,
  shapeOfValue,
  sqrt,
  exp,
  log,
  sub,
  toFlat,
  unwrap,
  where,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn/tensor'
import type { ExponentialFamily, SampleOptions, Support, Univariate } from './types'

// ── Shapes and raw values ────────────────────────────────────────────────────────────────────────────────────────────

/** The broadcast shape of the parameters: the batch shape. */
export function batchShapeOf(...params: Value[]): number[] {
  return broadcastShapes(...params.map(shapeOfValue))
}

/** `v` broadcast to its broadcast shape with `others` (a batch mean when only the scale is batched, say). */
export function atBatch(v: Value, ...others: Value[]): Value {
  const batch = batchShapeOf(v, ...others)
  const shape = shapeOfValue(v)
  if (shape.length === batch.length && shape.every((d, k) => d === batch[k])) return v
  return broadcastTo(v, batch)
}

/** The untraced value of `v`; a traced value is an error naming `where` (the operation has no derivative). */
export function raw(v: Value, where: string): Raw {
  if (isTraced(v)) throw new Error(`${where}: not differentiable (a traced value was passed)`)
  return v
}

/** True when any of the values is traced. */
export function anyTraced(...vs: Value[]): boolean {
  return vs.some(isTraced)
}

/** A number, or a tensor as a number when it has one element; for scalar-only code paths. */
export function scalarOf(v: Raw): number {
  return typeof v === 'number' ? v : item(v)
}

/**
 * Apply a scalar function elementwise to broadcast raw values (a number when all are numbers). For masks and for
 * quantities computed without derivatives (entropies by summation, cdfs by quadrature).
 */
export function rawMap(values: readonly Value[], f: (...xs: number[]) => number): Raw {
  const rs = values.map((v) => unwrap(v))
  if (rs.every((r) => typeof r === 'number')) return f(...(rs as number[]))
  const shape = broadcastShapes(...rs.map((r) => (typeof r === 'number' ? [] : r.shape)))
  const flats = rs.map((r) => (typeof r === 'number' ? null : toFlat(broadcastTo(r, shape))))
  const n = shape.reduce((a, b) => a * b, 1)
  const out = new Float64Array(n)
  const args = new Array<number>(rs.length)
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < rs.length; j++) args[j] = flats[j] ? flats[j]![k] : (rs[j] as number)
    out[k] = f(...args)
  }
  return fromData(out, shape)
}

/** Like `rawMap`, but refuses traced inputs: for quantities with no derivative rule. */
export function rawOnly(where: string, values: readonly Value[], f: (...xs: number[]) => number): Raw {
  values.forEach((v) => raw(v, where))
  return rawMap(values, f)
}

// ── Masks ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A mask (1 where true) of a raw predicate over broadcast values. */
export function mask(values: readonly Value[], test: (...xs: number[]) => boolean): Raw {
  return rawMap(values, (...xs) => (test(...xs) ? 1 : 0))
}

/**
 * Replace `x` by `safe` where `valid` is 0, before evaluating an expression that is only defined on the valid set. The
 * result is then masked again with `outside`. Guarding the argument (not only the result) keeps the derivative finite:
 * a NaN or ∞ in the unused branch of `where` would otherwise turn a zero cotangent into NaN.
 */
export function guard(x: Value, valid: Raw, safe: number): Value {
  if (valid === 1) return x
  return where(valid, x, safe)
}

/** `expr` where `valid`, and `fill` elsewhere. */
export function outside(valid: Raw, expr: Value, fill: number): Value {
  if (valid === 1) return expr
  return where(valid, expr, fill)
}

/** Is x an integer? */
export const isInteger = (x: number): boolean => Number.isFinite(x) && Math.floor(x) === x

// ── Primitives not in aifn/special ───────────────────────────────────────────────────────────────────────────────────

// TODO(consolidation WP5): remove; import `xlogy` and `xlog1py` from aifn/special.
export { xlog1py, xlogy } from 'aifn/special'

/** arctan x, elementwise. */
export const atan = defineUnary('atan', Math.atan, (x) => 1 / (1 + x * x))

/** tan x, elementwise (radians). */
export const tan = defineUnary('tan', Math.tan, (_x, y) => 1 + y * y)

/**
 * The standard Cauchy cdf 1/2 + arctan(z)/π, computed as arctan(−1/z)/π for z < 0 so that the lower tail keeps its
 * relative accuracy.
 */
export const standardCauchyCdf = defineUnary(
  'standardCauchyCdf',
  (z) => (z < 0 ? Math.atan(-1 / z) / Math.PI : 0.5 + Math.atan(z) / Math.PI),
  (z) => 1 / (Math.PI * (1 + z * z)),
)

/** The Euler–Mascheroni constant γ. */
export const EULER_GAMMA = 0.5772156649015329

export const LOG_2PI = Math.log(2 * Math.PI)

// ── Numerical inversion of a cdf ─────────────────────────────────────────────────────────────────────────────────────

/** The bounds of a univariate support as raw values (−∞ and ∞ for the real line). */
export function supportBounds(support: Support): [Value, Value] {
  if (support.type === 'interval' || support.type === 'integers' || support.type === 'circle')
    return [support.lower, support.upper]
  return [-Infinity, Infinity]
}

/**
 * The quantile of a univariate distribution by bracketing and bisection on its cdf, elementwise over the broadcast of p
 * and the batch. Continuous: bisection to adjacent doubles (or a relative width of 4ε). Discrete: the smallest integer
 * k with cdf(k) ≥ p. p = 0 and p = 1 give the support's ends; p outside [0, 1] gives NaN. Not differentiable.
 */
export function invertCdf(
  name: string,
  cdf: (x: Tensor) => Value,
  p: Value,
  batchShape: readonly number[],
  support: Support,
  discrete: boolean,
): Raw {
  const pr = raw(p, `${name}.quantile`)
  const [lower, upper] = supportBounds(support).map((b) => raw(b, `${name}.quantile`))
  const shape = broadcastShapes(shapeOfValue(pr), batchShape)
  const n = shape.reduce((a, b) => a * b, 1)
  const at = (v: Raw) =>
    typeof v === 'number' ? new Float64Array(n).fill(v) : Float64Array.from(toFlat(broadcastTo(v, shape)))
  const ps = at(pr)
  const L = at(lower)
  const U = at(upper)
  const evaluate = (xs: Float64Array) => {
    const c = unwrap(cdf(fromData(xs, shape)))
    return typeof c === 'number' ? new Float64Array(n).fill(c) : Float64Array.from(toFlat(broadcastTo(c, shape)))
  }
  const out = new Float64Array(n)
  const active: boolean[] = []
  const lo = new Float64Array(n)
  const hi = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const q = ps[k]
    active[k] = false
    if (!(q >= 0 && q <= 1)) out[k] = NaN
    else if (q === 0) out[k] = L[k]
    else if (q === 1) out[k] = U[k]
    else {
      active[k] = true
      // Start from the finite end(s) of the support, or around 0 on the real line.
      lo[k] = Number.isFinite(L[k]) ? (discrete ? L[k] - 1 : L[k]) : Number.isFinite(U[k]) ? U[k] - 1 : -1
      hi[k] = Number.isFinite(U[k]) ? U[k] : Number.isFinite(L[k]) ? L[k] + 1 : 1
    }
  }
  // Expand the bracket until cdf(lo) < p ≤ cdf(hi); an infinite end is never evaluated.
  for (let it = 0, step = 1; it < 1100; it++, step *= 2) {
    const cl = evaluate(lo)
    const ch = evaluate(hi)
    let moved = false
    for (let k = 0; k < n; k++) {
      if (!active[k]) continue
      if (cl[k] >= ps[k] && lo[k] > L[k]) {
        lo[k] = Math.max(L[k], lo[k] - step)
        moved = true
      }
      if (ch[k] < ps[k] && hi[k] < U[k]) {
        hi[k] = Math.min(U[k], hi[k] + step)
        moved = true
      }
    }
    if (!moved) break
  }
  // Bisection, keeping cdf(lo) < p ≤ cdf(hi).
  for (let it = 0; it < 3000; it++) {
    const mid = Float64Array.from(lo)
    const live = new Uint8Array(n)
    let open = false
    for (let k = 0; k < n; k++) {
      if (!active[k]) continue
      if (discrete) {
        if (hi[k] - lo[k] <= 1) continue
        mid[k] = Math.floor((lo[k] + hi[k]) / 2)
      } else {
        const m = lo[k] + (hi[k] - lo[k]) / 2
        const width = 4 * Number.EPSILON * Math.max(Math.abs(lo[k]), Math.abs(hi[k]))
        if (m <= lo[k] || m >= hi[k] || hi[k] - lo[k] <= width) continue
        mid[k] = m
      }
      live[k] = 1
      open = true
    }
    if (!open) break
    const cm = evaluate(mid)
    for (let k = 0; k < n; k++) {
      if (!live[k]) continue
      if (cm[k] >= ps[k]) hi[k] = mid[k]
      else lo[k] = mid[k]
    }
  }
  for (let k = 0; k < n; k++) if (active[k]) out[k] = discrete ? hi[k] : lo[k] + (hi[k] - lo[k]) / 2
  return shape.length === 0 && typeof pr === 'number' ? out[0] : fromData(out, shape)
}

// ── The builder ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** A family's specification; everything else a `Univariate` needs is derived from it. */
export type UnivariateSpec = {
  name: string
  params: Record<string, Value>
  /** Default: the parameters' broadcast shape (families with a parameter vector per member override it). */
  batchShape?: readonly number[]
  support: Support
  discrete?: boolean
  logProb(x: Value): Value
  cdf(x: Value): Value
  /** Default log(cdf). */
  logcdf?(x: Value): Value
  /** Default 1 − cdf. */
  survival?(x: Value): Value
  /** Default log(survival). */
  logSurvival?(x: Value): Value
  /** Default: numerical inversion of the cdf (not differentiable). */
  quantile?(p: Value): Value
  /** Draws of the full shape `[...sampleShape, ...batchShape]`, from raw parameters. */
  sample(s: Stream, shape: number[]): Tensor
  mean(): Value
  variance(): Value
  entropy(): Value
  mode(): Value
  expFamily?: ExponentialFamily
}

/** Draws of a batch: the sample shape and the batch shape, and whether to return a number. */
export function drawShape(
  batchShape: readonly number[],
  options?: SampleOptions,
): { shape: number[]; scalar: boolean } {
  const shape = [...(options?.shape ?? []), ...batchShape]
  return { shape, scalar: options?.shape === undefined && batchShape.length === 0 }
}

/** Build a `Univariate` from a specification. */
export function univariate<P extends Value>(spec: UnivariateSpec): Univariate<P> {
  const batchShape = spec.batchShape ?? batchShapeOf(...Object.values(spec.params))
  const discrete = spec.discrete ?? false
  const self = {
    name: spec.name,
    params: spec.params,
    batchShape,
    eventShape: [] as const,
    support: spec.support,
    discrete,
    expFamily: spec.expFamily,
    logProb: (x: Value) => spec.logProb(x),
    prob: (x: Value) => exp(spec.logProb(x)),
    cdf: (x: Value) => spec.cdf(x),
    logcdf: (x: Value) => (spec.logcdf ? spec.logcdf(x) : log(spec.cdf(x))),
    survival: (x: Value) => (spec.survival ? spec.survival(x) : sub(1, spec.cdf(x))),
    logSurvival: (x: Value) =>
      spec.logSurvival ? spec.logSurvival(x) : log(spec.survival ? spec.survival(x) : sub(1, spec.cdf(x))),
    quantile: (p: Value) =>
      spec.quantile
        ? spec.quantile(p)
        : invertCdf(spec.name, (x) => spec.cdf(x), p, batchShape, spec.support, discrete),
    sample: (s: Stream, options?: SampleOptions) => {
      for (const [k, v] of Object.entries(spec.params)) raw(v, `${spec.name}.sample (parameter ${k})`)
      const { shape, scalar } = drawShape(batchShape, options)
      const t = spec.sample(s, shape)
      return scalar ? item(t) : t
    },
    mean: () => spec.mean(),
    variance: () => spec.variance(),
    stddev: () => sqrt(spec.variance()),
    entropy: () => spec.entropy(),
    mode: () => spec.mode(),
  }
  return self as unknown as Univariate<P>
}

/** Draw uniforms of a shape and map them through a quantile function (inverse-transform sampling). */
export function inverseTransform(s: Stream, shape: number[], quantile: (u: Tensor) => Value): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  const u = new Float64Array(n)
  for (let k = 0; k < n; k++) u[k] = s.uniform()
  const x = unwrap(quantile(fromData(u, shape)))
  return isTensor(x) ? broadcastTo(x, shape) : fromData(new Float64Array(n).fill(x), shape)
}

/**
 * Check a parameter's raw values (traced values are checked through their underlying value): throws a RangeError
 * naming the family, the parameter and the requirement when any element fails `test`.
 */
export function check(family: string, label: string, v: Value, test: (x: number) => boolean, what: string): void {
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : toFlat(r)
  for (const x of values) if (!test(x)) throw new RangeError(`${family}: ${label} must be ${what}, got ${x}`)
}
