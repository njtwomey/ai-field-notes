/**
 * Reductions along axes, as NumPy's: `axis` is one axis, several, or (omitted or null) all of them, and `keepDims`
 * keeps reduced axes with length 1. Reducing every axis without `keepDims` returns a number. Sums and statistics
 * accumulate and return float64; `max` and `min` keep the dtype. Each is a primitive with its derivative rule, or a
 * composition of primitives.
 */

import { AifnError } from 'aifn/foundation/errors'
import { normaliseAxes, sizeOf, type Axis, type Tensor } from './core'
import { div, equalTo, mul, abs, square, sqrt, pow, exp, sub, notEqualTo, where } from './elementwise'
import { pairwiseSum, reducedShape, reduceKernel, type GroupReducer } from './kernels'
import { defineOp, type Raw } from './primitive'
import type { PrimitiveTest } from './registry'
import { broadcastTo, reshape, shapeOfValue } from './structure'
import { isTraced, unwrap, type Traced, type Value } from './tape'

/**
 * A reduction: every axis without `keepDims` gives a number, otherwise a tensor; traced inputs give traced results.
 * `E` lists extra trailing parameters (e.g. `ddof` for `variance`).
 */
export interface Reduction<E extends unknown[] = []> {
  (x: number | Tensor, axis?: null, keepDims?: false, ...extra: E): number
  (x: Tensor, axis: Axis, keepDims?: boolean, ...extra: E): Tensor
  (x: Tensor, axis: Axis | null | undefined, keepDims: true, ...extra: E): Tensor
  (x: Traced, axis?: Axis | null, keepDims?: boolean, ...extra: E): Traced
  (x: Value, axis?: Axis | null, keepDims?: boolean, ...extra: E): Value
}

type ReduceParams = { axis: Axis | null | undefined; keepDims: boolean }

/** True when the reduction covers every axis and drops them, so the result is a number. */
function toNumber({ axis, keepDims }: ReduceParams): boolean {
  return (axis === undefined || axis === null) && !keepDims
}

/** Run a per-group reducer as a reduction on a raw input. */
function reduceRaw(x: Raw, p: ReduceParams, fn: GroupReducer, keepDType = false): Raw {
  if (typeof x === 'number') {
    if (p.axis !== undefined && p.axis !== null) normaliseAxes(p.axis, 0, 'reduce')
    return fn([x], 0, 1)
  }
  const out = reduceKernel(x, toNumber(p) ? null : p.axis, p.keepDims, fn, keepDType ? x.dtype : 'float64')
  return toNumber(p) ? out.data[0] : out
}

/** Number of elements reduced into each output. */
function groupSize(x: Value, axis: Axis | null | undefined): number {
  const shape = shapeOfValue(x)
  return sizeOf(normaliseAxes(axis, shape.length, 'reduce').map((k) => shape[k]))
}

/** A reduced value (output or cotangent) broadcast back to the input's shape. */
function expand(v: Value, x: Value, p: ReduceParams): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 0) return v
  const keep = reducedShape(shape, normaliseAxes(p.axis, shape.length, 'reduce'), true)
  return broadcastTo(reshape(v, keep), shape)
}

/** Raw version of `expand`, for constants inside derivative rules. */
function expandRaw(v: Raw, x: Raw, p: ReduceParams): Raw {
  return unwrap(expand(v, x, p))
}

/** Test cases shared by the reductions: all axes, one axis, and one axis kept, on a 2×3 input. */
function reductionCases(extra: unknown[]): PrimitiveTest['cases'] {
  return (draw) => [
    { inputs: [draw([2, 3])], params: { axis: null, keepDims: false, extra } },
    { inputs: [draw([2, 3])], params: { axis: 1, keepDims: false, extra } },
    { inputs: [draw([2, 3])], params: { axis: [0], keepDims: true, extra } },
  ]
}

function reduction<E extends unknown[]>(
  name: string,
  forward: (x: Raw, p: ReduceParams, extra: E) => Raw,
  vjp: ((g: Value, x: Value, y: Value, p: ReduceParams, extra: E) => Value | null) | null,
  summary: string,
  extra: unknown[] = [],
): Reduction<E> {
  const op = defineOp<ReduceParams & { extra: E }>(
    `foundation/tensor/${name}`,
    ([x], p) => forward(x, p, p.extra),
    vjp && ((g, [x], y, p) => [vjp(g, x, y, p, p.extra)]),
    { arity: 1, doc: { summary }, test: { secondOrder: true, cases: reductionCases(extra) } },
  )
  return ((x: Value, axis?: Axis | null, keepDims = false, ...extra: E) =>
    op([x], { axis, keepDims, extra })) as Reduction<E>
}

/** Sum of elements, by pairwise summation above 128 terms per group (error O(ε log n), as NumPy). */
export const sum: Reduction = reduction(
  'sum',
  (x, p) => reduceRaw(x, p, pairwiseSum),
  (g, x, _y, p) => expand(g, x, p),
  'The sum of the elements.',
)

/** Arithmetic mean of elements. */
export const mean: Reduction = ((x: Value, axis?: Axis | null, keepDims = false) =>
  div(sum(x, axis, keepDims), groupSize(x, axis))) as Reduction

/** Product of elements. The derivative handles zeros exactly (the product of the other elements). */
export const prod: Reduction = reduction(
  'prod',
  (x, p) =>
    reduceRaw(x, p, (v, start, width) => {
      let s = 1
      for (let i = start, end = start + width; i < end; i++) s *= v[i]
      return s
    }),
  (g, x, y, p) => {
    const rx = unwrap(x)
    const zero = equalTo(rx, 0)
    const hasZero = typeof zero === 'number' ? zero === 1 : zero.data.some((z) => z !== 0)
    if (!hasZero) return mul(expand(g, x, p), div(expand(y, x, p), x))
    // With zeros, ∂(Π v)/∂v_i is the product of the others: the product of the non-zero ones if v_i is the only zero,
    // and 0 otherwise. These are constants here, so a second derivative through a zero is not supported.
    const withoutZeros = where(zero, 1, rx)
    const nonZeroProduct = expandRaw(prod(withoutZeros, p.axis, true) as Raw, rx, p)
    const zeros = expandRaw(sum(zero, p.axis, true) as Raw, rx, p)
    const others = where(zero, where(equalTo(zeros, 1), nonZeroProduct, 0), div(expandRaw(unwrap(y), rx, p), rx))
    return mul(expand(g, x, p), others)
  },
  'The product of the elements.',
)

function extreme(name: string, better: (a: number, b: number) => boolean, summary: string): Reduction {
  return reduction(
    name,
    (x, p) =>
      reduceRaw(
        x,
        p,
        (v, start, width) => {
          if (width === 0) throw new AifnError(name, `${name}: empty reduction`)
          let best = v[start]
          for (let i = start, end = start + width; i < end; i++) {
            if (v[i] !== v[i]) return NaN
            if (better(v[i], best)) best = v[i]
          }
          return best
        },
        true,
      ),
    (g, x, y, p) => {
      // The cotangent goes to the extreme elements, split equally among ties.
      const rx = unwrap(x)
      const hit = equalTo(rx, expandRaw(unwrap(y), rx, p))
      const count = expandRaw(sum(hit, p.axis, true) as Raw, rx, p)
      return mul(expand(g, x, p), div(hit, count))
    },
    summary,
  )
}

/** Largest element (NaN if any element is NaN). Ties share the derivative equally. */
export const max: Reduction = extreme('max', (a, b) => a > b, 'The largest element.')

/** Smallest element (NaN if any element is NaN). Ties share the derivative equally. */
export const min: Reduction = extreme('min', (a, b) => a < b, 'The smallest element.')

/**
 * log Σ exp(x), computed as m + log Σ exp(x − m) with m the maximum so that it neither overflows nor underflows
 * (−∞ when every element is −∞). Its derivative is the softmax of x.
 */
export const logsumexp: Reduction = reduction(
  'logsumexp',
  (x, p) =>
    reduceRaw(x, p, (v, start, width) => {
      const end = start + width
      let m = -Infinity
      for (let i = start; i < end; i++) if (v[i] > m || v[i] !== v[i]) m = v[i]
      if (m !== m) return NaN
      if (m === -Infinity || m === Infinity) return m
      let s = 0
      for (let i = start; i < end; i++) s += Math.exp(v[i] - m)
      return m + Math.log(s)
    }),
  (g, x, y, p) => mul(expand(g, x, p), exp(sub(x, expand(y, x, p)))),
  'log Σ exp(x), without overflow.',
)

/** The fused two-pass variance of one group: the mean first, then the centred sum of squares (Welford-free). */
function groupVariance(ddof: number): GroupReducer {
  return (v, start, width) => {
    const m = pairwiseSum(v, start, width) / width
    let s = 0
    for (let i = start, end = start + width; i < end; i++) s += (v[i] - m) * (v[i] - m)
    return s / (width - ddof)
  }
}

/**
 * Variance Σ(x − x̄)² / (n − ddof). `ddof` = 0 (default) is the population variance and 1 the unbiased sample
 * variance. A composition (design K §3.4, T9): traced inputs go through `mean` and `square`, so the derivative is theirs
 * and differentiable to any order; raw inputs take a fused two-pass kernel with the same value.
 */
export const variance: Reduction<[ddof?: number]> = ((x: Value, axis?: Axis | null, keepDims = false, ddof = 0) => {
  if (!isTraced(x)) return reduceRaw(x as Raw, { axis, keepDims }, groupVariance(ddof))
  const n = groupSize(x, axis)
  const centred = sub(x, mean(x, axis, true))
  return div(sum(square(centred), axis, keepDims), n - ddof)
}) as Reduction<[ddof?: number]>

/** Standard deviation, the square root of `variance` (same `ddof`). */
export const std: Reduction<[ddof?: number]> = ((x: Value, axis?: Axis | null, keepDims = false, ddof = 0) =>
  sqrt(variance(x, axis, keepDims, ddof))) as Reduction<[ddof?: number]>

/**
 * √Σx² of one group without overflow or underflow: the sum of squares of x / s with s the largest |x| (LAPACK's
 * `dnrm2` scaling, Blue 1978), times s.
 */
const groupEuclidean: GroupReducer = (v, start, width) => {
  const end = start + width
  let scale = 0
  for (let i = start; i < end; i++) {
    const a = Math.abs(v[i])
    if (a !== a) return NaN
    if (a > scale) scale = a
  }
  if (scale === 0 || scale === Infinity) return scale
  let s = 0
  for (let i = start; i < end; i++) {
    const r = v[i] / scale
    s += r * r
  }
  return scale * Math.sqrt(s)
}

/**
 * The Euclidean norm √Σx² along axes: the stable kernel of `norm` (no overflow for large entries, no underflow for
 * small ones). Its derivative is x/‖x‖, taken as 0 where ‖x‖ = 0 (the minimum-norm subgradient), so a zero vector has
 * a zero gradient rather than NaN.
 */
const euclidean: Reduction = reduction(
  'euclideanNorm',
  (x, p) => reduceRaw(x, p, groupEuclidean),
  (g, x, y, p) => {
    const ry = unwrap(y)
    const zero = equalTo(ry, 0)
    const safe = where(zero, 1, y)
    return mul(expand(g, x, p), where(expandRaw(zero, unwrap(x), p), 0, div(x, expand(safe, x, p))))
  },
  'The Euclidean norm √Σx², scaled against overflow.',
)

/**
 * Vector p-norm of the elements along `axis` (of all elements when omitted, which for a matrix is the Frobenius
 * norm): `ord` = 2 (default; the scaled kernel above, so no overflow and a zero gradient at 0), 1, ∞ (largest |x|),
 * −∞ (smallest |x|), 0 (count of non-zeros) or any p > 0.
 */
export const norm: Reduction<[ord?: number]> = ((x: Value, axis?: Axis | null, keepDims = false, ord = 2) => {
  if (ord === 2) return euclidean(x, axis, keepDims)
  if (ord === 1) return sum(abs(x), axis, keepDims)
  if (ord === Infinity) return max(abs(x), axis, keepDims)
  if (ord === -Infinity) return min(abs(x), axis, keepDims)
  if (ord === 0) return sum(notEqualTo(x, 0), axis, keepDims)
  if (!(ord > 0)) throw new AifnError('norm', `norm: unsupported order ${ord}`)
  return pow(sum(pow(abs(x), ord), axis, keepDims), 1 / ord)
}) as Reduction<[ord?: number]>

/** Index of the extreme element in each group, first occurrence winning; a NaN counts as the extreme. */
function argExtreme(name: string, better: (a: number, b: number) => boolean) {
  function arg(x: Value): number
  function arg(x: Value, axis: number, keepDims?: boolean): Tensor
  function arg(x: Value, axis?: number, keepDims = false): number | Tensor {
    const raw = unwrap(x)
    const pick: GroupReducer = (v, start, width) => {
      if (width === 0) throw new AifnError(name, `${name}: empty reduction`)
      let best = start
      for (let i = start, end = start + width; i < end; i++) {
        if (v[i] !== v[i]) return i - start
        if (better(v[i], v[best])) best = i
      }
      return best - start
    }
    if (typeof raw === 'number') return 0
    if (axis === undefined) return reduceKernel(raw, null, false, pick, 'int32').data[0]
    return reduceKernel(raw, axis, keepDims, pick, 'int32')
  }
  return arg
}

/**
 * Index of the largest element: a flat row-major index when `axis` is omitted, else an int32 tensor of indices along
 * `axis`. The first occurrence wins ties. Integer-valued, so it is never traced.
 */
export const argmax = argExtreme('argmax', (a, b) => a > b)

/** Index of the smallest element; see `argmax`. */
export const argmin = argExtreme('argmin', (a, b) => a < b)
