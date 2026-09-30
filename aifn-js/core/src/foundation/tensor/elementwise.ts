/**
 * Elementwise primitives with NumPy broadcasting. Each is defined once with its derivative rule; the rules are written
 * with primitives so that derivatives of derivatives work for these built-ins.
 */

import { allocate, flatData, fromData, sizeOf, type Tensor } from './core'
import { binaryDType, binaryKernel, unaryKernel } from './kernels'
import { binaryPrimitive, defineOp, sumLike, unaryPrimitive, type Binary, type Raw, type Unary } from './primitive'
import { isTraced, unwrap, type Traced, type Value } from './tape'
import { broadcastShapes, broadcastView } from './views'

// ── Unary ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** −x. */
export const neg: Unary = unaryPrimitive(
  'foundation/tensor/neg',
  (x) => -x,
  (g) => neg(g),
  { integer: true, doc: { summary: 'Negation −x.' } },
)

/** The sign of x: −1, 0 or 1 (NaN for NaN). Its derivative is zero wherever it exists. */
export const sign: Unary = unaryPrimitive('foundation/tensor/sign', Math.sign, () => null, {
  integer: true,
  doc: { summary: 'The sign of x: −1, 0 or 1.' },
})

/** |x|. The derivative at 0 is taken as 0 (the subgradient sign(0)). */
export const abs: Unary = unaryPrimitive('foundation/tensor/abs', Math.abs, (g, x) => mul(g, sign(x)), {
  integer: true,
  doc: { summary: 'The absolute value |x|.' },
})

/** x². */
export const square: Unary = unaryPrimitive(
  'foundation/tensor/square',
  (x) => x * x,
  (g, x) => mul(g, mul(2, x)),
  { integer: true, doc: { summary: 'The square x².' } },
)

/** eˣ. */
export const exp: Unary = unaryPrimitive('foundation/tensor/exp', Math.exp, (g, _x, y) => mul(g, y), {
  doc: { summary: 'The exponential eˣ.' },
})

/** eˣ − 1, accurate for small x. */
export const expm1: Unary = unaryPrimitive('foundation/tensor/expm1', Math.expm1, (g, _x, y) => mul(g, add(y, 1)), {
  doc: { summary: 'eˣ − 1, accurate for small x.' },
})

/** Natural logarithm (NaN below 0, −∞ at 0). */
export const log: Unary = unaryPrimitive('foundation/tensor/log', Math.log, (g, x) => div(g, x), {
  doc: { summary: 'The natural logarithm.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/** log(1 + x), accurate for small x. */
export const log1p: Unary = unaryPrimitive('foundation/tensor/log1p', Math.log1p, (g, x) => div(g, add(x, 1)), {
  doc: { summary: 'log(1 + x), accurate for small x.' },
  test: { domain: { lo: -0.5, hi: 2 } },
})

/** √x (NaN below 0). */
export const sqrt: Unary = unaryPrimitive('foundation/tensor/sqrt', Math.sqrt, (g, _x, y) => div(g, mul(2, y)), {
  doc: { summary: 'The square root.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/** sin x (radians). */
export const sin: Unary = unaryPrimitive('foundation/tensor/sin', Math.sin, (g, x) => mul(g, cos(x)), {
  doc: { summary: 'The sine (radians).' },
})

/** cos x (radians). */
export const cos: Unary = unaryPrimitive('foundation/tensor/cos', Math.cos, (g, x) => neg(mul(g, sin(x))), {
  doc: { summary: 'The cosine (radians).' },
})

/** tanh x. */
export const tanh: Unary = unaryPrimitive(
  'foundation/tensor/tanh',
  Math.tanh,
  (g, _x, y) => mul(g, sub(1, square(y))),
  {
    doc: { summary: 'The hyperbolic tangent.' },
  },
)

// ── Binary ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** a + b. */
export const add: Binary = binaryPrimitive(
  'foundation/tensor/add',
  (a, b) => a + b,
  (g) => [g, g],
  { integer: true, kernel: 'add', doc: { summary: 'The sum a + b.' } },
)

/** a − b. */
export const sub: Binary = binaryPrimitive(
  'foundation/tensor/sub',
  (a, b) => a - b,
  (g) => [g, neg(g)],
  { integer: true, kernel: 'sub', doc: { summary: 'The difference a − b.' } },
)

/** a · b (elementwise). */
export const mul: Binary = binaryPrimitive(
  'foundation/tensor/mul',
  (a, b) => a * b,
  (g, a, b) => [mul(g, b), mul(g, a)],
  { integer: true, kernel: 'mul', doc: { summary: 'The product a·b.' } },
)

/** a / b (always floating point). */
export const div: Binary = binaryPrimitive(
  'foundation/tensor/div',
  (a, b) => a / b,
  (g, _a, b, y) => [div(g, b), neg(div(mul(g, y), b))],
  {
    kernel: 'div',
    doc: { summary: 'The quotient a / b.' },
    test: {
      domain: [
        { lo: -2, hi: 2 },
        { lo: 0.5, hi: 2 },
      ],
    },
  },
)

/**
 * aᵇ (always floating point). The derivative in b is y·log a, taken as 0 where a = 0 (the limit for b > 0) and NaN
 * for a < 0, where aᵇ is not differentiable in b.
 */
export const pow: Binary = binaryPrimitive(
  'foundation/tensor/pow',
  Math.pow,
  (g, a, b, y) => [
    mul(g, mul(b, pow(a, sub(b, 1)))),
    // Only a traced exponent needs its cotangent; for a constant one, log a (NaN for a < 0) is not computed.
    isTraced(b) ? mul(g, where(equalTo(unwrap(a), 0), 0, mul(y, log(a)))) : null,
  ],
  {
    doc: { summary: 'The power aᵇ.' },
    test: {
      domain: [
        { lo: 0.2, hi: 2 },
        { lo: -2, hi: 2 },
      ],
    },
  },
)

/** The larger of a and b. Where they tie the cotangent goes to a. NaN propagates. */
export const maximum: Binary = binaryPrimitive(
  'foundation/tensor/maximum',
  (a, b) => (a !== a || b !== b ? NaN : a >= b ? a : b),
  (g, a, b) => {
    const pick = greaterEqual(unwrap(a), unwrap(b))
    return [where(pick, g, 0), where(pick, 0, g)]
  },
  { integer: true, doc: { summary: 'The larger of a and b.' } },
)

/** The smaller of a and b. Where they tie the cotangent goes to a. NaN propagates. */
export const minimum: Binary = binaryPrimitive(
  'foundation/tensor/minimum',
  (a, b) => (a !== a || b !== b ? NaN : a <= b ? a : b),
  (g, a, b) => {
    const pick = lessEqual(unwrap(a), unwrap(b))
    return [where(pick, g, 0), where(pick, 0, g)]
  },
  { integer: true, doc: { summary: 'The smaller of a and b.' } },
)

/** x limited to [lo, hi] elementwise (an explicit operation: nothing in aifn clips silently). */
export function clip(x: Value, lo: Value, hi: Value): Value {
  return minimum(maximum(x, lo), hi)
}

// ── Comparisons (int32 results, 1 for true; piecewise constant, so their cotangents are zero) ────────────────────────

/** A comparison: numbers give 1 or 0, tensors give an int32 mask. */
export interface Comparison {
  (a: number, b: number): number
  (a: Tensor, b: Value): Tensor
  (a: Value, b: Tensor): Tensor
  (a: Value, b: Value): Tensor | number
}

// A mask does not depend differentiably on its inputs, so comparisons unwrap traced inputs and are never recorded.
function comparison(test: (a: number, b: number) => boolean): Comparison {
  const f = (a: number, b: number) => (test(a, b) ? 1 : 0)
  return ((x: Value, y: Value) => {
    const a = unwrap(x)
    const b = unwrap(y)
    return typeof a === 'number' && typeof b === 'number' ? f(a, b) : binaryKernel(a, b, f, 'int32')
  }) as Comparison
}

/** a < b. */
export const less = comparison((a, b) => a < b)
/** a ≤ b. */
export const lessEqual = comparison((a, b) => a <= b)
/** a > b. */
export const greater = comparison((a, b) => a > b)
/** a ≥ b. */
export const greaterEqual = comparison((a, b) => a >= b)
/** a = b (NaN equals nothing). */
export const equalTo = comparison((a, b) => a === b)
/** a ≠ b. */
export const notEqualTo = comparison((a, b) => a !== b)

// ── Selection and arbitrary maps ─────────────────────────────────────────────────────────────────────────────────────

const whereOp = defineOp<undefined>(
  'foundation/tensor/where',
  ([c, a, b]) => {
    if (typeof c === 'number' && typeof a === 'number' && typeof b === 'number') return c !== 0 ? a : b
    const asTensor = (v: Raw): Tensor => (typeof v === 'number' ? fromData(new Float64Array([v]), []) : v)
    const [tc, ta, tb] = [asTensor(c), asTensor(a), asTensor(b)]
    const shape = broadcastShapes(tc.shape, ta.shape, tb.shape)
    const [fc, fa, fb] = [tc, ta, tb].map((t) => flatData(broadcastView(t, shape)))
    const out = allocate(binaryDType(a, b), sizeOf(shape))
    for (let k = 0; k < out.length; k++) out[k] = fc[k] !== 0 ? fa[k] : fb[k]
    return fromData(out, shape)
  },
  (g, [c, a, b]) => {
    const mask = unwrap(c)
    return [null, sumLike(where(mask, g, 0), a), sumLike(where(mask, 0, g), b)]
  },
  {
    arity: 3,
    differentiable: [false, true, true],
    doc: { summary: 'Elementwise choice: a where the condition is non-zero, b elsewhere.' },
    test: {
      secondOrder: true,
      cases: (draw) => [{ inputs: [draw([2, 3], { lo: 0, hi: 1, integer: true }), draw([2, 3]), draw([3])] }],
    },
  },
)

/**
 * Elementwise choice: a where `condition` is non-zero, b elsewhere, all three broadcast. Differentiable in a and b;
 * the condition is treated as a constant.
 */
export function where(condition: Value, a: number, b: number): number
export function where(condition: Value, a: Traced, b: Value): Traced
export function where(condition: Value, a: Value, b: Traced): Traced
export function where(condition: Value, a: Tensor, b: Tensor | number): Tensor
export function where(condition: Value, a: number, b: Tensor): Tensor
export function where(condition: Value, a: Value, b: Value): Value
export function where(condition: Value, a: Value, b: Value): Value {
  return whereOp([unwrap(condition), a, b], undefined)
}

/**
 * Apply an arbitrary scalar function elementwise. It has no derivative: to make a function differentiable, define it
 * with `elementwise` and give its derivative.
 */
export function map(x: number, f: (v: number) => number): number
export function map(x: Tensor, f: (v: number) => number): Tensor
export function map(x: Value, f: (v: number) => number): Value
export function map(x: Value, f: (v: number) => number): Value {
  return defineOp<undefined>(
    'map',
    ([v]) => (typeof v === 'number' ? f(v) : unaryKernel(v, f, 'float64')),
    null,
  )([x], undefined)
}

/** Apply an arbitrary scalar function of two broadcast arguments elementwise. It has no derivative (see `map`). */
export function map2(a: number, b: number, f: (x: number, y: number) => number): number
export function map2(a: Tensor, b: Tensor | number, f: (x: number, y: number) => number): Tensor
export function map2(a: number, b: Tensor, f: (x: number, y: number) => number): Tensor
export function map2(a: Value, b: Value, f: (x: number, y: number) => number): Value
export function map2(a: Value, b: Value, f: (x: number, y: number) => number): Value {
  return defineOp<undefined>(
    'map2',
    ([x, y]) => (typeof x === 'number' && typeof y === 'number' ? f(x, y) : binaryKernel(x, y, f, 'float64')),
    null,
  )([a, b], undefined)
}
