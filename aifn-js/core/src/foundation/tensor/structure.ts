/**
 * Structural operations: reshaping, permuting, slicing, joining and broadcasting. Views share data where possible.
 *
 * The primitives are `reshape`, `broadcastTo`, `sumTo`, `permute`, `slice` (with its adjoint `scatterSlice`) and
 * `concat`; each has a derivative rule that moves the cotangent back through the inverse rearrangement (Griewank and
 * Walther, 2008, ch. 3), e.g. a slice's cotangent is scattered into zeros of the input's shape. Everything else here
 * is a composition of primitives (design K §3.4, T8), so it has no rule of its own to test: `flatten`, `squeeze` and
 * `expandDims` are reshapes, `get` sums a one-element slice, `set` is a `where` against a one-hot mask, `stack` is
 * `expandDims` and `concat`, and `diagonal`/`diag` are `gather`/`scatterAdd` at the diagonal's flat indices.
 */

import { AifnError, ShapeError } from 'aifn/foundation/errors'
import { flatData, fromData, normaliseAxis, showShape, size, sizeOf, type Axis, type Tensor } from './core'
import { zeros } from './create'
import { where } from './elementwise'
import { sumToKernel } from './kernels'
import { defineOp, sumLike, type NumberResult, type Raw, type TensorResult } from './primitive'
import { unwrap, type Value } from './tape'
import { gather, scatterAdd } from './gather'
import { sum } from './reduce'
import {
  broadcastView,
  checkPermutation,
  concatRaw,
  flatIndex,
  permuteView,
  reshapeView,
  sliceView,
  squeezedAxes,
  type SliceSpec,
} from './views'

/** A raw input as a tensor; numbers become scalar tensors. */
function asTensor(x: Raw): Tensor {
  return typeof x === 'number' ? fromData(new Float64Array([x]), []) : x
}

/** The shape of a value (`[]` for a number). */
export function shapeOfValue(x: Value): number[] {
  const raw = unwrap(x)
  return typeof raw === 'number' ? [] : [...raw.shape]
}

const reshapeOp = defineOp<readonly number[]>(
  'foundation/tensor/reshape',
  ([x], shape) => reshapeView(asTensor(x), shape),
  (g, [x]) => [typeof unwrap(x) === 'number' ? sumLike(g, x) : reshape(g, shapeOfValue(x))],
  {
    arity: 1,
    doc: { summary: 'The same elements with a new shape.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        { inputs: [draw([2, 3])], params: [3, 2] },
        { inputs: [draw([2, 3])], params: [-1] },
      ],
    },
  },
)

/**
 * The same elements with a new shape; one entry may be -1 (inferred). A view when the tensor is contiguous, otherwise
 * a copy.
 */
export function reshape<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return reshapeOp([x], shape) as TensorResult<X>
}

/** A rank-1 tensor of the elements in row-major order. */
export function flatten<X extends Value>(x: X): TensorResult<X> {
  return reshape(x, [-1])
}

const broadcastToOp = defineOp<readonly number[]>(
  'foundation/tensor/broadcastTo',
  ([x], shape) => broadcastView(asTensor(x), shape),
  (g, [x]) => [sumLike(g, x)],
  {
    arity: 1,
    doc: { summary: 'Broadcast to a shape by NumPy rules.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        { inputs: [draw([3])], params: [2, 3] },
        { inputs: [draw([2, 1])], params: [2, 3] },
      ],
    },
  },
)

/** `x` broadcast to `shape` by NumPy rules (a view with stride 0 along repeated axes). */
export function broadcastTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return broadcastToOp([x], shape) as TensorResult<X>
}

const sumToOp = defineOp<readonly number[]>(
  'foundation/tensor/sumTo',
  ([x], shape) => sumToKernel(asTensor(x), shape),
  (g, [x]) => [broadcastTo(g, shapeOfValue(x))],
  {
    arity: 1,
    doc: { summary: 'Sum down to a shape: the adjoint of broadcasting.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        { inputs: [draw([2, 3])], params: [3] },
        { inputs: [draw([2, 3])], params: [2, 1] },
      ],
    },
  },
)

/**
 * Sum `x` down to `shape`, the adjoint of broadcasting `shape` up to `x`'s shape: leading axes are summed away and
 * axes where `shape` has length 1 are summed keeping length 1.
 */
export function sumTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return sumToOp([x], shape) as TensorResult<X>
}

const permuteOp = defineOp<readonly number[]>(
  'foundation/tensor/permute',
  ([x], axes) => permuteView(asTensor(x), checkPermutation(axes, asTensor(x).shape.length)),
  (g, [x], _y, axes) => {
    const order = checkPermutation(axes, shapeOfValue(x).length)
    const inverse = new Array<number>(order.length)
    order.forEach((a, k) => (inverse[a] = k))
    return [permute(g, inverse)]
  },
  {
    arity: 1,
    doc: { summary: 'Reorder the axes.' },
    test: { secondOrder: true, cases: (draw) => [{ inputs: [draw([2, 3, 4])], params: [2, 0, 1] }] },
  },
)

/** A view with the axes reordered: axis k of the result is axis `axes[k]` of `x`. */
export function permute<X extends Value>(x: X, axes: readonly number[]): TensorResult<X> {
  return permuteOp([x], axes) as TensorResult<X>
}

/** A view with the axes reversed (the matrix transpose for rank 2), or permuted by `axes` when given. */
export function transpose<X extends Value>(x: X, axes?: readonly number[]): TensorResult<X> {
  const rank = shapeOfValue(x).length
  return permute(x, axes ?? Array.from({ length: rank }, (_, k) => rank - 1 - k))
}

/** A view without axes of length 1: all of them, or only `axis` (each of which must have length 1). A reshape. */
export function squeeze<X extends Value>(x: X, axis?: Axis): TensorResult<X> {
  const shape = shapeOfValue(x)
  const drop = squeezedAxes(shape, axis)
  return reshape(
    x,
    shape.filter((_, k) => !drop.includes(k)),
  )
}

/**
 * A view with a new axis of length 1 inserted at position `axis` of the result (negative counts from the end). A
 * reshape.
 */
export function expandDims<X extends Value>(x: X, axis: number): TensorResult<X> {
  const shape = shapeOfValue(x)
  const a = normaliseAxis(axis, shape.length + 1, 'expandDims')
  return reshape(x, [...shape.slice(0, a), 1, ...shape.slice(a)])
}

type Scatter = { shape: readonly number[]; specs: readonly SliceSpec[] }

// The adjoint of slicing: zeros of the input's shape with the cotangent written into the sliced positions.
const scatterSliceOp = defineOp<Scatter>(
  'foundation/tensor/scatterSlice',
  ([g], { shape, specs }) => {
    const out = zeros(shape)
    const target = sliceView(out, specs)
    const src = asTensor(g)
    const values = flatData(src, 'float64')
    let k = 0
    forEachTarget(target, (off) => (out.data[off] = values[k++]))
    return out
  },
  (g, _inputs, _y, { specs }) => [slice(g, ...specs)],
  {
    arity: 1,
    doc: { summary: 'Zeros with values written into a slice: the adjoint of slicing.' },
    test: {
      secondOrder: true,
      cases: (draw) => [{ inputs: [draw([2, 3])], params: { shape: [4, 3], specs: [[0, 4, 2]] } }],
    },
  },
)

/** Visit the data offsets of a view in row-major order. */
function forEachTarget(t: Tensor, body: (off: number) => void): void {
  const n = size(t)
  const index = new Array<number>(t.shape.length).fill(0)
  for (let k = 0; k < n; k++) {
    let off = t.offset
    for (let a = 0; a < index.length; a++) off += index[a] * t.strides[a]
    body(off)
    for (let a = index.length - 1; a >= 0; a--) {
      if (++index[a] < t.shape[a]) break
      index[a] = 0
    }
  }
}

const sliceOp = defineOp<readonly SliceSpec[]>(
  'foundation/tensor/slice',
  ([x], specs) => sliceView(asTensor(x), specs),
  (g, [x], _y, specs) => [scatterSliceOp([g], { shape: shapeOfValue(x), specs })],
  {
    arity: 1,
    doc: { summary: 'Basic indexing: a view of part of a tensor.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        { inputs: [draw([4, 3])], params: [[1, 3], null] },
        { inputs: [draw([5])], params: [[null, null, -1]] },
        { inputs: [draw([3, 4])], params: [1] },
      ],
    },
  },
)

/**
 * A view selecting part of a tensor, one spec per leading axis (missing trailing specs keep their axes whole), as
 * NumPy basic indexing. An integer picks a position and drops the axis, `null` keeps the axis, and
 * `[start, stop, step]` is Python's `start:stop:step` (negative values count from the end). Never copies.
 *
 * @example slice(m, 0) // the first row of a matrix
 * @example slice(m, null, [0, null, 2]) // every second column
 * @example slice(v, [null, null, -1]) // v reversed
 */
export function slice<X extends Value>(x: X, ...specs: SliceSpec[]): TensorResult<X> {
  return sliceOp([x], specs) as TensorResult<X>
}

/** A one-hot mask: 1 at `index`, 0 elsewhere. */
function oneHot(shape: readonly number[], index: readonly number[]): Tensor {
  const out = new Float64Array(sizeOf(shape))
  out[flatIndex(shape, index, 'oneHot')] = 1
  return fromData(out, shape)
}

/**
 * The element at a full multi-index; negative indices count from the end. `get(m, i, j)` reads m[i, j]. A composition:
 * the sum of the one-element slice, so its derivative scatters the cotangent back to that element.
 */
export function get<X extends Value>(x: X, ...index: number[]): NumberResult<X> {
  const shape = shapeOfValue(x)
  flatIndex(shape, index, 'get')
  return sum(slice(x, ...index)) as NumberResult<X>
}

/**
 * A copy of `x` with the element at `index` replaced by `value` (a number, possibly traced); `x` is unchanged. A
 * composition: `where` against a one-hot mask, so it is differentiable in both `x` and `value`.
 */
export function set<X extends Value>(x: X, index: readonly number[], value: Value): TensorResult<X> {
  if (shapeOfValue(value).length !== 0) throw new AifnError('set', 'set: value must be a number')
  return where(oneHot(shapeOfValue(x), index), value, x) as TensorResult<X>
}

const concatOp = defineOp<number>(
  'foundation/tensor/concat',
  (xs, axis) => concatRaw(xs.map(asTensor), axis),
  (g, xs, _y, axis) => {
    const rank = shapeOfValue(xs[0]).length
    const a = normaliseAxis(axis, rank, 'concat')
    let start = 0
    return xs.map((x) => {
      const n = shapeOfValue(x)[a]
      const specs: SliceSpec[] = Array.from({ length: a + 1 }, (_, k) => (k === a ? [start, start + n] : null))
      start += n
      return slice(g, ...specs)
    })
  },
  {
    doc: { summary: 'Join tensors along an existing axis.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        { inputs: [draw([2, 3]), draw([1, 3])], params: 0 },
        { inputs: [draw([2, 3]), draw([2, 2])], params: 1 },
      ],
    },
  },
)

/** Join tensors along an existing axis; the other axes must match. Copies; the dtype is the common promotion. */
export function concat(xs: readonly Tensor[], axis?: number): Tensor
export function concat(xs: readonly Value[], axis?: number): Value
export function concat(xs: readonly Value[], axis = 0): Value {
  return concatOp(xs, axis)
}

/** Join tensors of equal shape along a new axis at position `axis`. Copies. */
export function stack(xs: readonly Tensor[], axis?: number): Tensor
export function stack(xs: readonly Value[], axis?: number): Value
export function stack(xs: readonly Value[], axis = 0): Value {
  if (xs.length === 0) throw new ShapeError('stack', 'stack: nothing to join')
  const first = shapeOfValue(xs[0])
  for (const x of xs) {
    const s = shapeOfValue(x)
    if (s.length !== first.length || s.some((d, k) => d !== first[k])) {
      throw new ShapeError('stack', `stack: shape ${showShape(s)} differs from ${showShape(first)}`, [s, first])
    }
  }
  const a = normaliseAxis(axis, first.length + 1, 'stack')
  return concat(
    xs.map((x) => expandDims(x, a)),
    a,
  )
}

/** The flat row-major indices of the main diagonal of an m × n matrix. */
function diagonalIndices(m: number, n: number): Int32Array {
  return Int32Array.from({ length: Math.min(m, n) }, (_, i) => i * n + i)
}

/** The main diagonal of a matrix, as a vector of length min(m, n). A `gather` at the diagonal's flat indices. */
export function diagonal<X extends Value>(x: X): TensorResult<X> {
  const shape = shapeOfValue(x)
  if (shape.length !== 2)
    throw new ShapeError('diagonal', `diagonal: expected a matrix, got shape ${showShape(shape)}`, [shape])
  const indices = diagonalIndices(shape[0], shape[1])
  return gather(x, indices, [indices.length]) as TensorResult<X>
}

/** A square matrix with `v` on its diagonal and zeros elsewhere. A `scatterAdd` at the diagonal's flat indices. */
export function diag<X extends Value>(v: X): TensorResult<X> {
  const shape = shapeOfValue(v)
  if (shape.length !== 1)
    throw new ShapeError('diag', `diag: expected a vector, got shape ${showShape(shape)}`, [shape])
  const n = shape[0]
  return scatterAdd(v, diagonalIndices(n, n), [n, n]) as TensorResult<X>
}
