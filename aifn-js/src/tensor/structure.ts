/**
 * Structural primitives: reshaping, permuting, slicing, joining and broadcasting. Views share data where possible.
 * Each has a derivative rule: the cotangent is moved back through the inverse rearrangement (Griewank and Walther,
 * 2008, ch. 3), e.g. a slice's cotangent is scattered into zeros of the input's shape.
 */

import { flatData, fromData, normaliseAxis, showShape, size, sizeOf, view, type Axis, type Tensor } from './core'
import { zeros } from './create'
import { mul } from './elementwise'
import { sumToKernel } from './kernels'
import { defineOp, sumLike, type NumberResult, type Raw, type TensorResult } from './primitive'
import { unwrap, type Value } from './tape'
import {
  broadcastView,
  checkPermutation,
  concatRaw,
  expandDimsView,
  flatIndex,
  getRaw,
  permuteView,
  reshapeView,
  sliceView,
  squeezedAxes,
  squeezeView,
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
  'reshape',
  ([x], shape) => reshapeView(asTensor(x), shape),
  (g, [x]) => [typeof unwrap(x) === 'number' ? sumLike(g, x) : reshape(g, shapeOfValue(x))],
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
  'broadcastTo',
  ([x], shape) => broadcastView(asTensor(x), shape),
  (g, [x]) => [sumLike(g, x)],
)

/** `x` broadcast to `shape` by NumPy rules (a view with stride 0 along repeated axes). */
export function broadcastTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return broadcastToOp([x], shape) as TensorResult<X>
}

const sumToOp = defineOp<readonly number[]>(
  'sumTo',
  ([x], shape) => sumToKernel(asTensor(x), shape),
  (g, [x]) => [broadcastTo(g, shapeOfValue(x))],
)

/**
 * Sum `x` down to `shape`, the adjoint of broadcasting `shape` up to `x`'s shape: leading axes are summed away and
 * axes where `shape` has length 1 are summed keeping length 1.
 */
export function sumTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return sumToOp([x], shape) as TensorResult<X>
}

const permuteOp = defineOp<readonly number[]>(
  'permute',
  ([x], axes) => permuteView(asTensor(x), checkPermutation(axes, asTensor(x).shape.length)),
  (g, [x], _y, axes) => {
    const order = checkPermutation(axes, shapeOfValue(x).length)
    const inverse = new Array<number>(order.length)
    order.forEach((a, k) => (inverse[a] = k))
    return [permute(g, inverse)]
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

const squeezeOp = defineOp<Axis | undefined>(
  'squeeze',
  ([x], axis) => {
    const t = asTensor(x)
    return squeezeView(t, squeezedAxes(t.shape, axis))
  },
  (g, [x]) => [reshape(g, shapeOfValue(x))],
)

/** A view without axes of length 1: all of them, or only `axis` (each of which must have length 1). */
export function squeeze<X extends Value>(x: X, axis?: Axis): TensorResult<X> {
  return squeezeOp([x], axis) as TensorResult<X>
}

const expandDimsOp = defineOp<number>(
  'expandDims',
  ([x], axis) => expandDimsView(asTensor(x), axis),
  (g, [x]) => [typeof unwrap(x) === 'number' ? sumLike(g, x) : reshape(g, shapeOfValue(x))],
)

/** A view with a new axis of length 1 inserted at position `axis` of the result (negative counts from the end). */
export function expandDims<X extends Value>(x: X, axis: number): TensorResult<X> {
  return expandDimsOp([x], axis) as TensorResult<X>
}

type Scatter = { shape: readonly number[]; specs: readonly SliceSpec[] }

// The adjoint of slicing: zeros of the input's shape with the cotangent written into the sliced positions.
const scatterSliceOp = defineOp<Scatter>(
  'scatterSlice',
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
  'slice',
  ([x], specs) => sliceView(asTensor(x), specs),
  (g, [x], _y, specs) => [scatterSliceOp([g], { shape: shapeOfValue(x), specs })],
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

/** A one-hot tensor: 1 at `index`, 0 elsewhere. */
function oneHot(shape: readonly number[], index: readonly number[], hot = 1, cold = 0): Tensor {
  const out = new Float64Array(sizeOf(shape)).fill(cold)
  out[flatIndex(shape, index, 'oneHot')] = hot
  return fromData(out, shape)
}

const getOp = defineOp<readonly number[]>(
  'get',
  ([x], index) => getRaw(asTensor(x), index),
  (g, [x], _y, index) => [mul(g, oneHot(shapeOfValue(x), index))],
)

/** The element at a full multi-index; negative indices count from the end. `get(m, i, j)` reads m[i, j]. */
export function get<X extends Value>(x: X, ...index: number[]): NumberResult<X> {
  return getOp([x], index) as NumberResult<X>
}

const setOp = defineOp<readonly number[]>(
  'set',
  ([x, value], index) => {
    const t = asTensor(x)
    const out = fromData(flatData(t), t.shape)
    out.data[flatIndex(t.shape, index, 'set')] = value as number
    return out
  },
  (g, [x], _y, index) => [mul(g, oneHot(shapeOfValue(x), index, 0, 1)), get(g, ...index)],
)

/** A copy of `x` with the element at `index` replaced by `value`; `x` is unchanged. */
export function set<X extends Value>(x: X, index: readonly number[], value: Value): TensorResult<X> {
  const raw = unwrap(value)
  if (typeof raw !== 'number') throw new Error('set: value must be a number')
  return setOp([x, value], index) as TensorResult<X>
}

const concatOp = defineOp<number>(
  'concat',
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
  if (xs.length === 0) throw new Error('stack: nothing to join')
  const first = shapeOfValue(xs[0])
  for (const x of xs) {
    const s = shapeOfValue(x)
    if (s.length !== first.length || s.some((d, k) => d !== first[k])) {
      throw new Error(`stack: shape ${showShape(s)} differs from ${showShape(first)}`)
    }
  }
  const a = normaliseAxis(axis, first.length + 1, 'stack')
  return concat(
    xs.map((x) => expandDims(x, a)),
    a,
  )
}

const diagonalOp = defineOp<undefined>(
  'diagonal',
  ([x]) => {
    const t = asTensor(x)
    if (t.shape.length !== 2) throw new Error(`diagonal: expected a matrix, got shape ${showShape(t.shape)}`)
    const k = Math.min(t.shape[0], t.shape[1])
    return fromData(flatData(view(t, [k], [t.strides[0] + t.strides[1]], t.offset)), [k])
  },
  (g, [x]) => [embedDiagonalOp([g], shapeOfValue(x))],
)

// The adjoint of `diagonal`: a matrix of the given shape, zero off the diagonal.
const embedDiagonalOp = defineOp<readonly number[]>(
  'embedDiagonal',
  ([v], shape) => {
    const d = flatData(asTensor(v), 'float64')
    const [m, n] = shape
    const out = new Float64Array(m * n)
    for (let i = 0; i < Math.min(m, n, d.length); i++) out[i * n + i] = d[i]
    return fromData(out, [m, n])
  },
  (g) => [diagonal(g)],
)

/** The main diagonal of a matrix, as a vector of length min(m, n). */
export function diagonal<X extends Value>(x: X): TensorResult<X> {
  return diagonalOp([x], undefined) as TensorResult<X>
}

/** A square matrix with `v` on its diagonal and zeros elsewhere. */
export function diag<X extends Value>(v: X): TensorResult<X> {
  const n = shapeOfValue(v)[0]
  return embedDiagonalOp([v], [n, n]) as TensorResult<X>
}
