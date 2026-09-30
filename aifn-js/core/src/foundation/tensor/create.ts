/** Constructors and converters between tensors and plain arrays (the chart boundary). */

import { AifnError, ShapeError } from 'aifn/foundation/errors'
import {
  allocate,
  checkShape,
  flatData,
  forEachOffset,
  fromData,
  isTensor,
  showShape,
  size,
  sizeOf,
  type DType,
  type NestedArray,
  type Tensor,
} from './core'

/**
 * The shape of nested arrays (checked to be rectangular) or of a tensor. A number has shape `[]`.
 *
 * @example shapeOf([[1, 2, 3], [4, 5, 6]]) // [2, 3]
 */
export function shapeOf(x: NestedArray | ArrayLike<number> | Tensor): number[] {
  if (isTensor(x)) return [...x.shape]
  if (typeof x === 'number') return []
  const shape: number[] = []
  let probe: unknown = x
  while (typeof probe === 'object' && probe !== null && 'length' in probe) {
    const list = probe as ArrayLike<unknown>
    shape.push(list.length)
    if (list.length === 0) break
    probe = list[0]
  }
  // Check that every branch has the same shape, so that ragged input is an error rather than silently truncated.
  const check = (node: unknown, depth: number): void => {
    if (depth === shape.length) {
      if (typeof node !== 'number') throw new ShapeError('shapeOf', 'shapeOf: ragged or non-numeric nested array')
      return
    }
    const list = node as ArrayLike<unknown>
    if (typeof node !== 'object' || node === null || list.length !== shape[depth]) {
      throw new ShapeError(
        'shapeOf',
        `shapeOf: ragged nested array (expected length ${shape[depth]} at depth ${depth})`,
      )
    }
    for (let i = 0; i < list.length; i++) check(list[i], depth + 1)
  }
  check(x, 0)
  return shape
}

/**
 * A tensor from nested arrays, a flat array (with `shape`) or a typed array. Always copies.
 *
 * @param values nested `number[]…`, or a flat `ArrayLike<number>` read in row-major order when `shape` is given
 * @param shape the shape to give flat values; defaults to the nested structure
 * @param dtype storage type, default `float64`; values are converted as a typed array converts them (int32 truncates)
 * @example tensor([[1, 2], [3, 4]]) // shape [2, 2]
 * @example tensor([1, 2, 3, 4, 5, 6], [2, 3])
 */
export function tensor(
  values: NestedArray | ArrayLike<number>,
  shape?: readonly number[],
  dtype: DType = 'float64',
): Tensor {
  if (typeof values === 'number') {
    if (shape && sizeOf(shape) !== 1)
      throw new ShapeError('tensor', `tensor: one value cannot fill shape ${showShape(shape)}`)
    return fromData(allocate(dtype, 1).fill(values), shape ?? [])
  }
  const nested = shapeOf(values as NestedArray)
  const flat = allocate(dtype, sizeOf(nested))
  let k = 0
  const fill = (node: unknown): void => {
    if (typeof node === 'number') flat[k++] = node
    else for (let i = 0; i < (node as ArrayLike<unknown>).length; i++) fill((node as ArrayLike<unknown>)[i])
  }
  fill(values)
  if (shape) {
    checkShape(shape, 'tensor')
    if (sizeOf(shape) !== flat.length) {
      throw new ShapeError('tensor', `tensor: ${flat.length} values do not fill shape ${showShape(shape)}`)
    }
    return fromData(flat, shape)
  }
  return fromData(flat, nested)
}

/** A scalar (rank-0) tensor. */
export function scalar(value: number, dtype: DType = 'float64'): Tensor {
  return tensor(value, [], dtype)
}

/** A tensor of the given shape filled with `value`. */
export function full(shape: readonly number[], value: number, dtype: DType = 'float64'): Tensor {
  checkShape(shape, 'full')
  return fromData(allocate(dtype, sizeOf(shape)).fill(value), shape)
}

/** A tensor of zeros. */
export function zeros(shape: readonly number[], dtype: DType = 'float64'): Tensor {
  return full(shape, 0, dtype)
}

/** A tensor of ones. */
export function ones(shape: readonly number[], dtype: DType = 'float64'): Tensor {
  return full(shape, 1, dtype)
}

/** The n×m identity-like matrix: ones on the diagonal `k` (0 main, positive above, negative below). m defaults to n. */
export function eye(n: number, m: number = n, k = 0, dtype: DType = 'float64'): Tensor {
  const data = allocate(dtype, n * m)
  for (let i = 0; i < n; i++) {
    const j = i + k
    if (j >= 0 && j < m) data[i * m + j] = 1
  }
  return fromData(data, [n, m])
}

/**
 * Evenly spaced values in [start, stop) with the given step, as `np.arange`: `arange(5)` is 0…4 and
 * `arange(2, 3, 0.25)` is 2, 2.25, 2.5, 2.75. The length is ⌈(stop − start) / step⌉.
 */
export function arange(start: number, stop?: number, step = 1, dtype: DType = 'float64'): Tensor {
  if (stop === undefined) {
    stop = start
    start = 0
  }
  if (step === 0 || !Number.isFinite(step)) throw new AifnError('arange', 'arange: step must be finite and non-zero')
  const n = Math.max(0, Math.ceil((stop - start) / step))
  const data = allocate(dtype, n)
  for (let i = 0; i < n; i++) data[i] = start + i * step
  return fromData(data)
}

/**
 * `num` evenly spaced values from `start` to `stop`, as `np.linspace`. With `endpoint` (default) the last value is
 * exactly `stop`; without it the values stop one step short.
 */
export function linspace(start: number, stop: number, num = 50, endpoint = true): Tensor {
  if (!Number.isInteger(num) || num < 0) throw new AifnError('linspace', 'linspace: num must be a non-negative integer')
  const data = new Float64Array(num)
  const div = endpoint ? num - 1 : num
  const step = div > 0 ? (stop - start) / div : 0
  for (let i = 0; i < num; i++) data[i] = start + i * step
  if (endpoint && num > 1) data[num - 1] = stop
  return fromData(data)
}

/** A matrix from rows; every row must have the same length. */
export function fromRows(rows: readonly (readonly number[] | ArrayLike<number>)[], dtype: DType = 'float64'): Tensor {
  const m = rows.length
  const n = m > 0 ? rows[0].length : 0
  const data = allocate(dtype, m * n)
  for (let i = 0; i < m; i++) {
    const row = rows[i]
    if (row.length !== n) throw new ShapeError('fromRows', `fromRows: row ${i} has length ${row.length}, expected ${n}`)
    for (let j = 0; j < n; j++) data[i * n + j] = row[j]
  }
  return fromData(data, [m, n])
}

/** The elements in row-major order as a plain array. */
export function toFlat(t: Tensor): number[] {
  return Array.from(flatData(t))
}

/** The tensor as nested plain arrays (a number for a scalar tensor). */
export function toArray(t: Tensor): NestedArray {
  const flat = flatData(t)
  if (t.shape.length === 0) return flat[0]
  let k = 0
  const build = (axis: number): NestedArray => {
    const n = t.shape[axis]
    const out: NestedArray[] = new Array(n)
    for (let i = 0; i < n; i++) out[i] = axis === t.shape.length - 1 ? flat[k++] : build(axis + 1)
    return out
  }
  return build(0)
}

/** A matrix as `number[][]` rows (a vector becomes one row). */
export function toRows(t: Tensor): number[][] {
  if (t.shape.length === 1) return [toFlat(t)]
  if (t.shape.length !== 2) throw new ShapeError('toRows', `toRows: expected a matrix, got shape ${showShape(t.shape)}`)
  const [m, n] = t.shape
  const flat = flatData(t)
  return Array.from({ length: m }, (_, i) => Array.from(flat.subarray(i * n, (i + 1) * n)))
}

/** The single element of a tensor of size 1 (any rank). */
export function item(t: Tensor): number {
  if (size(t) !== 1)
    throw new ShapeError('item', `item: tensor of shape ${showShape(t.shape)} has ${size(t)} elements, not 1`)
  let value = 0
  forEachOffset(t.shape, t.strides, t.offset, (off) => {
    value = t.data[off]
  })
  return value
}

/** A contiguous row-major copy of a tensor, optionally converted to another dtype. */
export function copy(t: Tensor, dtype: DType = t.dtype): Tensor {
  return fromData(flatData(t, dtype), t.shape)
}

/** The tensor converted to another dtype (a copy; int32 truncates towards zero). */
export function astype(t: Tensor, dtype: DType): Tensor {
  return copy(t, dtype)
}
