/**
 * The tensor type, its storage and the strided iteration every other tensor function builds on.
 *
 * The layout follows NumPy's ndarray (Harris et al., 2020, "Array programming with NumPy", Nature 585): a flat typed
 * array, a shape, per-axis strides in elements and an offset. A view is a new header over the same data.
 */

/** Element type of a tensor's storage. */
export type DType = 'float64' | 'float32' | 'int32'

/** The typed array that backs a tensor. */
export type TensorData = Float64Array | Float32Array | Int32Array

/**
 * An n-dimensional array: `shape[k]` elements along axis k, stored in `data` at
 * `offset + Σ_k index[k] · strides[k]`. Immutable by convention: operations return new tensors, and views share `data`.
 */
export interface Tensor {
  /** Length of each axis; `[]` for a scalar tensor. */
  readonly shape: readonly number[]
  /** Step in `data`, in elements, for a unit step along each axis. Row-major (C order) by default. */
  readonly strides: readonly number[]
  /** Position in `data` of the element at index (0, …, 0). */
  readonly offset: number
  readonly dtype: DType
  readonly data: TensorData
}

/** A rank-1 tensor. */
export type Vector = Tensor
/** A rank-2 tensor. */
export type Matrix = Tensor

/** A number or a tensor: elementwise operations accept either and broadcast a number as a scalar. */
export type TensorLike = Tensor | number

/** Nested arrays of numbers, as accepted by `tensor` and returned by `toArray`. */
export type NestedArray = number | NestedArray[]

/** One axis or several; negative axes count from the end. */
export type Axis = number | readonly number[]

/** Allocate storage of a dtype. */
export function allocate(dtype: DType, size: number): TensorData {
  if (dtype === 'float64') return new Float64Array(size)
  if (dtype === 'float32') return new Float32Array(size)
  return new Int32Array(size)
}

/** The dtype of a typed array. */
export function dtypeOf(data: TensorData): DType {
  if (data instanceof Float64Array) return 'float64'
  if (data instanceof Float32Array) return 'float32'
  return 'int32'
}

/** Number of elements of a shape (1 for the scalar shape `[]`). */
export function sizeOf(shape: readonly number[]): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

/** Number of elements of a tensor. */
export function size(t: Tensor): number {
  return sizeOf(t.shape)
}

/** Row-major (C order) strides of a shape, in elements. */
export function rowMajorStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length)
  let step = 1
  for (let k = shape.length - 1; k >= 0; k--) {
    strides[k] = step
    step *= shape[k]
  }
  return strides
}

/** Check that a shape is a list of non-negative integers. */
export function checkShape(shape: readonly number[], where: string): void {
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0) throw new Error(`${where}: invalid shape [${shape.join(', ')}]`)
  }
}

/**
 * Wrap a typed array as a tensor without copying. `shape` defaults to `[data.length]`; strides are row-major.
 *
 * The caller hands over `data`: it must not be mutated afterwards, since tensors are immutable by convention.
 */
export function fromData(data: TensorData, shape: readonly number[] = [data.length]): Tensor {
  checkShape(shape, 'fromData')
  if (sizeOf(shape) !== data.length) {
    throw new Error(`fromData: ${data.length} values do not fill shape [${shape.join(', ')}]`)
  }
  return { shape: [...shape], strides: rowMajorStrides(shape), offset: 0, dtype: dtypeOf(data), data }
}

/** A tensor header over existing data; used internally to build views. */
export function view(t: Tensor, shape: readonly number[], strides: readonly number[], offset: number): Tensor {
  return { shape: [...shape], strides: [...strides], offset, dtype: t.dtype, data: t.data }
}

/** True when a value is a tensor (rather than a number or an array). */
export function isTensor(x: unknown): x is Tensor {
  return typeof x === 'object' && x !== null && 'shape' in x && 'strides' in x && 'data' in x
}

/**
 * True when the tensor's elements lie in row-major order in a single run of `data`, starting at `offset`. Axes of length
 * 1 are ignored, since their stride is never used.
 */
export function isContiguous(t: Tensor): boolean {
  let step = 1
  for (let k = t.shape.length - 1; k >= 0; k--) {
    if (t.shape[k] === 1) continue
    if (t.strides[k] !== step) return false
    step *= t.shape[k]
  }
  return true
}

/** Normalise an axis (negative counts from the end) and check its range. */
export function normaliseAxis(axis: number, rank: number, where: string): number {
  const a = axis < 0 ? axis + rank : axis
  if (!Number.isInteger(a) || a < 0 || a >= rank)
    throw new Error(`${where}: axis ${axis} is out of range for rank ${rank}`)
  return a
}

/** Normalise one or several axes; `undefined` or `null` means every axis. Duplicates are an error. */
export function normaliseAxes(axis: Axis | null | undefined, rank: number, where: string): number[] {
  if (axis === undefined || axis === null) return Array.from({ length: rank }, (_, k) => k)
  const list = typeof axis === 'number' ? [axis] : [...axis]
  const out = list.map((a) => normaliseAxis(a, rank, where))
  if (new Set(out).size !== out.length) throw new Error(`${where}: repeated axis in [${list.join(', ')}]`)
  return out.sort((a, b) => a - b)
}

/**
 * Visit every element of a strided layout in row-major order of `shape`, calling `body(offset, k)` with the element's
 * position in `data` and its row-major index k.
 */
export function forEachOffset(
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  body: (offset: number, k: number) => void,
): void {
  const n = sizeOf(shape)
  if (n === 0) return
  const rank = shape.length
  if (rank === 0) {
    body(offset, 0)
    return
  }
  // An odometer over all but the last axis; the last axis is a tight inner loop.
  const last = rank - 1
  const inner = shape[last]
  const step = strides[last]
  const index = new Array<number>(rank).fill(0)
  let base = offset
  for (let k = 0; k < n;) {
    let off = base
    for (let j = 0; j < inner; j++, k++, off += step) body(off, k)
    // Advance the odometer on axes last-1 … 0.
    let axis = last - 1
    while (axis >= 0) {
      index[axis]++
      base += strides[axis]
      if (index[axis] < shape[axis]) break
      base -= strides[axis] * shape[axis]
      index[axis] = 0
      axis--
    }
    if (axis < 0) break
  }
}

/**
 * Visit the elements of two strided layouts in lockstep over a shared `shape` (strides already broadcast), calling
 * `body(offsetA, offsetB, k)`.
 */
export function forEachOffset2(
  shape: readonly number[],
  stridesA: readonly number[],
  offsetA: number,
  stridesB: readonly number[],
  offsetB: number,
  body: (a: number, b: number, k: number) => void,
): void {
  const n = sizeOf(shape)
  if (n === 0) return
  const rank = shape.length
  if (rank === 0) {
    body(offsetA, offsetB, 0)
    return
  }
  const last = rank - 1
  const inner = shape[last]
  const stepA = stridesA[last]
  const stepB = stridesB[last]
  const index = new Array<number>(rank).fill(0)
  let baseA = offsetA
  let baseB = offsetB
  for (let k = 0; k < n;) {
    let a = baseA
    let b = baseB
    for (let j = 0; j < inner; j++, k++, a += stepA, b += stepB) body(a, b, k)
    let axis = last - 1
    while (axis >= 0) {
      index[axis]++
      baseA += stridesA[axis]
      baseB += stridesB[axis]
      if (index[axis] < shape[axis]) break
      baseA -= stridesA[axis] * shape[axis]
      baseB -= stridesB[axis] * shape[axis]
      index[axis] = 0
      axis--
    }
    if (axis < 0) break
  }
}

/** Copy a tensor's elements, in row-major order, into a new typed array of the given dtype. */
export function flatData(t: Tensor, dtype: DType = t.dtype): TensorData {
  const n = size(t)
  if (isContiguous(t) && dtype === t.dtype) return t.data.slice(t.offset, t.offset + n)
  const out = allocate(dtype, n)
  const src = t.data
  forEachOffset(t.shape, t.strides, t.offset, (off, k) => {
    out[k] = src[off]
  })
  return out
}

/** The elements in row-major order as a Float64Array; no copy when the tensor is contiguous float64 at offset 0. */
export function float64Data(t: Tensor): Float64Array {
  if (t.dtype === 'float64' && isContiguous(t) && t.offset === 0 && t.data.length === size(t)) {
    return t.data as Float64Array
  }
  return flatData(t, 'float64') as Float64Array
}

/**
 * The result dtype of combining two dtypes: int32 < float32 < float64, except that int32 with float32 gives float64
 * (float32 cannot hold every int32 exactly), as in NumPy.
 */
export function promote(a: DType, b: DType): DType {
  if (a === b) return a
  if (a === 'float64' || b === 'float64') return 'float64'
  return 'float64' // int32 with float32
}

/** The dtype a plain number takes next to a tensor of dtype `other`: NumPy's "weak scalar" rule. */
export function scalarDType(value: number, other: DType): DType {
  if (other === 'int32') return Number.isInteger(value) ? 'int32' : 'float64'
  return other
}

/** Format a shape for error messages. */
export function showShape(shape: readonly number[]): string {
  return `[${shape.join(', ')}]`
}
