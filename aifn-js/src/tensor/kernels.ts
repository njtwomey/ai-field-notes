/**
 * Raw computational kernels on plain tensors: elementwise maps with broadcasting, reductions and batched matrix
 * products. Primitives call these for their forward values. Contiguous float64 inputs take tight loops over `data`.
 */

import {
  allocate,
  forEachOffset,
  forEachOffset2,
  fromData,
  isContiguous,
  normaliseAxes,
  promote,
  scalarDType,
  showShape,
  size,
  sizeOf,
  type Axis,
  type DType,
  type Tensor,
} from './core'
import { broadcastShapes, broadcastView, permuteView } from './views'

/** Apply `f` to every element, writing a new tensor of dtype `dtype`. */
export function unaryKernel(x: Tensor, f: (v: number) => number, dtype: DType): Tensor {
  const n = size(x)
  const out = allocate(dtype, n)
  const src = x.data
  if (isContiguous(x)) {
    const o = x.offset
    for (let k = 0; k < n; k++) out[k] = f(src[o + k])
  } else {
    forEachOffset(x.shape, x.strides, x.offset, (off, k) => {
      out[k] = f(src[off])
    })
  }
  return fromData(out, x.shape)
}

/**
 * Apply `f` elementwise to two broadcast operands, writing dtype `dtype`. A plain number broadcasts as a scalar.
 */
export function binaryKernel(
  a: Tensor | number,
  b: Tensor | number,
  f: (x: number, y: number) => number,
  dtype: DType,
): Tensor {
  if (typeof a === 'number' && typeof b === 'number') return fromData(allocate(dtype, 1).fill(f(a, b)), [])
  if (typeof a === 'number') {
    const bt = b as Tensor
    return unaryKernel(bt, (y) => f(a, y), dtype)
  }
  if (typeof b === 'number') return unaryKernel(a, (x) => f(x, b), dtype)
  const shape = broadcastShapes(a.shape, b.shape)
  const n = sizeOf(shape)
  const out = allocate(dtype, n)
  const da = a.data
  const db = b.data
  const same = a.shape.length === b.shape.length && a.shape.every((d, k) => d === b.shape[k])
  if (same && isContiguous(a) && isContiguous(b)) {
    const oa = a.offset
    const ob = b.offset
    for (let k = 0; k < n; k++) out[k] = f(da[oa + k], db[ob + k])
    return fromData(out, shape)
  }
  const va = broadcastView(a, shape)
  const vb = broadcastView(b, shape)
  forEachOffset2(shape, va.strides, va.offset, vb.strides, vb.offset, (i, j, k) => {
    out[k] = f(da[i], db[j])
  })
  return fromData(out, shape)
}

/**
 * Apply `f` elementwise to three broadcast operands, writing dtype `dtype`. Plain numbers broadcast as scalars; three
 * numbers give a rank-0 tensor.
 */
export function ternaryKernel(
  a: Tensor | number,
  b: Tensor | number,
  c: Tensor | number,
  f: (x: number, y: number, z: number) => number,
  dtype: DType,
): Tensor {
  const operands = [a, b, c]
  const shape = broadcastShapes(...operands.map((v) => (typeof v === 'number' ? [] : v.shape)))
  const n = sizeOf(shape)
  const out = allocate(dtype, n)
  // Each operand as a flat array over the broadcast shape (a stride-0 view, copied once) or a constant.
  const flat = operands.map((v) => {
    if (typeof v === 'number') return null
    const view = broadcastView(v, shape)
    const values = new Float64Array(n)
    const src = v.data
    forEachOffset(view.shape, view.strides, view.offset, (off, k) => {
      values[k] = src[off]
    })
    return values
  })
  const [fa, fb, fc] = flat
  const [ca, cb, cc] = operands.map((v) => (typeof v === 'number' ? v : 0))
  for (let k = 0; k < n; k++) out[k] = f(fa ? fa[k] : ca, fb ? fb[k] : cb, fc ? fc[k] : cc)
  return fromData(out, shape)
}

/** The dtype of a binary operation's result, before any operation-specific rule (e.g. division gives floats). */
export function binaryDType(a: Tensor | number, b: Tensor | number): DType {
  if (typeof a === 'number' && typeof b === 'number') return 'float64'
  if (typeof a === 'number') return scalarDType(a, (b as Tensor).dtype)
  if (typeof b === 'number') return scalarDType(b, a.dtype)
  return promote(a.dtype, b.dtype)
}

/** Shape bookkeeping for a reduction over `axes` (sorted, normalised). */
export function reducedShape(shape: readonly number[], axes: readonly number[], keepDims: boolean): number[] {
  return keepDims ? shape.map((d, k) => (axes.includes(k) ? 1 : d)) : shape.filter((_, k) => !axes.includes(k))
}

/**
 * Reduce over `axes`: for every output position, `fn` receives the reduced elements (in row-major order of the
 * reduced axes) as a Float64Array and returns one number. Returns a tensor of dtype `dtype`.
 */
export function reduceKernel(
  x: Tensor,
  axis: Axis | null | undefined,
  keepDims: boolean,
  fn: (values: Float64Array) => number,
  dtype: DType = 'float64',
): Tensor {
  const axes = normaliseAxes(axis, x.shape.length, 'reduce')
  const outShape = reducedShape(x.shape, axes, keepDims)
  const kept = x.shape.map((_, k) => k).filter((k) => !axes.includes(k))
  const groups = sizeOf(kept.map((k) => x.shape[k]))
  const width = sizeOf(axes.map((k) => x.shape[k]))
  // Move the kept axes first: in row-major order, each run of `width` elements then belongs to one output.
  const moved = permuteView(x, [...kept, ...axes])
  let values: Float64Array
  if (x.dtype === 'float64' && isContiguous(moved)) {
    values = (x.data as Float64Array).subarray(moved.offset, moved.offset + groups * width)
  } else {
    values = new Float64Array(groups * width)
    const src = x.data
    forEachOffset(moved.shape, moved.strides, moved.offset, (off, k) => {
      values[k] = src[off]
    })
  }
  const out = allocate(dtype, groups)
  for (let g = 0; g < groups; g++) out[g] = fn(values.subarray(g * width, (g + 1) * width))
  return fromData(out, outShape)
}

/**
 * Sum `x` down to `shape`, the inverse of broadcasting `shape` up to `x.shape`: leading axes are summed away and axes
 * where `shape` has length 1 are summed with the length kept.
 */
export function sumToKernel(x: Tensor, shape: readonly number[]): Tensor {
  const lead = x.shape.length - shape.length
  if (lead < 0) throw new Error(`sumTo: cannot reduce ${showShape(x.shape)} to ${showShape(shape)}`)
  const axes: number[] = []
  for (let k = 0; k < x.shape.length; k++) {
    if (k < lead) axes.push(k)
    else if (shape[k - lead] === 1 && x.shape[k] !== 1) axes.push(k)
    else if (shape[k - lead] !== x.shape[k]) {
      throw new Error(`sumTo: cannot reduce ${showShape(x.shape)} to ${showShape(shape)}`)
    }
  }
  const summed = reduceKernel(x, axes, true, sumValues, x.dtype === 'int32' ? 'int32' : 'float64')
  return fromData(summed.data, shape)
}

/** Sum of a run of values. */
export function sumValues(v: Float64Array): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i]
  return s
}

/**
 * Batched matrix product for rank ≥ 2 operands: the last two axes multiply as matrices ([…, m, k] × […, k, n]) and the
 * leading (batch) axes broadcast.
 */
export function matmulKernel(a: Tensor, b: Tensor): Tensor {
  const ra = a.shape.length
  const rb = b.shape.length
  if (ra < 2 || rb < 2) throw new Error('matmul: kernel needs rank ≥ 2 operands')
  const [m, k] = a.shape.slice(-2)
  const [k2, n] = b.shape.slice(-2)
  if (k !== k2)
    throw new Error(`matmul: shapes ${showShape(a.shape)} and ${showShape(b.shape)} do not align (${k} ≠ ${k2})`)
  const batch = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2))
  const va = broadcastView(a, [...batch, m, k])
  const vb = broadcastView(b, [...batch, k, n])
  const dtype = promote(a.dtype, b.dtype)
  const nb = sizeOf(batch)
  const out = allocate(dtype, nb * m * n)
  // Per-batch offsets of each operand's matrix.
  const offA: number[] = []
  const offB: number[] = []
  forEachOffset2(batch, va.strides.slice(0, -2), va.offset, vb.strides.slice(0, -2), vb.offset, (i, j) => {
    offA.push(i)
    offB.push(j)
  })
  const [sai, sak] = va.strides.slice(-2)
  const [sbk, sbj] = vb.strides.slice(-2)
  const da = a.data
  const db = b.data
  const row = new Float64Array(n)
  for (let p = 0; p < nb; p++) {
    const oa = offA[p]
    const ob = offB[p]
    const base = p * m * n
    for (let i = 0; i < m; i++) {
      // i-k-j order: stream a row of B per element of A's row, which is cache-friendly for row-major B.
      row.fill(0)
      for (let q = 0; q < k; q++) {
        const aiq = da[oa + i * sai + q * sak]
        const bq = ob + q * sbk
        for (let j = 0; j < n; j++) row[j] += aiq * db[bq + j * sbj]
      }
      for (let j = 0; j < n; j++) out[base + i * n + j] = row[j]
    }
  }
  return fromData(out, [...batch, m, n])
}
