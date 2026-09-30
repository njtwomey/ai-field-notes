/**
 * Indexed reads and writes as primitives: `gather` (read flat indices), `scatterAdd` (add into flat indices) and
 * `take` (rows along the first axis, e.g. an embedding lookup). `gather` and `scatterAdd` are each other's adjoint, so
 * both are differentiable to any order.
 */

import { fromData, isContiguous, sizeOf, type Tensor } from './core'
import { toFlat } from './create'
import { defineOp, type Op } from './primitive'
import { shapeOfValue } from './structure'
import type { Value } from './tape'

/** Row-major float64 data of a raw value (no copy when already contiguous float64 at offset 0). */
function f64(x: number | Tensor, where: string): Float64Array {
  if (typeof x === 'number') throw new Error(`${where}: expected a tensor, got a number`)
  if (x.dtype === 'float64' && isContiguous(x) && x.offset === 0 && x.data.length === sizeOf(x.shape)) {
    return x.data as Float64Array
  }
  return Float64Array.from(toFlat(x))
}

type GatherParams = { indices: Int32Array; shape: readonly number[]; source: readonly number[] }

const gatherOp: Op<GatherParams> = defineOp<GatherParams>(
  'gather',
  ([x], { indices, shape }) => {
    const data = f64(x, 'gather')
    const out = new Float64Array(indices.length)
    for (let k = 0; k < indices.length; k++) out[k] = data[indices[k]]
    return fromData(out, shape)
  },
  (g, _inputs, _y, p) => [scatterAddOp([g], p)],
)

const scatterAddOp: Op<GatherParams> = defineOp<GatherParams>(
  'scatterAdd',
  ([g], { indices, source }) => {
    const data = f64(g, 'scatterAdd')
    const out = new Float64Array(sizeOf(source))
    for (let k = 0; k < indices.length; k++) out[indices[k]] += data[k]
    return fromData(out, source)
  },
  (u, _inputs, _y, p) => [gatherOp([u], p)],
)

function checkIndices(indices: Int32Array, size: number, where: string): void {
  for (let k = 0; k < indices.length; k++)
    if (indices[k] < 0 || indices[k] >= size) throw new RangeError(`${where}: index ${indices[k]} outside [0, ${size})`)
}

/**
 * out[k] = x.flat[indices[k]] (row-major flat indices), reshaped to `shape` (whose size must equal indices.length).
 * Differentiable in x: the cotangent is scattered back and summed where an index repeats.
 */
export function gather(x: Value, indices: Int32Array, shape: readonly number[]): Value {
  if (indices.length !== sizeOf(shape)) throw new Error('gather: indices and shape disagree')
  const source = shapeOfValue(x)
  checkIndices(indices, sizeOf(source), 'gather')
  return gatherOp([x], { indices, shape, source })
}

/**
 * The adjoint of `gather`: a tensor of shape `shape` holding, at each flat index, the sum of the values of `g` gathered
 * from it (out.flat[indices[k]] += g.flat[k]). `g` must have indices.length elements. Differentiable in g.
 */
export function scatterAdd(g: Value, indices: Int32Array, shape: readonly number[]): Value {
  const from = shapeOfValue(g)
  if (indices.length !== sizeOf(from)) throw new Error('scatterAdd: indices and values disagree')
  checkIndices(indices, sizeOf(shape), 'scatterAdd')
  return scatterAddOp([g], { indices, shape: from, source: shape })
}

/**
 * The rows `indices` of x along its first axis: shape [...indices shape, ...x.shape.slice(1)]. An embedding lookup is
 * `take(table, ids)`. Indices are integers in [0, x.shape[0]); the gradient adds up over repeated rows.
 */
export function take(x: Value, indices: Tensor | ArrayLike<number>): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 0) throw new Error('take: needs a tensor of rank ≥ 1')
  const isT = 'shape' in indices && 'strides' in indices
  const ids = isT ? toFlat(indices) : Array.from(indices as ArrayLike<number>)
  const idShape = isT ? [...indices.shape] : [ids.length]
  const rows = shape[0]
  const width = sizeOf(shape.slice(1))
  const flat = new Int32Array(ids.length * width)
  ids.forEach((id, k) => {
    if (!Number.isInteger(id) || id < 0 || id >= rows) throw new RangeError(`take: index ${id} outside [0, ${rows})`)
    for (let j = 0; j < width; j++) flat[k * width + j] = id * width + j
  })
  return gather(x, flat, [...idShape, ...shape.slice(1)])
}
