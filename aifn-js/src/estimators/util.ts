/** Private helpers: dense float64 views of tensors and small shape checks. */

import { copy, fromData, isContiguous, type Tensor } from 'aifn/tensor'

/**
 * The elements of `t` in row-major order as a Float64Array. Shares `t.data` when it is already a contiguous float64
 * layout starting at 0, so the result must not be mutated.
 */
export function values(t: Tensor): Float64Array {
  if (t.dtype === 'float64' && t.offset === 0 && isContiguous(t) && t.data.length === sizeOf(t.shape)) {
    return t.data as Float64Array
  }
  return copy(t, 'float64').data as Float64Array
}

export function sizeOf(shape: readonly number[]): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

/** A float64 tensor over `data` with `shape`. */
export function tensorOf(data: Float64Array | Int32Array, shape: readonly number[] = [data.length]): Tensor {
  return fromData(data, shape)
}

/** Rows and columns of a matrix, or throw naming the caller. */
export function matrixShape(x: Tensor, where: string): [number, number] {
  if (x.shape.length !== 2) throw new Error(`${where}: expected a matrix [n, d], got shape [${x.shape.join(', ')}]`)
  return [x.shape[0], x.shape[1]]
}

/** Values of a label vector (rank 1, or [n, 1]) as numbers. */
export function labels(y: Tensor, where: string): Float64Array {
  if (y.shape.length > 2 || (y.shape.length === 2 && y.shape[1] !== 1)) {
    throw new Error(`${where}: expected a vector of targets [n], got shape [${y.shape.join(', ')}]`)
  }
  return values(y)
}
