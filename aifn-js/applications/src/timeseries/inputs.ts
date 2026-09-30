/**
 * Private input conversions of `aifn-applied/timeseries`: series and coefficient arguments (`VectorLike`,
 * `MatrixLike`) as arrays of numbers for the scalar recursions. Matrix algebra uses `aifn/foundation/tensor` and
 * `aifn/numerics/linalg`.
 */

import { dense, isTensor, toFlat, toRows } from 'aifn/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn/foundation/contracts'

// Types defined once, in `aifn/foundation/contracts` (re-exported for this module).
export type { VectorLike, MatrixLike } from 'aifn/foundation/contracts'

type Mat = number[][]

/** A vector input as an array of numbers (`dense.toF64`). */
export function toVec(v: VectorLike, where: string): number[] {
  return Array.from(dense.toF64(v, where))
}

function toMat(a: MatrixLike | number, where: string): Mat {
  if (typeof a === 'number') return [[a]]
  if (isTensor(a)) {
    if (a.shape.length === 0) return [[toFlat(a)[0]]]
    if (a.shape.length !== 2) throw new Error(`${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    return toRows(a)
  }
  const rows = (a as readonly ArrayLike<number>[]).map((r) => Array.from(r))
  if (rows.some((r) => r.length !== rows[0].length)) throw new Error(`${where}: ragged matrix rows`)
  return rows
}

/** Observations as T rows of m numbers: a vector is T scalar observations, a matrix is T×m. */
export function toSeries(y: VectorLike | MatrixLike, where: string): Mat {
  if (isTensor(y)) return y.shape.length === 1 ? toFlat(y).map((v) => [v]) : toMat(y, where)
  const list = y as ArrayLike<number> | readonly ArrayLike<number>[]
  if (list.length > 0 && typeof list[0] === 'number') return Array.from(list as ArrayLike<number>, (v) => [v])
  return toMat(list as readonly ArrayLike<number>[], where)
}
