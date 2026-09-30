/**
 * Private helpers: small dense matrices as rows of numbers (the state-space models here have a handful of states, and
 * the filters run inside likelihood optimisations, so plain arrays keep the code close to the equations and fast),
 * plus the conversions to and from the public tensor surface. The factorisations are not copied: solves use
 * `aifn/numerics/linalg`'s `solveDense` and square roots its `eigh`.
 */

import { eigh, solveDense } from 'aifn/numerics/linalg'
import { fromData, isTensor, type Tensor, toFlat, toRows } from 'aifn/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn/foundation/contracts'

// Types defined once, in `aifn/foundation/contracts` (re-exported for this module).
export type { VectorLike, MatrixLike } from 'aifn/foundation/contracts'

/** A small dense matrix as rows of numbers. */
export type Mat = number[][]
/** A small dense vector as an array of numbers. */
export type Vec = number[]

/** A copy of a vector argument as an array of numbers. */
export function toVec(v: VectorLike, where: string): Vec {
  if (isTensor(v)) {
    if (v.shape.length > 1) throw new Error(`${where}: expected a vector, got shape [${v.shape.join(', ')}]`)
    return toFlat(v)
  }
  return Array.from(v as ArrayLike<number>)
}

/** A copy of a matrix argument as rows of numbers; a number or a scalar tensor is a 1×1 matrix. */
export function toMat(a: MatrixLike | number, where: string): Mat {
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

export const vecT = (v: Vec): Tensor => fromData(Float64Array.from(v), [v.length])
export const matT = (a: Mat): Tensor => fromData(Float64Array.from(a.flat()), [a.length, a[0]?.length ?? 0])
/** Stack T vectors of length n into a [T, n] tensor. */
export const stackVecs = (vs: Vec[], n: number): Tensor => fromData(Float64Array.from(vs.flat()), [vs.length, n])
/** Stack T n×m matrices into a [T, n, m] tensor. */
export const stackMats = (ms: Mat[], n: number, m: number): Tensor =>
  fromData(Float64Array.from(ms.flatMap((a) => a.flat())), [ms.length, n, m])

export const zerosM = (n: number, m = n): Mat => Array.from({ length: n }, () => new Array<number>(m).fill(0))
export const eyeM = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)))
export const transpose = (a: Mat): Mat => (a.length ? a[0].map((_, j) => a.map((r) => r[j])) : [])
export const matmul = (a: Mat, b: Mat): Mat =>
  a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
export const matvec = (a: Mat, x: Vec): Vec => a.map((row) => row.reduce((s, v, k) => s + v * x[k], 0))
export const addM = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v + b[i][j]))
export const subM = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v - b[i][j]))
export const scaleM = (a: Mat, s: number): Mat => a.map((r) => r.map((v) => v * s))
export const addV = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i])
export const subV = (a: Vec, b: Vec): Vec => a.map((v, i) => v - b[i])
export const outer = (a: Vec, b: Vec): Mat => a.map((u) => b.map((v) => u * v))
/** (A + Aᵀ)/2, to remove the asymmetry rounding leaves in a covariance. */
export const symmetrise = (a: Mat): Mat => a.map((r, i) => r.map((v, j) => (v + a[j][i]) / 2))
/** A B Aᵀ */
export const sandwich = (a: Mat, b: Mat): Mat => matmul(matmul(a, b), transpose(a))

/**
 * Solve A X = B (A n×n, B n×k) and log|det A|, by `aifn/numerics/linalg`'s `solveDense` (LU with partial pivoting).
 * `singular` is true when a pivot is at most n·ε·max|A| (then X is null): the caller reports it rather than dividing
 * by zero.
 */
export function solveM(a: Mat, b: Mat): { x: Mat | null; logDet: number; singular: boolean } {
  const n = a.length
  const k = b[0]?.length ?? 0
  const r = solveDense(a.flat(), b.flat(), n)
  if (!r.x) return { x: null, logDet: -Infinity, singular: true }
  const x = r.x
  return {
    x: Array.from({ length: n }, (_, i) => Array.from(x.subarray(i * k, (i + 1) * k))),
    logDet: r.logAbsDet,
    singular: false,
  }
}

/** A⁻¹, or null when A is singular to working precision. */
export const inverseM = (a: Mat): Mat | null => solveM(a, eyeM(a.length)).x

/**
 * A square root S with S Sᵀ = A for a symmetric positive semi-definite A, from `aifn/numerics/linalg`'s symmetric
 * eigendecomposition: S = V diag(√max(λ, 0)). Unlike a Cholesky factor it exists for singular A (a noise covariance
 * with a deterministic component, e.g. a zero diagonal entry), so sampling such noise gives exact zeros rather than
 * NaN. `negative` is the most negative eigenvalue found (0 for a valid covariance); callers report it.
 */
export function sqrtPsd(a: Mat): { S: Mat; negative: number } {
  const n = a.length
  if (n === 0) return { S: [], negative: 0 }
  const { values, vectors } = eigh(matT(symmetrise(a)))
  const lambda = toFlat(values)
  const V = toRows(vectors)
  let negative = 0
  const roots = lambda.map((l) => {
    negative = Math.min(negative, l)
    return Math.sqrt(Math.max(l, 0))
  })
  return { S: V.map((row) => row.map((v, j) => v * roots[j])), negative }
}
