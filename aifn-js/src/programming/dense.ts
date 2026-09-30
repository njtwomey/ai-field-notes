/**
 * Internal helpers for `aifn/programming`: reading problem data given as tensors or plain arrays into dense row-major
 * working arrays, wrapping results as tensors, and a dense linear solve built on `aifn/linalg`'s LU.
 */

import { lu, luSolve } from 'aifn/linalg'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn/tensor'

/** A vector given as a tensor (rank 1) or a plain array of numbers. */
export type VectorInput = Tensor | readonly number[]

/** A matrix given as a tensor (rank 2) or a plain array of rows. */
export type MatrixInput = Tensor | readonly (readonly number[])[]

/** A dense row-major working matrix: `a[i * n + j]` is element (i, j). */
export type Mat = { m: number; n: number; a: Float64Array }

/** Read a vector into a fresh Float64Array; `undefined` reads as a vector of length `length` (default 0) of zeros. */
export function readVector(x: VectorInput | undefined, where: string, length?: number): Float64Array {
  if (x === undefined) return new Float64Array(length ?? 0)
  let out: Float64Array
  if (isTensor(x)) {
    if (x.shape.length !== 1) throw new Error(`${where}: expected a vector, got shape [${x.shape.join(', ')}]`)
    out = Float64Array.from(toFlat(x))
  } else out = Float64Array.from(x)
  if (length !== undefined && out.length !== length)
    throw new Error(`${where}: expected length ${length}, got ${out.length}`)
  return out
}

/**
 * Read a matrix with `columns` columns into a dense working copy; `undefined` reads as a 0×columns matrix. An empty
 * array of rows is also 0×columns.
 */
export function readMatrix(x: MatrixInput | undefined, where: string, columns: number): Mat {
  if (x === undefined) return { m: 0, n: columns, a: new Float64Array(0) }
  if (isTensor(x)) {
    if (x.shape.length !== 2) throw new Error(`${where}: expected a matrix, got shape [${x.shape.join(', ')}]`)
    if (x.shape[0] > 0 && x.shape[1] !== columns)
      throw new Error(`${where}: expected ${columns} columns, got ${x.shape[1]}`)
    return { m: x.shape[0], n: columns, a: Float64Array.from(toFlat(x)) }
  }
  const m = x.length
  const a = new Float64Array(m * columns)
  for (let i = 0; i < m; i++) {
    if (x[i].length !== columns) throw new Error(`${where}: row ${i} has ${x[i].length} entries, expected ${columns}`)
    a.set(x[i], i * columns)
  }
  return { m, n: columns, a }
}

/** Throw unless every entry is finite. */
export function checkFinite(a: ArrayLike<number>, where: string): void {
  for (let k = 0; k < a.length; k++) if (!Number.isFinite(a[k])) throw new Error(`${where}: non-finite entry`)
}

/** A float64 vector tensor holding a copy of `a`. */
export function vector(a: ArrayLike<number>): Tensor {
  return fromData(Float64Array.from(a))
}

/** An int32 vector tensor holding a copy of `a`. */
export function intVector(a: ArrayLike<number>): Tensor {
  return fromData(Int32Array.from(a))
}

/** An int32 tensor of the given shape holding a copy of `a` (row-major). */
export function intTensor(a: ArrayLike<number>, shape: readonly number[]): Tensor {
  return fromData(Int32Array.from(a), shape)
}

/** A float64 m×n matrix tensor holding a copy of `a` (row-major). */
export function matrix(a: ArrayLike<number>, m: number, n: number): Tensor {
  return fromData(Float64Array.from(a), [m, n])
}

/** y = A x for a dense matrix. */
export function matVec(A: Mat, x: ArrayLike<number>): Float64Array {
  const y = new Float64Array(A.m)
  for (let i = 0; i < A.m; i++) {
    let s = 0
    for (let j = 0; j < A.n; j++) s += A.a[i * A.n + j] * x[j]
    y[i] = s
  }
  return y
}

/** y = Aᵀ x for a dense matrix. */
export function matTVec(A: Mat, x: ArrayLike<number>): Float64Array {
  const y = new Float64Array(A.n)
  for (let i = 0; i < A.m; i++) {
    const xi = x[i]
    if (xi !== 0) for (let j = 0; j < A.n; j++) y[j] += A.a[i * A.n + j] * xi
  }
  return y
}

/** The dot product of two equal-length arrays. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let k = 0; k < a.length; k++) s += a[k] * b[k]
  return s
}

/** The infinity norm (largest absolute entry; 0 for an empty array). */
export function normInf(a: ArrayLike<number>): number {
  let s = 0
  for (let k = 0; k < a.length; k++) s = Math.max(s, Math.abs(a[k]))
  return s
}

/** The Euclidean norm. */
export function norm2(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a))
}

/**
 * Solve the dense n×n system `a x = b` by LU with partial pivoting (`aifn/linalg`). Returns `singular: true` (and a
 * zero `x`) instead of throwing when the matrix is singular to working precision.
 */
export function solveSystem(a: Float64Array, n: number, b: Float64Array): { x: Float64Array; singular: boolean } {
  if (n === 0) return { x: new Float64Array(0), singular: false }
  for (let k = 0; k < a.length; k++) if (!Number.isFinite(a[k])) return { x: new Float64Array(n), singular: true }
  const f = lu(fromData(Float64Array.from(a), [n, n]))
  if (f.singular) return { x: new Float64Array(n), singular: true }
  const x = Float64Array.from(toFlat(luSolve(f, fromData(Float64Array.from(b)))))
  return { x, singular: false }
}

/**
 * Indices of a maximal set of linearly independent rows of `A` (in order, greedily), by modified Gram–Schmidt with
 * re-orthogonalisation. A row is dependent when its residual norm is at most `tol` times its own norm.
 * With `rhs`, also reports `inconsistent`: a dependent row whose augmented row [aᵢ, bᵢ] is independent, i.e. the system
 * A x = b has no solution.
 */
export function independentRows(
  A: Mat,
  rhs?: ArrayLike<number>,
  tol = 1e-9,
): { rows: number[]; inconsistent: boolean } {
  const basis: Float64Array[] = []
  const augmented: Float64Array[] = []
  const rows: number[] = []
  let inconsistent = false
  const reduce = (v: Float64Array, against: Float64Array[]) => {
    for (let pass = 0; pass < 2; pass++)
      for (const q of against) {
        const d = dot(v, q)
        for (let k = 0; k < v.length; k++) v[k] -= d * q[k]
      }
    return norm2(v)
  }
  for (let i = 0; i < A.m; i++) {
    const row = A.a.slice(i * A.n, (i + 1) * A.n)
    const size = norm2(row)
    const aug = new Float64Array(A.n + 1)
    aug.set(row)
    aug[A.n] = rhs ? rhs[i] : 0
    const augSize = norm2(aug)
    const r = reduce(row, basis)
    const ra = reduce(aug, augmented)
    if (r <= tol * Math.max(size, 1e-300) || size === 0) {
      if (rhs && ra > tol * Math.max(augSize, 1)) inconsistent = true
      continue
    }
    rows.push(i)
    basis.push(row.map((v) => v / r))
    augmented.push(aug.map((v) => v / ra))
  }
  return { rows, inconsistent }
}
