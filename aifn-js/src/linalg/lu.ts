/**
 * LU factorisation with partial pivoting, PA = LU (Golub and Van Loan, 2013, Algorithm 3.4.1), and the solve,
 * inverse and determinants built on it. `solve`, `inverse`, `det` and `logDet` are primitives with derivative rules
 * from Giles (2008), "Collected matrix derivative results for forward and reverse mode algorithmic differentiation",
 * §2.2: for X = A⁻¹B, B̄ = A⁻ᵀX̄ and Ā = −B̄Xᵀ; for Y = A⁻¹, Ā = −Yᵀ Ȳ Yᵀ; for d = det A, Ā = d̄·d·A⁻ᵀ.
 */

import {
  defineOp,
  fromData,
  matmul,
  mul,
  neg,
  reshape,
  shapeOfValue,
  transpose,
  unwrap,
  type NumberResult,
  type Op,
  type Tensor,
  type TensorResult,
  type Value,
} from 'aifn/tensor'
import { dense, denseSquare, EPS, LinAlgError, matrix, maxAbs, untraced, type Dense } from './dense'

/** The result of `lu`. */
export type LU = {
  /** Unit lower-triangular factor (n×n). */
  L: Tensor
  /** Upper-triangular factor (n×n). */
  U: Tensor
  /** Permutation matrix with PA = LU. */
  P: Tensor
  /** The same permutation as indices: row i of PA is row `perm[i]` of A. */
  perm: Int32Array
  /** det P = ±1, the parity of the row swaps. */
  sign: 1 | -1
  /**
   * True when a pivot is at most n·ε·max|A| in magnitude: A is singular to working precision, and solves with it
   * would be dominated by rounding.
   */
  singular: boolean
}

/** Factor a dense square matrix in place: L below the diagonal (unit diagonal implied), U on and above. */
function factor({ n, a }: Dense): { perm: Int32Array; sign: 1 | -1; singular: boolean } {
  const perm = Int32Array.from({ length: n }, (_, i) => i)
  const tolerance = n * EPS * maxAbs(a)
  let sign: 1 | -1 = 1
  let singular = false
  for (let k = 0; k < n; k++) {
    // Partial pivoting: the largest |a_ik| on or below the diagonal; the first one wins ties.
    let p = k
    for (let i = k + 1; i < n; i++) if (Math.abs(a[i * n + k]) > Math.abs(a[p * n + k])) p = i
    if (p !== k) {
      for (let j = 0; j < n; j++) [a[k * n + j], a[p * n + j]] = [a[p * n + j], a[k * n + j]]
      ;[perm[k], perm[p]] = [perm[p], perm[k]]
      sign = -sign as 1 | -1
    }
    const pivot = a[k * n + k]
    if (Math.abs(pivot) <= tolerance) singular = true
    // An exactly zero column needs no elimination (and dividing by the zero pivot would make NaN).
    if (pivot === 0) continue
    for (let i = k + 1; i < n; i++) {
      const l = (a[i * n + k] /= pivot)
      if (l !== 0) for (let j = k + 1; j < n; j++) a[i * n + j] -= l * a[k * n + j]
    }
  }
  return { perm, sign, singular }
}

/**
 * LU factorisation with partial pivoting of a square matrix: PA = LU with L unit lower-triangular and U upper
 * triangular. Never throws for singular input: `singular` reports it. Non-finite entries throw `LinAlgError`.
 */
export function lu(a: Tensor): LU {
  untraced(a, 'lu')
  const d = denseSquare(a, 'lu')
  const { n } = d
  const { perm, sign, singular } = factor(d)
  const L = new Float64Array(n * n)
  const U = new Float64Array(n * n)
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    P[i * n + perm[i]] = 1
    for (let j = 0; j < n; j++) {
      if (j < i) L[i * n + j] = d.a[i * n + j]
      else U[i * n + j] = d.a[i * n + j]
    }
    L[i * n + i] = 1
  }
  return { L: matrix(L, n, n), U: matrix(U, n, n), P: matrix(P, n, n), perm, sign, singular }
}

/** Solve with a factored dense matrix, in place on an n×r right-hand side (rows already permuted). */
function substitute(lu: Float64Array, n: number, b: Float64Array, r: number): void {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const l = lu[i * n + j]
      if (l !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= l * b[j * r + c]
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      const u = lu[i * n + j]
      if (u !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= u * b[j * r + c]
    }
    const d = lu[i * n + i]
    for (let c = 0; c < r; c++) b[i * r + c] /= d
  }
}

/** Solve A X = B for dense A (factored here) and a dense right-hand side; throws when A is singular. */
function solveDense(A: Dense, B: Dense, where: string): Float64Array {
  if (B.m !== A.n) throw new LinAlgError(`${where}: A is ${A.n}×${A.n} but B has ${B.m} rows`, 'shape')
  const { perm, singular } = factor(A)
  if (singular) throw new LinAlgError(`${where}: the matrix is singular to working precision`, 'singular')
  const r = B.n
  const x = new Float64Array(A.n * r)
  for (let i = 0; i < A.n; i++) for (let c = 0; c < r; c++) x[i * r + c] = B.a[perm[i] * r + c]
  substitute(A.a, A.n, x, r)
  return x
}

/** A vector or matrix right-hand side as a dense n×r array, and whether it was a vector. */
function rightHandSide(b: Value, where: string): { B: Dense; isVector: boolean } {
  const shape = shapeOfValue(b)
  if (shape.length === 1) return { B: dense(reshape(unwrap(b), [-1, 1]), where), isVector: true }
  return { B: dense(b, where), isVector: false }
}

/**
 * Solve a factored system: X with A X = B, given `lu(A)` and B (n or n×r). Throws `LinAlgError` when the
 * factorisation is `singular`.
 */
export function luSolve(factorisation: LU, b: Tensor): Tensor {
  if (factorisation.singular) throw new LinAlgError('luSolve: the matrix is singular to working precision', 'singular')
  const n = factorisation.L.shape[0]
  const { B, isVector } = rightHandSide(b, 'luSolve')
  if (B.m !== n) throw new LinAlgError(`luSolve: A is ${n}×${n} but B has ${B.m} rows`, 'shape')
  // Rebuild the packed form: L below the diagonal, U on and above.
  const packed = new Float64Array(n * n)
  const L = factorisation.L.data
  const U = factorisation.U.data
  for (let k = 0; k < n * n; k++) packed[k] = Math.floor(k / n) > k % n ? L[k] : U[k]
  const x = new Float64Array(n * B.n)
  for (let i = 0; i < n; i++) for (let c = 0; c < B.n; c++) x[i * B.n + c] = B.a[factorisation.perm[i] * B.n + c]
  substitute(packed, n, x, B.n)
  return isVector ? fromData(x, [n]) : matrix(x, n, B.n)
}

const solveOp: Op<undefined> = defineOp<undefined>(
  'solve',
  ([a, b]) => {
    const { B, isVector } = rightHandSide(b, 'solve')
    const x = solveDense(denseSquare(a, 'solve'), B, 'solve')
    return isVector ? fromData(x, [B.m]) : matrix(x, B.m, B.n)
  },
  (g, [a, b], x) => {
    const gb = solveOp([transpose(a), g], undefined)
    const column = (v: Value) => (shapeOfValue(v).length === 1 ? reshape(v, [-1, 1]) : v)
    return [neg(matmul(column(gb), transpose(column(x)))), shapeOfValue(b).length === 0 ? null : gb]
  },
)

/**
 * Solve A X = B for square A (n×n) and B (n or n×r), by LU with partial pivoting. Throws `LinAlgError` ('singular')
 * when A is singular to working precision; call `lu` first to test without throwing.
 */
export function solve<A extends Value, B extends Value>(a: A, b: B): TensorResult<A | B> {
  return solveOp([a, b], undefined) as TensorResult<A | B>
}

const inverseOp: Op<undefined> = defineOp<undefined>(
  'inverse',
  ([a]) => {
    const A = denseSquare(a, 'inverse')
    const I = new Float64Array(A.n * A.n)
    for (let i = 0; i < A.n; i++) I[i * A.n + i] = 1
    return matrix(solveDense(A, { m: A.n, n: A.n, a: I }, 'inverse'), A.n, A.n)
  },
  (g, _inputs, y) => [neg(matmul(matmul(transpose(y), g), transpose(y)))],
)

/**
 * The inverse of a square matrix, by LU with partial pivoting. Throws `LinAlgError` ('singular') when A is singular to
 * working precision. Prefer `solve` or `choleskySolve` to multiplying by an inverse.
 */
export function inverse<A extends Value>(a: A): TensorResult<A> {
  return inverseOp([a], undefined) as TensorResult<A>
}

/** Sign and log |det| from a dense LU factorisation. */
function determinant(a: Value, where: string): { sign: number; logAbs: number; det: number } {
  const A = denseSquare(a, where)
  const { sign } = factor(A)
  let s: number = sign
  let logAbs = 0
  let det: number = sign
  for (let i = 0; i < A.n; i++) {
    const u = A.a[i * A.n + i]
    s *= Math.sign(u)
    logAbs += Math.log(Math.abs(u))
    det *= u
  }
  // Report an exactly zero determinant as +0, not −0.
  return { sign: s === 0 ? 0 : s, logAbs, det: det === 0 ? 0 : det }
}

const detOp: Op<undefined> = defineOp<undefined>(
  'det',
  ([a]) => determinant(a, 'det').det,
  (g, [a], d) => [mul(mul(g, d), transpose(inverse(a)))],
)

/** The determinant of a square matrix (0 for a singular one), from its LU factorisation. */
export function det<A extends Value>(a: A): NumberResult<A> {
  return detOp([a], undefined) as NumberResult<A>
}

const logDetOp: Op<undefined> = defineOp<undefined>(
  'logDet',
  ([a]) => determinant(a, 'logDet').logAbs,
  (g, [a]) => [mul(g, transpose(inverse(a)))],
)

/**
 * log |det A| for a square matrix, from its LU factorisation, without the overflow of forming det A (−∞ for a
 * singular matrix). The sign is that of `det(A)`; for a positive-definite matrix, `choleskyLogDet` is cheaper.
 */
export function logDet<A extends Value>(a: A): NumberResult<A> {
  return logDetOp([a], undefined) as NumberResult<A>
}

/** The sign of det A (−1, 0 or 1), from its LU factorisation. */
export function signDet(a: Tensor): number {
  return determinant(a, 'signDet').sign
}
