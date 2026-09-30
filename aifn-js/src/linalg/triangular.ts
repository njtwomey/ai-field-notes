/**
 * Triangular solves by forward and back substitution (Golub and Van Loan, 2013, "Matrix Computations", 4th ed.,
 * Algorithms 3.1.1–3.1.2), as a primitive with its derivative rule: for X = T⁻¹B the cotangents are
 * B̄ = T⁻ᵀX̄ and T̄ = −B̄Xᵀ restricted to T's triangle (Giles, 2008, §2.2.3).
 */

import {
  defineOp,
  matmul,
  mul,
  neg,
  reshape,
  shapeOfValue,
  transpose,
  unwrap,
  type Op,
  type Raw,
  type Tensor,
  type TensorResult,
  type Value,
} from 'aifn/tensor'
import { dense, denseSquare, LinAlgError, matrix } from './dense'

/** Options of `solveTriangular`. */
export type TriangularOptions = {
  /** T is lower triangular (default true); otherwise upper. Only that triangle of T is read. */
  lower?: boolean
  /** Solve Tᵀ X = B instead of T X = B (default false). */
  transpose?: boolean
  /** Take T's diagonal as ones without reading it (default false). */
  unitDiagonal?: boolean
}

type Params = Required<TriangularOptions>

/** Solve in place on a dense n×r right-hand side. Throws on a zero diagonal. */
export function substitute(t: Float64Array, n: number, b: Float64Array, r: number, p: Params, where: string): void {
  // Tᵀ of a lower matrix is upper: solving with Tᵀ walks the other way and reads T by columns.
  const forward = p.lower !== p.transpose
  const at = p.transpose ? (i: number, j: number) => t[j * n + i] : (i: number, j: number) => t[i * n + j]
  for (let step = 0; step < n; step++) {
    const i = forward ? step : n - 1 - step
    const d = p.unitDiagonal ? 1 : at(i, i)
    if (d === 0)
      throw new LinAlgError(`${where}: the triangular matrix is singular (zero at diagonal ${i})`, 'singular')
    for (let c = 0; c < r; c++) {
      let s = b[i * r + c]
      if (forward) for (let j = 0; j < i; j++) s -= at(i, j) * b[j * r + c]
      else for (let j = i + 1; j < n; j++) s -= at(i, j) * b[j * r + c]
      b[i * r + c] = s / d
    }
  }
}

/** A mask of T's triangle (with or without its diagonal). */
function triangleMask(n: number, lower: boolean, withDiagonal: boolean): Tensor {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if ((lower ? j < i : j > i) || (withDiagonal && i === j)) out[i * n + j] = 1
    }
  }
  return matrix(out, n, n)
}

const solveTriangularOp: Op<Params> = defineOp<Params>(
  'solveTriangular',
  ([t, b], p) => {
    const T = denseSquare(t, 'solveTriangular')
    const isVector = typeof b !== 'number' && b.shape.length === 1
    const B = isVector ? dense(reshape(b as Tensor, [-1, 1]), 'solveTriangular') : dense(b as Raw, 'solveTriangular')
    if (B.m !== T.n) throw new LinAlgError(`solveTriangular: T is ${T.n}×${T.n} but B has ${B.m} rows`, 'shape')
    substitute(T.a, T.n, B.a, B.n, p, 'solveTriangular')
    return isVector ? reshape(matrix(B.a, B.m, B.n), [-1]) : matrix(B.a, B.m, B.n)
  },
  (g, [t, b], x, p) => {
    const n = shapeOfValue(t)[0]
    const gb = solveTriangularOp([t, g], { ...p, transpose: !p.transpose })
    const column = (v: Value) => (shapeOfValue(v).length === 1 ? reshape(v, [-1, 1]) : v)
    // T̄ = −B̄ Xᵀ for T X = B, and −X B̄ᵀ for Tᵀ X = B; only T's triangle was read, so only it gets a cotangent.
    const outer = p.transpose ? matmul(column(x), transpose(column(gb))) : matmul(column(gb), transpose(column(x)))
    const gt = mul(neg(outer), triangleMask(n, p.lower, !p.unitDiagonal))
    return [gt, typeof unwrap(b) === 'number' ? null : gb]
  },
)

/**
 * Solve T X = B (or Tᵀ X = B) for triangular T (n×n) and B (n or n×r), by substitution. Only T's triangle is read.
 * Throws `LinAlgError` ('singular') when a diagonal entry is exactly zero.
 */
export function solveTriangular<T extends Value, B extends Value>(
  t: T,
  b: B,
  { lower = true, transpose = false, unitDiagonal = false }: TriangularOptions = {},
): TensorResult<T | B> {
  return solveTriangularOp([t, b], { lower, transpose, unitDiagonal }) as TensorResult<T | B>
}
