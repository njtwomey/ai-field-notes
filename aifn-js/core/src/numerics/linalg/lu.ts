/**
 * LU factorisation with partial pivoting, PA = LU (Golub and Van Loan, 2013, "Matrix Computations", 4th ed.,
 * Algorithm 3.4.1), and the solves, inverse and determinants built on it. A matrix is factored once: `luFactor` keeps
 * the packed factor, `luSolve` is a primitive that only substitutes with it (linear in B), and `solve`, `inverse`,
 * `det` and `logDet` factor once and hand the factor to their derivative rules, which never refactor. The rules are
 * Giles (2008), "Collected matrix derivative results for forward and reverse mode algorithmic differentiation", §2.2:
 * for X = A⁻¹B, B̄ = A⁻ᵀX̄ (a transposed solve with the same factor) and Ā = −B̄Xᵀ; for d = det A, Ā = d̄·d·A⁻ᵀ; for
 * log |det A|, Ā = A⁻ᵀ.
 */

import {
  eye,
  fromData,
  matmul,
  mul,
  neg,
  type NumberResult,
  type Op,
  reshape,
  shapeOfValue,
  type Tensor,
  type TensorResult,
  transpose,
  unwrap,
  definePrimitive,
  type Value,
} from 'aifn/foundation/tensor'
import type { Index } from 'aifn/foundation/contracts'
import { ShapeError } from 'aifn/foundation/errors'
import { dense, denseSquare, EPS, LinAlgError, matrix, maxAbs, untraced, wellConditioned, type Dense } from './dense'

/**
 * A matrix factored once, PA = LU, for repeated solves: `luSolve(f, b)` substitutes with it, and derivatives of the
 * solves with respect to A reuse it (no refactorisation).
 */
export type LuFactor<A extends Value = Tensor> = {
  /** The matrix that was factored (kept, possibly traced, so solves are differentiable in it). */
  readonly matrix: A
  /** The packed factor (n×n): L strictly below the diagonal (its unit diagonal implied), U on and above. */
  readonly packed: Tensor
  /** The row permutation: row i of PA is row `perm[i]` of A. */
  readonly perm: Int32Array
  /** det P = ±1, the parity of the row swaps. */
  readonly sign: 1 | -1
  /**
   * True when a pivot is at most n·ε·max|A| in magnitude: A is singular to working precision, and solves with it
   * would be dominated by rounding. `luSolve` then throws.
   */
  readonly singular: boolean
}

/** The result of `lu`: the factor of `luFactor` with L, U and P unpacked as matrices. */
export type LU = LuFactor & {
  /** Unit lower-triangular factor (n×n). */
  readonly L: Tensor
  /** Upper-triangular factor (n×n). */
  readonly U: Tensor
  /** Permutation matrix with PA = LU. */
  readonly P: Tensor
}

/**
 * Factor a dense square matrix in place: L below the diagonal (unit diagonal implied), U on and above. Shared with
 * `solveDense` (the one LU elimination in this module).
 */
export function factor({ n, a }: Dense): { perm: Int32Array; sign: 1 | -1; singular: boolean } {
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

/** The raw factor of a square matrix (traced input is read through its value). */
function factorOf(a: Value, where: string): { packed: Float64Array; n: number; perm: Int32Array; sign: 1 | -1; singular: boolean } {
  const d = denseSquare(a, where)
  return { ...factor(d), packed: d.a, n: d.n }
}

/**
 * Factor a square matrix once, PA = LU with partial pivoting, for `luSolve`. Accepts traced input: the factor keeps
 * the matrix, so solves with it are differentiable in A and B. Never throws for singular input (`singular` reports
 * it); non-finite entries throw `LinAlgError`.
 *
 * @example const f = luFactor(A); const x = luSolve(f, b); const y = luSolve(f, c) // one factorisation
 */
export function luFactor<A extends Value>(a: A): LuFactor<A> {
  const { packed, n, perm, sign, singular } = factorOf(a, 'luFactor')
  return { matrix: a, packed: matrix(packed, n, n), perm, sign, singular }
}

/**
 * LU factorisation with partial pivoting of a square matrix, unpacked: PA = LU with L unit lower-triangular, U upper
 * triangular and P the permutation matrix (for display and teaching; `luFactor` is the solver's form). Never throws
 * for singular input: `singular` reports it. Non-finite entries throw `LinAlgError`; traced input throws
 * `NotDifferentiableError` (L and U have no derivative rule; differentiate through `luSolve` instead).
 */
export function lu(a: Tensor): LU {
  untraced(a, 'lu')
  const f = luFactor(a)
  const n = f.packed.shape[0]
  const packed = f.packed.data
  const L = new Float64Array(n * n)
  const U = new Float64Array(n * n)
  const P = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    P[i * n + f.perm[i]] = 1
    for (let j = 0; j < n; j++) {
      if (j < i) L[i * n + j] = packed[i * n + j]
      else U[i * n + j] = packed[i * n + j]
    }
    L[i * n + i] = 1
  }
  return { ...f, L: matrix(L, n, n), U: matrix(U, n, n), P: matrix(P, n, n) }
}

/** Solve with a packed factor, in place on an n×r right-hand side whose rows are already permuted: L then U. */
export function substitute(lu: ArrayLike<number>, n: number, b: Float64Array, r: number): void {
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

/**
 * Solve Aᵀ X = B with the factor of A, in place on an n×r right-hand side: PA = LU gives Aᵀ = Uᵀ Lᵀ P, so solve
 * Uᵀ y = B (forward), Lᵀ z = y (backward, unit diagonal), then X = Pᵀ z.
 */
function substituteTransposed(lu: ArrayLike<number>, n: number, perm: ArrayLike<Index>, b: Float64Array, r: number) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const u = lu[j * n + i]
      if (u !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= u * b[j * r + c]
    }
    const d = lu[i * n + i]
    for (let c = 0; c < r; c++) b[i * r + c] /= d
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      const l = lu[j * n + i]
      if (l !== 0) for (let c = 0; c < r; c++) b[i * r + c] -= l * b[j * r + c]
    }
  }
  const out = new Float64Array(n * r)
  for (let i = 0; i < n; i++) for (let c = 0; c < r; c++) out[perm[i] * r + c] = b[i * r + c]
  b.set(out)
}

/** A vector or matrix right-hand side as a dense n×r array, and whether it was a vector. */
function rightHandSide(b: Value, where: string): { B: Dense; isVector: boolean } {
  const shape = shapeOfValue(b)
  if (shape.length === 1) return { B: dense(reshape(unwrap(b), [-1, 1]), where), isVector: true }
  return { B: dense(b, where), isVector: false }
}

/**
 * Parameters of the `luSolve` primitive: the factor of its first input (computed from it when absent, as in the
 * generated tests) and whether to solve with Aᵀ.
 */
type SolveParams = {
  readonly factor?: { readonly packed: ArrayLike<number>; readonly perm: Int32Array; readonly singular: boolean }
  readonly transpose: boolean
}

const luSolveOp: Op<SolveParams> = definePrimitive<SolveParams>({
  id: 'numerics/linalg/luSolve',
  arity: 2,
  impl: ([a, b], p) => {
    const f = p.factor ?? factorOf(a, 'luSolve')
    if (f.singular) throw new LinAlgError('luSolve: the matrix is singular to working precision', 'singular')
    const n = f.perm.length
    const { B, isVector } = rightHandSide(b, 'luSolve')
    if (B.m !== n) throw new ShapeError('luSolve', `luSolve: A is ${n}×${n} but B has ${B.m} rows`)
    const r = B.n
    const x = new Float64Array(n * r)
    if (p.transpose) {
      x.set(B.a)
      substituteTransposed(f.packed, n, f.perm, x, r)
    } else {
      for (let i = 0; i < n; i++) for (let c = 0; c < r; c++) x[i * r + c] = B.a[f.perm[i] * r + c]
      substitute(f.packed, n, x, r)
    }
    return isVector ? fromData(x, [n]) : matrix(x, n, r)
  },
  vjp: (g, [a], x, p) => {
    // B̄ = A⁻ᵀX̄ (the other orientation, same factor); Ā = −B̄Xᵀ, or −X B̄ᵀ when solving with Aᵀ.
    const gb = luSolveOp([a, g], { ...p, transpose: !p.transpose })
    const column = (v: Value) => (shapeOfValue(v).length === 1 ? reshape(v, [-1, 1]) : v)
    const ga = p.transpose ? matmul(column(x), transpose(column(gb))) : matmul(column(gb), transpose(column(x)))
    return [neg(ga), gb]
  },
  doc: { summary: 'Solve A X = B (or Aᵀ X = B) by substitution with the LU factor of A.' },
  test: {
    secondOrder: true,
    cases: (draw) => [
      { inputs: [wellConditioned(draw, 3), draw([3, 2])], params: { transpose: false } },
      { inputs: [wellConditioned(draw, 3), draw([3])], params: { transpose: true } },
    ],
  },
})

/** The primitive's parameters for a factor. */
const paramsOf = (f: LuFactor<Value>, transpose = false): SolveParams => ({
  factor: { packed: f.packed.data, perm: f.perm, singular: f.singular },
  transpose,
})

/**
 * Solve a factored system: X with A X = B (or Aᵀ X = B with `transpose`), given `luFactor(A)` (or `lu(A)`) and B (n
 * or n×r), by substitution only. Linear in B and differentiable in both A (through the factor's matrix) and B; the
 * derivative reuses the factor. Throws `LinAlgError` ('singular') when the factor is `singular`.
 */
export function luSolve<A extends Value, B extends Value>(
  f: LuFactor<A>,
  b: B,
  { transpose = false }: { transpose?: boolean } = {},
): TensorResult<A | B> {
  return luSolveOp([f.matrix, b], paramsOf(f, transpose)) as TensorResult<A | B>
}

/**
 * Solve A X = B for square A (n×n) and B (n or n×r): one LU factorisation with partial pivoting, then `luSolve`, so
 * the derivative reuses the factor. Throws `LinAlgError` ('singular') when A is singular to working precision; call
 * `luFactor` first to test without throwing.
 */
export function solve<A extends Value, B extends Value>(a: A, b: B): TensorResult<A | B> {
  return luSolveOp([a, b], paramsOf(luFactor(a))) as TensorResult<A | B>
}

/**
 * The inverse of a square matrix: `luSolve` of its factor against I (so differentiable, with Ā = −A⁻ᵀ Ā⁻¹ A⁻ᵀ through
 * the solve's rule). Throws `LinAlgError` ('singular') when A is singular to working precision. Prefer `solve` or
 * `choleskySolve` to multiplying by an inverse.
 */
export function inverse<A extends Value>(a: A): TensorResult<A> {
  const f = luFactor(a)
  return luSolveOp([a, eye(f.perm.length)], paramsOf(f)) as TensorResult<A>
}

/** Sign, log |det| and det from a raw factor. */
function determinant(f: { packed: ArrayLike<number>; n: number; sign: 1 | -1 }): {
  sign: number
  logAbs: number
  det: number
} {
  let s: number = f.sign
  let logAbs = 0
  let det: number = f.sign
  for (let i = 0; i < f.n; i++) {
    const u = f.packed[i * f.n + i]
    s *= Math.sign(u)
    logAbs += Math.log(Math.abs(u))
    det *= u
  }
  // Report an exactly zero determinant as +0, not −0.
  return { sign: s === 0 ? 0 : s, logAbs, det: det === 0 ? 0 : det }
}

/** Parameters of `det` and `logDet`: the factor of the input, computed when absent. */
type DetParams = { readonly factor?: LuFactor<Value> }

/** A⁻ᵀ from the factor (a transposed solve against I); throws `LinAlgError` at singular A. */
function inverseTransposed(a: Value, p: DetParams): Value {
  const f = p.factor ?? luFactor(unwrap(a))
  return luSolveOp([a, eye(f.perm.length)], paramsOf({ ...f, matrix: a }, true))
}

const factorFor = (a: Value, p: DetParams, where: string) => {
  if (p.factor) return { packed: p.factor.packed.data, n: p.factor.perm.length, sign: p.factor.sign }
  return factorOf(a, where)
}

const detOp: Op<DetParams> = definePrimitive<DetParams>({
  id: 'numerics/linalg/det',
  arity: 1,
  impl: ([a], p) => determinant(factorFor(a, p, 'det')).det,
  // At a singular A the rule throws (the adjugate form is not implemented).
  vjp: (g, [a], d, p) => [mul(mul(g, d), inverseTransposed(a, p))],
  doc: { summary: 'The determinant of a square matrix.' },
  test: { secondOrder: true, cases: (draw) => [{ inputs: [wellConditioned(draw, 3)], params: {} }] },
})

/**
 * The determinant of a square matrix (0 for a singular one), from its LU factorisation. Its derivative d·A⁻ᵀ reuses
 * the factor; at a singular A differentiating throws `LinAlgError` ('singular').
 */
export function det<A extends Value>(a: A): NumberResult<A> {
  return detOp([a], { factor: luFactor(a) }) as NumberResult<A>
}

const logDetOp: Op<DetParams> = definePrimitive<DetParams>({
  id: 'numerics/linalg/logDet',
  arity: 1,
  impl: ([a], p) => determinant(factorFor(a, p, 'logDet')).logAbs,
  vjp: (g, [a], _y, p) => [mul(g, inverseTransposed(a, p))],
  doc: { summary: 'log |det A|.' },
  test: { secondOrder: true, cases: (draw) => [{ inputs: [wellConditioned(draw, 3)], params: {} }] },
})

/**
 * log |det A| for a square matrix, from its LU factorisation, without the overflow of forming det A (−∞ for a
 * singular matrix). The sign is that of `det(A)`; for a positive-definite matrix, `choleskyLogDet` is cheaper.
 */
export function logDet<A extends Value>(a: A): NumberResult<A> {
  return logDetOp([a], { factor: luFactor(a) }) as NumberResult<A>
}

/** The sign of det A (−1, 0 or 1), from its LU factorisation. */
export function signDet(a: Tensor): number {
  return determinant(factorOf(a, 'signDet')).sign
}
