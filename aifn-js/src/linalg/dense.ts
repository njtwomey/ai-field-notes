/** Internal helpers: dense row-major working copies of matrices, and the error type of this module. */

import { fromData, isTraced, isTensor, toFlat, unwrap, type Tensor, type Value } from 'aifn/tensor'

/** Machine epsilon for float64. */
export const EPS = 2 ** -52

/**
 * Raised when a linear-algebra operation cannot produce a meaningful result: a singular system passed to a solver, a
 * non-finite input, or mismatched shapes. Factorisations report such conditions in their results instead (`singular`,
 * `failed`), so a caller that must not throw can factor first and check.
 */
export class LinAlgError extends Error {
  /** What went wrong. */
  readonly kind: 'singular' | 'not-finite' | 'shape' | 'no-derivative'

  constructor(message: string, kind: LinAlgError['kind']) {
    super(message)
    this.name = 'LinAlgError'
    this.kind = kind
  }
}

/** A dense row-major working copy of a matrix: `a[i * n + j]` is element (i, j). */
export type Dense = { m: number; n: number; a: Float64Array }

/** Copy a rank-2 value (untraced) into a dense float64 working array, checking that every element is finite. */
export function dense(x: Value, where: string): Dense {
  const t = unwrap(x)
  if (!isTensor(t) || t.shape.length !== 2) {
    throw new LinAlgError(
      `${where}: expected a matrix, got ${isTensor(t) ? `shape [${t.shape.join(', ')}]` : 'a number'}`,
      'shape',
    )
  }
  const a = Float64Array.from(toFlat(t))
  for (let k = 0; k < a.length; k++) {
    if (!Number.isFinite(a[k])) throw new LinAlgError(`${where}: the matrix has a non-finite entry`, 'not-finite')
  }
  return { m: t.shape[0], n: t.shape[1], a }
}

/** As `dense`, and check that the matrix is square. */
export function denseSquare(x: Value, where: string): Dense {
  const d = dense(x, where)
  if (d.m !== d.n) throw new LinAlgError(`${where}: expected a square matrix, got ${d.m}×${d.n}`, 'shape')
  return d
}

/** For decompositions without a derivative rule: refuse traced input rather than silently drop the derivative. */
export function untraced(x: Value, where: string): void {
  if (isTraced(x)) throw new LinAlgError(`${where}: no derivative rule; it cannot be differentiated`, 'no-derivative')
}

/** Wrap a dense array as a matrix tensor. */
export function matrix(a: Float64Array, m: number, n: number): Tensor {
  return fromData(a, [m, n])
}

/** Wrap a Float64Array as a vector tensor. */
export function vector(a: Float64Array): Tensor {
  return fromData(a, [a.length])
}

/** Largest absolute entry. */
export function maxAbs(a: Float64Array): number {
  let m = 0
  for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k]))
  return m
}
