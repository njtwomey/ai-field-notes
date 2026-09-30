/**
 * A dense solve for small systems inside inner loops, on row-major `Float64Array`s rather than tensors: the LU
 * elimination with partial pivoting of `luFactor` (Golub & Van Loan, 2013, "Matrix Computations", 4th ed., Algorithm
 * 3.4.1), without the tensor wrapping. It reports a singular matrix instead of throwing, so iterative methods can react
 * (damp, regularise, stop).
 */

import type { Size } from 'aifn/foundation/contracts'
import { ShapeError } from 'aifn/foundation/errors'
import { factor, substitute } from './lu'

/** The result of `solveDense`. */
export type DenseSolution = {
  /** X (n × k, row-major; length n for one right-hand side), or null when A is singular. */
  x: Float64Array | null
  /** log |det A| (−∞ when singular). */
  logAbsDet: number
  /** True when a pivot is at most n·ε·max|A| (or A has a non-finite entry). */
  singular: boolean
}

/**
 * Solve A X = B for a row-major n × n matrix `a` and a row-major n × k right-hand side `b` (k = b.length / n; a vector
 * for k = 1). Neither input is modified. `singular` is set, and `x` is null, when a pivot is at most n·ε·max|A|.
 */
export function solveDense(a: ArrayLike<number>, b: ArrayLike<number>, n: Size): DenseSolution {
  if (a.length !== n * n)
    throw new ShapeError('solveDense', `solveDense: a has ${a.length} entries, expected ${n}×${n}`)
  if (n === 0) return { x: new Float64Array(0), logAbsDet: 0, singular: false }
  if (b.length % n !== 0)
    throw new ShapeError('solveDense', `solveDense: b has ${b.length} entries, not a multiple of ${n}`)
  const k = b.length / n
  const m = Float64Array.from(a)
  for (let i = 0; i < m.length; i++) if (!Number.isFinite(m[i])) return { x: null, logAbsDet: NaN, singular: true }
  const { perm, singular } = factor({ m: n, n, a: m })
  if (singular) return { x: null, logAbsDet: -Infinity, singular: true }
  let logAbsDet = 0
  for (let i = 0; i < n; i++) logAbsDet += Math.log(Math.abs(m[i * n + i]))
  const x = new Float64Array(n * k)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) x[i * k + j] = b[perm[i] * k + j]
  substitute(m, n, x, k)
  return { x, logAbsDet, singular: false }
}
