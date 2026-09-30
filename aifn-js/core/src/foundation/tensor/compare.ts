/** Whole-tensor comparisons for tests and assertions (not primitives: they return booleans). */

import { flatData, type Tensor } from './core'
import { binaryKernel } from './kernels'
import { unwrap, type Value } from './tape'

/** Options for `allclose`. */
export type CloseOptions = {
  /** Relative tolerance (default 1e-5, as NumPy). */
  rtol?: number
  /** Absolute tolerance (default 1e-8, as NumPy). */
  atol?: number
  /** Count NaN as equal to NaN (default false). */
  equalNan?: boolean
}

/**
 * True when |a − b| ≤ atol + rtol·|b| for every broadcast pair, as `np.allclose` (note the asymmetry: b is the
 * reference). Infinities must match exactly. Incompatible shapes are an error.
 */
export function allclose(
  a: Value,
  b: Value,
  { rtol = 1e-5, atol = 1e-8, equalNan = false }: CloseOptions = {},
): boolean {
  const close = (x: number, y: number) => {
    if (x !== x || y !== y) return equalNan && x !== x && y !== y ? 1 : 0
    if (x === y) return 1
    if (!Number.isFinite(x) || !Number.isFinite(y)) return 0
    return Math.abs(x - y) <= atol + rtol * Math.abs(y) ? 1 : 0
  }
  const ra = unwrap(a)
  const rb = unwrap(b)
  if (typeof ra === 'number' && typeof rb === 'number') return close(ra, rb) === 1
  return binaryKernel(ra, rb, close, 'int32').data.every((v) => v === 1)
}

/**
 * True when a and b have the same shape and equal elements (NaN equals nothing), as `np.array_equal`. The dtypes may
 * differ.
 */
export function equal(a: Value, b: Value): boolean {
  const ra = unwrap(a)
  const rb = unwrap(b)
  if (typeof ra === 'number' || typeof rb === 'number') {
    if (typeof ra === 'number' && typeof rb === 'number') return ra === rb
    const t = (typeof ra === 'number' ? rb : ra) as Tensor
    const x = (typeof ra === 'number' ? ra : rb) as number
    return t.shape.length === 0 && flatData(t)[0] === x
  }
  if (ra.shape.length !== rb.shape.length || ra.shape.some((d, k) => d !== rb.shape[k])) return false
  const fa = flatData(ra)
  const fb = flatData(rb)
  for (let k = 0; k < fa.length; k++) if (fa[k] !== fb[k]) return false
  return true
}
