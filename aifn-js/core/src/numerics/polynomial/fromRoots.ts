/**
 * The monic polynomial with given roots, by multiplying out the linear factors (s − r) in complex arithmetic (the
 * convolution form of numpy's `poly`; Press et al., 2007, "Numerical Recipes", 3rd ed., §5.1). Used for pole placement.
 */

import { dense, fromData, type Vector } from 'aifn/foundation/tensor'
import type { VectorLike } from 'aifn/foundation/contracts'
import { DomainError, ShapeError } from 'aifn/foundation/errors'

/** Desired closed-loop poles: real parts and imaginary parts (complex poles in conjugate pairs). */
export type PoleSet = { real: VectorLike; imag?: VectorLike }

/**
 * The monic polynomial with the given roots, coefficients highest degree first ([1, c₁, …, cₙ]). Complex roots must
 * come in conjugate pairs so the coefficients are real; throws otherwise.
 */
export function polynomialFromRoots({ real, imag }: PoleSet): Vector {
  const re = Array.from(dense.toF64(real, 'polynomialFromRoots'))
  const im = imag === undefined ? re.map(() => 0) : Array.from(dense.toF64(imag, 'polynomialFromRoots'))
  if (im.length !== re.length)
    throw new ShapeError('polynomialFromRoots', 'polynomialFromRoots: real and imag differ in length')
  let pr = [1]
  let pi = [0]
  re.forEach((r, k) => {
    const i = im[k]
    // Multiply by (s − (r + i·j)).
    const nr = new Array(pr.length + 1).fill(0)
    const ni = new Array(pr.length + 1).fill(0)
    for (let j = 0; j < pr.length; j++) {
      nr[j] += pr[j]
      ni[j] += pi[j]
      nr[j + 1] -= pr[j] * r - pi[j] * i
      ni[j + 1] -= pr[j] * i + pi[j] * r
    }
    pr = nr
    pi = ni
  })
  const scale = Math.max(1, ...pr.map(Math.abs))
  if (pi.some((v) => Math.abs(v) > 1e-9 * scale))
    throw new DomainError('polynomialFromRoots', 'polynomialFromRoots: complex roots must come in conjugate pairs')
  return fromData(Float64Array.from(pr), [pr.length])
}
