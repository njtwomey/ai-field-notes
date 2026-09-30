/**
 * Polynomial roots as the eigenvalues of the companion matrix, as numpy's `roots` computes them (Edelman & Murakami,
 * 1995, "Polynomial roots from companion matrix eigenvalues", Math. Comp. 64(210)). The eigenvalues come from
 * `aifn/numerics/linalg`'s `eig` (balancing and the EISPACK `hqr` shifted QR algorithm, the one copy of it); the
 * companion matrix is already upper Hessenberg, so the Hessenberg reduction leaves it unchanged. Values by Horner's
 * rule.
 */

import { add, dense, fromData, mul, toFlat, type Tensor, type Traced, type Value } from 'aifn/foundation/tensor'
import type { Scalar, VectorLike } from 'aifn/foundation/contracts'
import { DomainError } from 'aifn/foundation/errors'
import { eig } from 'aifn/numerics/linalg'

const { toF64 } = dense

/** The roots of a polynomial: real and imaginary parts, conjugate pairs adjacent. */
export type PolynomialRoots = {
  /** Real parts (length = degree after stripping leading zeros). */
  real: Tensor
  /** Imaginary parts. */
  imag: Tensor
  /** False when the QR iteration did not converge within its iteration budget (the roots are then unreliable). */
  converged: boolean
}

/**
 * The roots of p(x) = c₀xⁿ + c₁xⁿ⁻¹ + … + cₙ, coefficients highest degree first (numpy's convention), as the
 * eigenvalues of the companion matrix. Leading zeros are stripped; trailing zeros give roots at 0. Roots are sorted by
 * real part, then imaginary part, both descending. Accuracy degrades for clustered or multiple roots (a double root is
 * found to about √ε).
 */
export function polynomialRoots(coefficients: VectorLike): PolynomialRoots {
  const c = Array.from(toF64(coefficients, 'polynomialRoots'))
  if (!c.every(Number.isFinite)) throw new DomainError('polynomialRoots', 'polynomialRoots: coefficients must be finite')
  while (c.length && c[0] === 0) c.shift()
  let zeros = 0
  while (c.length && c[c.length - 1] === 0) {
    c.pop()
    zeros++
  }
  const n = Math.max(c.length - 1, 0)
  let converged = true
  const roots: [number, number][] = []
  if (n > 0) {
    // Companion matrix: first row −c_k/c₀, ones on the subdiagonal.
    const a = new Float64Array(n * n)
    for (let j = 0; j < n; j++) a[j] = -c[j + 1] / c[0]
    for (let i = 1; i < n; i++) a[i * n + i - 1] = 1
    const e = eig(fromData(a, [n, n]), { vectors: false })
    converged = e.converged
    const re = toFlat(e.real)
    const im = toFlat(e.imag)
    for (let k = 0; k < n; k++) roots.push([re[k], im[k]])
  }
  for (let k = 0; k < zeros; k++) roots.push([0, 0])
  roots.sort((u, v) => v[0] - u[0] || v[1] - u[1])
  return {
    real: fromData(
      Float64Array.from(roots, (r) => r[0]),
      [roots.length],
    ),
    imag: fromData(
      Float64Array.from(roots, (r) => r[1]),
      [roots.length],
    ),
    converged,
  }
}

/**
 * p(x) by Horner's rule, coefficients highest degree first, elementwise in x: a number gives a number, a tensor of
 * points a tensor of the same shape, and a traced x a traced value (a composition of primitives, so differentiable in
 * x).
 */
export function polynomialValue(coefficients: VectorLike, x: Scalar): Scalar
export function polynomialValue(coefficients: VectorLike, x: Tensor): Tensor
export function polynomialValue(coefficients: VectorLike, x: Traced): Traced
export function polynomialValue(coefficients: VectorLike, x: Value): Value
export function polynomialValue(coefficients: VectorLike, x: Value): Value {
  const c = toF64(coefficients, 'polynomialValue')
  let v: Value = mul(0, x)
  for (let k = 0; k < c.length; k++) v = add(mul(v, x), c[k])
  return v
}
