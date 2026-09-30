/**
 * Symmetric eigendecomposition by the cyclic Jacobi method (Jacobi, 1846; Golub and Van Loan, 2013, Algorithm 8.5.3,
 * with Rutishauser's (1971) stable rotation formulas). Each rotation zeroes one off-diagonal pair; sweeps repeat until
 * the off-diagonal mass is negligible. Jacobi is slower than tridiagonal QR but simple and accurate: eigenvalues are
 * found to within about ε‖A‖, and small ones of well-scaled matrices to high relative accuracy (Demmel and Veselić,
 * 1992).
 */

import type { Tensor } from 'aifn/tensor'
import { denseSquare, EPS, matrix, untraced, vector } from './dense'

/** The result of `eigh`. */
export type Eigh = {
  /** Eigenvalues in descending order. */
  values: Tensor
  /**
   * Orthonormal eigenvectors as columns, column j for `values[j]`. Each is signed so that its largest-magnitude
   * component (the first of equals) is positive.
   */
  vectors: Tensor
  /** Number of Jacobi sweeps used. */
  sweeps: number
  /** False when `maxSweeps` ran out before the off-diagonal mass fell below tolerance. */
  converged: boolean
}

/**
 * Eigendecomposition A = V diag(λ) Vᵀ of a symmetric matrix (only its lower triangle is read), with eigenvalues in
 * descending order and eigenvectors as the columns of V.
 */
export function eigh(a: Tensor, { maxSweeps = 100 }: { maxSweeps?: number } = {}): Eigh {
  untraced(a, 'eigh')
  const { n, a: A } = denseSquare(a, 'eigh')
  // Symmetrise from the lower triangle, as LAPACK's dsyevd with UPLO='L'.
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) A[i * n + j] = A[j * n + i]
  const V = new Float64Array(n * n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  let total = 0
  for (let k = 0; k < n * n; k++) total += A[k] * A[k]
  const tolerance = (EPS * Math.sqrt(total)) ** 2
  let sweeps = 0
  let converged = false
  for (; sweeps <= maxSweeps; sweeps++) {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q]
    if (off <= tolerance) {
      converged = true
      break
    }
    if (sweeps === maxSweeps) break
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q]
        if (apq === 0) continue
        const app = A[p * n + p]
        const aqq = A[q * n + q]
        // tan of the rotation angle, the smaller root of t² + 2θt − 1 = 0 (Rutishauser).
        const theta = (aqq - app) / (2 * apq)
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p]
          const akq = A[k * n + q]
          A[k * n + p] = c * akp - s * akq
          A[k * n + q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k]
          const aqk = A[q * n + k]
          A[p * n + k] = c * apk - s * aqk
          A[q * n + k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p]
          const vkq = V[k * n + q]
          V[k * n + p] = c * vkp - s * vkq
          V[k * n + q] = s * vkp + c * vkq
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[j * n + j] - A[i * n + i])
  const values = new Float64Array(n)
  const vectors = new Float64Array(n * n)
  order.forEach((src, col) => {
    values[col] = A[src * n + src]
    let big = 0
    for (let k = 0; k < n; k++) if (Math.abs(V[k * n + src]) > Math.abs(V[big * n + src])) big = k
    const flip = V[big * n + src] < 0 ? -1 : 1
    for (let k = 0; k < n; k++) vectors[k * n + col] = flip * V[k * n + src]
  })
  return { values: vector(values), vectors: matrix(vectors, n, n), sweeps, converged }
}
