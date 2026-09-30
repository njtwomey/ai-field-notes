/**
 * QR factorisation by Householder reflections (Golub and Van Loan, 2013, Algorithm 5.2.1), with LAPACK's sign
 * convention (`dgeqrf`/`dlarfg`): each reflector maps its column to −sign(α)·‖x‖·e₁, so R's diagonal may be negative,
 * and a column that is already zero below the diagonal is left alone. This matches NumPy's `np.linalg.qr`.
 */

import type { Tensor } from 'aifn/tensor'
import { dense, matrix, untraced } from './dense'

/** The result of `qr`. */
export type QR = {
  /** Orthonormal columns: m×k for `reduced` (k = min(m, n)), m×m for `complete`. */
  Q: Tensor
  /** Upper triangular (trapezoidal): k×n for `reduced`, m×n for `complete`. */
  R: Tensor
}

/**
 * QR factorisation A = QR of an m×n matrix by Householder reflections. `mode` `reduced` (default) gives the thin
 * factors; `complete` gives a square Q.
 */
export function qr(a: Tensor, { mode = 'reduced' }: { mode?: 'reduced' | 'complete' } = {}): QR {
  untraced(a, 'qr')
  const { m, n, a: r } = dense(a, 'qr')
  const k = Math.min(m, n)
  // Householder vectors v (v[0] = 1 implied by storing it explicitly) and their scalars τ, with H = I − τvvᵀ.
  const vs: Float64Array[] = []
  const taus: number[] = []
  for (let j = 0; j < k; j++) {
    const alpha = r[j * n + j]
    let sigma = 0
    for (let i = j + 1; i < m; i++) sigma += r[i * n + j] * r[i * n + j]
    const v = new Float64Array(m - j)
    v[0] = 1
    if (sigma === 0) {
      // Nothing to annihilate: H = I (τ = 0), as dlarfg.
      vs.push(v)
      taus.push(0)
      continue
    }
    const norm = Math.hypot(alpha, Math.sqrt(sigma))
    const beta = alpha >= 0 ? -norm : norm
    const tau = (beta - alpha) / beta
    const scale = 1 / (alpha - beta)
    for (let i = j + 1; i < m; i++) v[i - j] = r[i * n + j] * scale
    // Apply H to the trailing columns of R.
    for (let c = j; c < n; c++) {
      let s = 0
      for (let i = j; i < m; i++) s += v[i - j] * r[i * n + c]
      s *= tau
      for (let i = j; i < m; i++) r[i * n + c] -= s * v[i - j]
    }
    r[j * n + j] = beta
    for (let i = j + 1; i < m; i++) r[i * n + j] = 0
    vs.push(v)
    taus.push(tau)
  }
  const cols = mode === 'complete' ? m : k
  // Q = H₀H₁⋯H_{k−1} applied to the first `cols` columns of I, accumulated backwards.
  const q = new Float64Array(m * cols)
  for (let i = 0; i < Math.min(m, cols); i++) q[i * cols + i] = 1
  for (let j = k - 1; j >= 0; j--) {
    const v = vs[j]
    const tau = taus[j]
    if (tau === 0) continue
    for (let c = 0; c < cols; c++) {
      let s = 0
      for (let i = j; i < m; i++) s += v[i - j] * q[i * cols + c]
      s *= tau
      for (let i = j; i < m; i++) q[i * cols + c] -= s * v[i - j]
    }
  }
  const rows = mode === 'complete' ? m : k
  return { Q: matrix(q, m, cols), R: matrix(r.slice(0, rows * n), rows, n) }
}
