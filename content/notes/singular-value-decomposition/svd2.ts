/** Closed-form SVD of a 2×2 matrix [[a, b], [c, d]], via the eigendecomposition of AᵀA. */
export type Vec = [number, number]
export type Svd2 = { s: [number, number]; u: [Vec, Vec]; v: [Vec, Vec] }

export function svd2(a: number, b: number, c: number, d: number): Svd2 {
  // AᵀA = [[p, q], [q, r]] is symmetric positive semi-definite.
  const p = a * a + c * c
  const q = a * b + c * d
  const r = b * b + d * d
  const mid = (p + r) / 2
  const rad = Math.hypot((p - r) / 2, q)
  const l1 = mid + rad
  const l2 = Math.max(mid - rad, 0)
  // Eigenvector of AᵀA for λ₁; the second is its perpendicular.
  const angle = 0.5 * Math.atan2(2 * q, p - r)
  const v1: Vec = [Math.cos(angle), Math.sin(angle)]
  const v2: Vec = [-v1[1], v1[0]]
  const s: [number, number] = [Math.sqrt(l1), Math.sqrt(l2)]
  const apply = (v: Vec): Vec => [a * v[0] + b * v[1], c * v[0] + d * v[1]]
  const unit = (w: Vec, fallback: Vec): Vec => {
    const n = Math.hypot(w[0], w[1])
    return n > 1e-12 ? [w[0] / n, w[1] / n] : fallback
  }
  const u1 = unit(apply(v1), [1, 0])
  // For a rank-deficient A, choose u₂ perpendicular to u₁ so that U stays orthogonal.
  const u2 = unit(apply(v2), [-u1[1], u1[0]])
  return { s, u: [u1, u2], v: [v1, v2] }
}
