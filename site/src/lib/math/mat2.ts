/** Closed-form helpers for 2×2 matrices, written [[a, b], [c, d]] row by row. */

export type Vec2 = [number, number]
export type Mat2 = [[number, number], [number, number]]

export const apply = ([[a, b], [c, d]]: Mat2, [x, y]: Vec2): Vec2 => [a * x + b * y, c * x + d * y]

export const det = ([[a, b], [c, d]]: Mat2) => a * d - b * c

export const unit = ([x, y]: Vec2): Vec2 => {
  const n = Math.hypot(x, y)
  return n > 1e-12 ? [x / n, y / n] : [1, 0]
}

/** Eigenvalues of a symmetric matrix [[p, q], [q, r]], largest first, with orthonormal eigenvectors. */
export function eigSym(p: number, q: number, r: number): { values: Vec2; vectors: [Vec2, Vec2] } {
  const mid = (p + r) / 2
  const rad = Math.hypot((p - r) / 2, q)
  const angle = 0.5 * Math.atan2(2 * q, p - r)
  const v1: Vec2 = [Math.cos(angle), Math.sin(angle)]
  return { values: [mid + rad, mid - rad], vectors: [v1, [-v1[1], v1[0]]] }
}

/**
 * Eigenvalues of a general 2×2 matrix. Real ones come with unit eigenvectors, larger first. A complex pair returns
 * `complex` with the real and imaginary parts of λ = re ± i·im, and no real eigenvectors.
 */
export type Eig2 = { kind: 'real'; values: Vec2; vectors: [Vec2, Vec2] } | { kind: 'complex'; re: number; im: number }

export function eig2(m: Mat2): Eig2 {
  const [[a, b], [c, d]] = m
  const tr = a + d
  const disc = (tr * tr) / 4 - det(m)
  if (disc < 0) return { kind: 'complex', re: tr / 2, im: Math.sqrt(-disc) }
  const root = Math.sqrt(disc)
  const values: Vec2 = [tr / 2 + root, tr / 2 - root]
  // (A − λI)v = 0: use whichever row of A − λI is larger, for accuracy.
  const vector = (l: number): Vec2 => {
    const r1: Vec2 = [a - l, b]
    const r2: Vec2 = [c, d - l]
    const row = Math.hypot(...r1) >= Math.hypot(...r2) ? r1 : r2
    if (Math.hypot(...row) < 1e-12) return [1, 0]
    return unit([-row[1], row[0]])
  }
  return { kind: 'real', values, vectors: [vector(values[0]), vector(values[1])] }
}

/** Lower-triangular Cholesky factor of a symmetric positive-definite [[p, q], [q, r]], or null if it is not PD. */
export function cholesky2(p: number, q: number, r: number): Mat2 | null {
  if (p <= 0) return null
  const l11 = Math.sqrt(p)
  const l21 = q / l11
  const s = r - l21 * l21
  if (s <= 0) return null
  return [
    [l11, 0],
    [l21, Math.sqrt(s)],
  ]
}

export type Svd2 = { s: Vec2; u: [Vec2, Vec2]; v: [Vec2, Vec2] }

/** Closed-form SVD of [[a, b], [c, d]], via the eigendecomposition of AᵀA. */
export function svd2(a: number, b: number, c: number, d: number): Svd2 {
  // AᵀA = [[p, q], [q, r]] is symmetric positive semi-definite.
  const {
    values: [l1, l2],
    vectors: [v1, v2],
  } = eigSym(a * a + c * c, a * b + c * d, b * b + d * d)
  const s: Vec2 = [Math.sqrt(l1), Math.sqrt(Math.max(l2, 0))]
  const m: Mat2 = [
    [a, b],
    [c, d],
  ]
  const direction = (w: Vec2, fallback: Vec2): Vec2 => {
    const n = Math.hypot(w[0], w[1])
    return n > 1e-12 ? [w[0] / n, w[1] / n] : fallback
  }
  const u1 = direction(apply(m, v1), [1, 0])
  // For a rank-deficient A, choose u₂ perpendicular to u₁ so that U stays orthogonal.
  const u2 = direction(apply(m, v2), [-u1[1], u1[0]])
  return { s, u: [u1, u2], v: [v1, v2] }
}
