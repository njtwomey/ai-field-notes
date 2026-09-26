/**
 * Two-coefficient geometry of penalised least squares in constraint form: minimise the least-squares quadratic
 * Q(w) = (w − c)ᵀH(w − c) subject to r‖w‖₁ + (1 − r)‖w‖₂² ≤ 1. Here c is the OLS solution and H = [[1, ρ], [ρ, 1]] is
 * the Gram matrix of two standardised features with correlation ρ. r = 1 is the lasso diamond, r = 0 the ridge disc,
 * anything between is an elastic-net ball.
 */
export type Vec = [number, number]

export const penalty = (w: Vec, r: number) =>
  r * (Math.abs(w[0]) + Math.abs(w[1])) + (1 - r) * (w[0] * w[0] + w[1] * w[1])

export const quadratic = (w: Vec, c: Vec, rho: number) => {
  const dx = w[0] - c[0]
  const dy = w[1] - c[1]
  return dx * dx + 2 * rho * dx * dy + dy * dy
}

/** Distance from the origin to the ball's boundary along the unit direction (cos θ, sin θ). */
export function radius(theta: number, r: number): number {
  const s = Math.abs(Math.cos(theta)) + Math.abs(Math.sin(theta))
  if (r >= 1) return 1 / s
  // Positive root of (1 − r)ρ² + r s ρ − 1 = 0.
  return (-r * s + Math.sqrt(r * r * s * s + 4 * (1 - r))) / (2 * (1 - r))
}

/** Boundary points, with θ stepping so that the four axis directions (the lasso's corners) are sampled exactly. */
export function boundary(r: number, steps = 1440): Vec[] {
  return Array.from({ length: steps + 1 }, (_, k) => {
    const theta = (2 * Math.PI * k) / steps
    const rho = radius(theta, r)
    // Snap the axis directions to exact zeros rather than 1e-17.
    const x = k % (steps / 2) === steps / 4 ? 0 : rho * Math.cos(theta)
    const y = k % (steps / 2) === 0 ? 0 : rho * Math.sin(theta)
    return [x, y]
  })
}

/**
 * The constrained minimiser: c itself when it lies inside the ball, otherwise the boundary point with the smallest
 * Q. The boundary is convex and Q is a convex quadratic, so the minimum on the boundary is the constrained optimum.
 */
export function solve(c: Vec, rho: number, r: number): { w: Vec; inside: boolean } {
  if (penalty(c, r) <= 1) return { w: c, inside: true }
  let best: Vec = [0, 0]
  let bestQ = Infinity
  for (const w of boundary(r, 7200)) {
    const q = quadratic(w, c, rho)
    if (q < bestQ) {
      bestQ = q
      best = w
    }
  }
  return { w: best, inside: false }
}

/** The level set Q(w) = q: an ellipse c + √q H^{-1/2} (cos φ, sin φ), using H's eigenvectors (1, ±1)/√2. */
export function contour(c: Vec, rho: number, q: number, steps = 120): Vec[] {
  const a = Math.sqrt(q / (1 + rho))
  const b = Math.sqrt(q / (1 - rho))
  const s = Math.SQRT1_2
  return Array.from({ length: steps + 1 }, (_, k) => {
    const phi = (2 * Math.PI * k) / steps
    const u = a * Math.cos(phi)
    const v = b * Math.sin(phi)
    return [c[0] + s * (u + v), c[1] + s * (u - v)]
  })
}
