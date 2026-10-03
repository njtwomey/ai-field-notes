/**
 * Siegmund's approximation to the average run length of a one-sided CUSUM on N(μ, 1) increments with reference k and
 * threshold h: with Δ = μ − k and b = h + 1.166, ARL ≈ (e^{−2Δb} + 2Δb − 1)/(2Δ²), or b² when Δ = 0.
 */
export function averageRunLength(mu: number, k: number, h: number): number {
  const d = mu - k
  const b = h + 1.166
  return Math.abs(d) < 1e-9 ? b * b : (Math.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d * d)
}
