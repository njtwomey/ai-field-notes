export type Vec = [number, number]
export type Cov = [[number, number], [number, number]]

const trace = (m: Cov) => m[0][0] + m[1][1]
const mul = (a: Cov, b: Cov): Cov => [
  [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
  [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
]
const det = (m: Cov) => m[0][0] * m[1][1] - m[0][1] * m[1][0]

/**
 * Fréchet distance between N(μ₁, Σ₁) and N(μ₂, Σ₂) in two dimensions: ‖μ₁ − μ₂‖² + tr Σ₁ + tr Σ₂ − 2 tr (Σ₁Σ₂)^{1/2},
 * with tr (Σ₁Σ₂)^{1/2} = √(tr M + 2√det M) for M = Σ₁Σ₂, whose eigenvalues are real and non-negative.
 */
export function frechet2d(m1: Vec, s1: Cov, m2: Vec, s2: Cov) {
  const m = mul(s1, s2)
  const traceSqrt = Math.sqrt(trace(m) + 2 * Math.sqrt(Math.max(det(m), 0)))
  const meanTerm = (m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2
  const covTerm = trace(s1) + trace(s2) - 2 * traceSqrt
  return { meanTerm, covTerm, fid: meanTerm + covTerm }
}
