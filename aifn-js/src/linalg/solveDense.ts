/**
 * A dense solve for small systems inside inner loops, on row-major `Float64Array`s rather than tensors: Gaussian
 * elimination with partial pivoting (Golub & Van Loan, 2013, "Matrix Computations", 4th ed., Algorithm 3.4.1). It
 * reports a singular matrix instead of throwing, so iterative methods can react (damp, regularise, stop).
 */

/** The result of `solveDense`. */
export type DenseSolution = {
  /** X (n × k, row-major; length n for one right-hand side), or null when A is singular. */
  x: Float64Array | null
  /** log |det A| (−∞ when singular). */
  logAbsDet: number
  /** True when a pivot is at most n·ε·max|A| (or A has a non-finite entry). */
  singular: boolean
}

/**
 * Solve A X = B for a row-major n × n matrix `a` and a row-major n × k right-hand side `b` (k = b.length / n; a vector
 * for k = 1). Neither input is modified. `singular` is set, and `x` is null, when a pivot is at most n·ε·max|A|.
 */
export function solveDense(a: ArrayLike<number>, b: ArrayLike<number>, n: number): DenseSolution {
  if (a.length !== n * n) throw new Error(`solveDense: a has ${a.length} entries, expected ${n}×${n}`)
  if (n === 0) return { x: new Float64Array(0), logAbsDet: 0, singular: false }
  if (b.length % n !== 0) throw new Error(`solveDense: b has ${b.length} entries, not a multiple of ${n}`)
  const k = b.length / n
  const m = Float64Array.from(a)
  const x = Float64Array.from(b)
  let big = 0
  for (let i = 0; i < m.length; i++) {
    if (!Number.isFinite(m[i])) return { x: null, logAbsDet: NaN, singular: true }
    big = Math.max(big, Math.abs(m[i]))
  }
  const tolerance = n * 2 ** -52 * big
  let logAbsDet = 0
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r * n + c]) > Math.abs(m[p * n + c])) p = r
    if (!(Math.abs(m[p * n + c]) > tolerance)) return { x: null, logAbsDet: -Infinity, singular: true }
    if (p !== c) {
      for (let j = 0; j < n; j++) [m[c * n + j], m[p * n + j]] = [m[p * n + j], m[c * n + j]]
      for (let j = 0; j < k; j++) [x[c * k + j], x[p * k + j]] = [x[p * k + j], x[c * k + j]]
    }
    const pivot = m[c * n + c]
    logAbsDet += Math.log(Math.abs(pivot))
    for (let r = c + 1; r < n; r++) {
      const f = m[r * n + c] / pivot
      if (f === 0) continue
      for (let j = c; j < n; j++) m[r * n + j] -= f * m[c * n + j]
      for (let j = 0; j < k; j++) x[r * k + j] -= f * x[c * k + j]
    }
  }
  for (let r = n - 1; r >= 0; r--)
    for (let j = 0; j < k; j++) {
      let s = x[r * k + j]
      for (let q = r + 1; q < n; q++) s -= m[r * n + q] * x[q * k + j]
      x[r * k + j] = s / m[r * n + r]
    }
  return { x, logAbsDet, singular: false }
}
