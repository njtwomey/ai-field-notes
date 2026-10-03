/** Helpers for the adaptive and statistical signal-processing notes. Figure-sized problems only. */

/** Biased autocorrelation estimate r[k] = (1/N) Σ_n x[n] x[n+k], for k = 0..maxLag. */
export function autocorrelation(x: ArrayLike<number>, maxLag: number): number[] {
  const n = x.length
  return Array.from({ length: maxLag + 1 }, (_, k) => {
    let s = 0
    for (let i = 0; i + k < n; i++) s += x[i] * x[i + k]
    return s / n
  })
}

/**
 * Levinson–Durbin recursion for the order-p predictor x̂[n] = Σ_{k=1}^{p} a_k x[n−k] from autocorrelations r[0..p].
 * Returns the predictor coefficients a_1..a_p, the reflection coefficients and the final prediction-error power.
 */
export function levinsonDurbin(r: ArrayLike<number>, p: number) {
  let a: number[] = []
  let error = r[0]
  const reflection: number[] = []
  for (let m = 1; m <= p; m++) {
    let acc = r[m]
    for (let j = 1; j < m; j++) acc -= a[j - 1] * r[m - j]
    const k = acc / error
    const next = a.map((aj, j) => aj - k * a[m - 2 - j])
    next.push(k)
    a = next
    reflection.push(k)
    error *= 1 - k * k
  }
  return { a, reflection, error }
}

/** Solve A x = b by Gaussian elimination with partial pivoting (small dense systems). */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length
  const m = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let pivot = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[pivot][c])) pivot = r
    ;[m[c], m[pivot]] = [m[pivot], m[c]]
    for (let r = c + 1; r < n; r++) {
      const f = m[r][c] / m[c][c]
      for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k]
    }
  }
  const x = new Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r][n]
    for (let k = r + 1; k < n; k++) s -= m[r][k] * x[k]
    x[r] = s / m[r][r]
  }
  return x
}

/** Symmetric Toeplitz matrix with first row r. */
export const toeplitz = (r: ArrayLike<number>): number[][] =>
  Array.from({ length: r.length }, (_, i) => Array.from({ length: r.length }, (_, j) => r[Math.abs(i - j)]))
