/**
 * Soft-margin support vector machine training by sequential minimal optimisation, for the figures in this category.
 * The working pair is the maximal violating pair (Keerthi et al. 2001, as in LIBSVM), which converges quickly on the
 * few dozen points these figures use.
 */

export type Point = [number, number]
export type Kernel2 = (a: Point, b: Point) => number

export const linearKernel: Kernel2 = (a, b) => a[0] * b[0] + a[1] * b[1]
export const rbfKernel =
  (ell: number): Kernel2 =>
  (a, b) =>
    Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * ell * ell))

export type SvmFit = {
  /** Dual variables, one per training point, in [0, C]. */
  alpha: number[]
  b: number
  iterations: number
  /** Largest KKT violation m(α) − M(α) at termination. */
  gap: number
}

/**
 * Solves min ½ αᵀQα − 1ᵀα subject to 0 ≤ α ≤ C and yᵀα = 0, with Q_ij = y_i y_j k(x_i, x_j) and labels y ∈ {−1, +1}.
 */
export function trainSvm(x: Point[], y: number[], C: number, kernel: Kernel2, tol = 1e-4, maxIter = 20000): SvmFit {
  const n = x.length
  const K = x.map((a) => x.map((b) => kernel(a, b)))
  const alpha = new Array<number>(n).fill(0)
  // Gradient of the dual objective, G = Qα − 1; zero α gives G = −1.
  const G = new Array<number>(n).fill(-1)
  let iterations = 0
  let gap = Infinity
  const up = (t: number) => (y[t] > 0 ? alpha[t] < C : alpha[t] > 0)
  const low = (t: number) => (y[t] > 0 ? alpha[t] > 0 : alpha[t] < C)
  for (; iterations < maxIter; iterations++) {
    let i = -1
    let j = -1
    let m = -Infinity
    let M = Infinity
    for (let t = 0; t < n; t++) {
      const v = -y[t] * G[t]
      if (up(t) && v > m) {
        m = v
        i = t
      }
      if (low(t) && v < M) {
        M = v
        j = t
      }
    }
    gap = m - M
    if (i < 0 || j < 0 || gap < tol) break
    // Move α_i by +y_i s and α_j by −y_j s, which keeps yᵀα fixed; the objective falls along s at rate m − M.
    const eta = Math.max(K[i][i] + K[j][j] - 2 * K[i][j], 1e-12)
    let s = gap / eta
    s = Math.min(s, y[i] > 0 ? C - alpha[i] : alpha[i])
    s = Math.min(s, y[j] > 0 ? alpha[j] : C - alpha[j])
    alpha[i] += y[i] * s
    alpha[j] -= y[j] * s
    for (let t = 0; t < n; t++) G[t] += y[t] * s * (K[t][i] - K[t][j])
  }
  // Bias from the free support vectors, where y_t f(x_t) = 1 exactly; otherwise the middle of the feasible interval.
  let sum = 0
  let count = 0
  let m = -Infinity
  let M = Infinity
  for (let t = 0; t < n; t++) {
    const v = -y[t] * G[t]
    if (alpha[t] > 1e-8 && alpha[t] < C - 1e-8) {
      sum += v
      count++
    }
    if (up(t)) m = Math.max(m, v)
    if (low(t)) M = Math.min(M, v)
  }
  const b = count > 0 ? sum / count : (m + M) / 2
  return { alpha, b, iterations, gap }
}

/** f(x) = Σ αᵢ yᵢ k(xᵢ, x) + b. */
export function decision(fit: SvmFit, x: Point[], y: number[], kernel: Kernel2, at: Point): number {
  let s = fit.b
  for (let i = 0; i < x.length; i++) if (fit.alpha[i] > 0) s += fit.alpha[i] * y[i] * kernel(x[i], at)
  return s
}
