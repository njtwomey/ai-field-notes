/**
 * Relevance vector machine regression by MacKay's re-estimation updates (Tipping 2001, section 2.2). The basis is a
 * bias plus one Gaussian bump per training input; a basis function is pruned once its precision α exceeds PRUNE.
 */

type Matrix = number[][]

// A weight with prior variance below 1e-6 contributes nothing visible, and waiting for α to diverge further is slow.
const PRUNE = 1e6
const MAX_ITER = 2000

function cholesky(a: Matrix): Matrix {
  const n = a.length
  const l: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      l[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / l[j][j]
    }
  }
  return l
}

/** Inverse of a symmetric positive definite matrix from its Cholesky factor. */
function inverse(a: Matrix): Matrix {
  const l = cholesky(a)
  const n = l.length
  // Invert L column by column, then Σ = L⁻ᵀ L⁻¹.
  const li: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let j = 0; j < n; j++) {
    li[j][j] = 1 / l[j][j]
    for (let i = j + 1; i < n; i++) {
      let s = 0
      for (let k = j; k < i; k++) s -= l[i][k] * li[k][j]
      li[i][j] = s / l[i][i]
    }
  }
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = 0
      for (let k = i; k < n; k++) s += li[k][i] * li[k][j]
      out[i][j] = s
      out[j][i] = s
    }
  return out
}

export type RvmFit = {
  /** Indices of the retained basis functions; 0 is the bias, i + 1 is the bump at training input i. */
  active: number[]
  mean: number[]
  covariance: Matrix
  alpha: number[]
  beta: number
  iterations: number
}

export const bump = (x: number, c: number, width: number) => Math.exp(-((x - c) ** 2) / (2 * width * width))

/** φ_j(x) for basis index j over the training inputs xs. */
export const basis = (j: number, x: number, xs: number[], width: number) => (j === 0 ? 1 : bump(x, xs[j - 1], width))

export function fitRvm(xs: number[], y: number[], width: number): RvmFit {
  const n = xs.length
  const phiAll = xs.map((x) => Array.from({ length: n + 1 }, (_, j) => basis(j, x, xs, width)))
  const yMean = y.reduce((s, v) => s + v, 0) / n
  const yVar = y.reduce((s, v) => s + (v - yMean) ** 2, 0) / n
  let active = Array.from({ length: n + 1 }, (_, j) => j)
  let alpha = active.map(() => 1)
  let beta = 100 / Math.max(yVar, 1e-6)
  let mean: number[] = []
  let covariance: Matrix = []
  let iterations = 0
  for (; iterations < MAX_ITER; iterations++) {
    const m = active.length
    const phi = phiAll.map((row) => active.map((j) => row[j]))
    // Posterior precision A + β ΦᵀΦ and mean β Σ Φᵀ y.
    const precision: Matrix = Array.from({ length: m }, (_, a) =>
      Array.from({ length: m }, (_, b) => {
        let s = 0
        for (let k = 0; k < n; k++) s += phi[k][a] * phi[k][b]
        return beta * s + (a === b ? alpha[a] : 0)
      }),
    )
    covariance = inverse(precision)
    const pty = Array.from({ length: m }, (_, a) => phi.reduce((s, row, k) => s + row[a] * y[k], 0))
    mean = covariance.map((row) => beta * row.reduce((s, v, b) => s + v * pty[b], 0))
    const gamma = alpha.map((a, i) => 1 - a * covariance[i][i])
    const resid = phi.reduce((s, row, k) => s + (y[k] - row.reduce((t, v, a) => t + v * mean[a], 0)) ** 2, 0)
    const nextAlpha = gamma.map((g, i) => g / Math.max(mean[i] ** 2, 1e-300))
    const nextBeta = Math.max(n - gamma.reduce((s, g) => s + g, 0), 1e-6) / Math.max(resid, 1e-12)
    const change = Math.max(
      ...nextAlpha.map((a, i) => Math.abs(Math.log(Math.min(a, PRUNE) / Math.min(alpha[i], PRUNE)))),
    )
    alpha = nextAlpha
    beta = nextBeta
    const keep = alpha.map((a, i) => (a < PRUNE ? i : -1)).filter((i) => i >= 0)
    if (keep.length < active.length) {
      active = keep.map((i) => active[i])
      alpha = keep.map((i) => alpha[i])
      continue
    }
    if (change < 1e-3) break
  }
  // Final posterior for the retained set.
  const phi = phiAll.map((row) => active.map((j) => row[j]))
  const m = active.length
  const precision: Matrix = Array.from({ length: m }, (_, a) =>
    Array.from({ length: m }, (_, b) => {
      let s = 0
      for (let k = 0; k < n; k++) s += phi[k][a] * phi[k][b]
      return beta * s + (a === b ? alpha[a] : 0)
    }),
  )
  covariance = inverse(precision)
  const pty = Array.from({ length: m }, (_, a) => phi.reduce((s, row, k) => s + row[a] * y[k], 0))
  mean = covariance.map((row) => beta * row.reduce((s, v, b) => s + v * pty[b], 0))
  return { active, mean, covariance, alpha, beta, iterations }
}
