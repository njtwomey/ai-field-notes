/**
 * Bayesian linear regression with prior w ~ N(0, α⁻¹I) and noise precision β, for the figures in this note. Matrices
 * are small (at most 13 × 13), so plain loops are enough.
 */

export type Matrix = number[][]

const zeros = (n: number): Matrix => Array.from({ length: n }, () => new Array<number>(n).fill(0))

/** Lower-triangular L with A = L Lᵀ, for symmetric positive definite A. */
export function cholesky(a: Matrix): Matrix {
  const n = a.length
  const l = zeros(n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      l[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / l[j][j]
    }
  }
  return l
}

/** Solves A x = b given the Cholesky factor of A. */
export function cholSolve(l: Matrix, b: number[]): number[] {
  const n = l.length
  const z = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= l[i][k] * z[k]
    z[i] = s / l[i][i]
  }
  const x = new Array<number>(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]
    for (let k = i + 1; k < n; k++) s -= l[k][i] * x[k]
    x[i] = s / l[i][i]
  }
  return x
}

/** Eigenvalues of a symmetric matrix by cyclic Jacobi rotations. */
export function symmetricEigenvalues(input: Matrix): number[] {
  const a = input.map((row) => [...row])
  const n = a.length
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2
    if (off < 1e-22) break
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const akp = a[k][p]
          const akq = a[k][q]
          a[k][p] = c * akp - s * akq
          a[k][q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k]
          const aqk = a[q][k]
          a[p][k] = c * apk - s * aqk
          a[q][k] = s * apk + c * aqk
        }
      }
    }
  }
  return a.map((row, i) => row[i])
}

/** ΦᵀΦ and Φᵀy for a design matrix Φ given as rows φ(xₙ)ᵀ. */
export function gram(phi: Matrix, y: number[]): { ptp: Matrix; pty: number[] } {
  const m = phi[0]?.length ?? 0
  const ptp = zeros(m)
  const pty = new Array<number>(m).fill(0)
  phi.forEach((row, n) => {
    for (let i = 0; i < m; i++) {
      pty[i] += row[i] * y[n]
      for (let j = 0; j < m; j++) ptp[i][j] += row[i] * row[j]
    }
  })
  return { ptp, pty }
}

export type Posterior = {
  /** Posterior mean m_N. */
  mean: number[]
  /** Posterior precision A = S_N⁻¹ = αI + βΦᵀΦ. */
  precision: Matrix
  /** Cholesky factor of the precision. */
  factor: Matrix
}

/** Posterior over w: A = αI + βΦᵀΦ and m_N = β A⁻¹ Φᵀy. */
export function posterior(phi: Matrix, y: number[], alpha: number, beta: number, m: number): Posterior {
  const { ptp, pty } = phi.length ? gram(phi, y) : { ptp: zeros(m), pty: new Array<number>(m).fill(0) }
  const precision = ptp.map((row, i) => row.map((v, j) => beta * v + (i === j ? alpha : 0)))
  const factor = cholesky(precision)
  const mean = cholSolve(
    factor,
    pty.map((v) => beta * v),
  )
  return { mean, precision, factor }
}

/** Parameter variance φᵀ S_N φ at one input, with S_N = A⁻¹. */
export function parameterVariance(post: Posterior, f: number[]): number {
  const s = cholSolve(post.factor, f)
  return f.reduce((acc, v, i) => acc + v * s[i], 0)
}

/** Log marginal likelihood ln p(y | α, β) in the form M/2 ln α + N/2 ln β − E(m_N) − ½ ln|A| − N/2 ln 2π. */
export function logEvidence(phi: Matrix, y: number[], alpha: number, beta: number, post: Posterior): number {
  const n = y.length
  const m = post.mean.length
  const resid = phi.reduce((acc, row, k) => acc + (y[k] - row.reduce((s, v, i) => s + v * post.mean[i], 0)) ** 2, 0)
  const energy = (beta / 2) * resid + (alpha / 2) * post.mean.reduce((s, v) => s + v * v, 0)
  const logDet = 2 * post.factor.reduce((s, row, i) => s + Math.log(row[i]), 0)
  return (m / 2) * Math.log(alpha) + (n / 2) * Math.log(beta) - energy - logDet / 2 - (n / 2) * Math.log(2 * Math.PI)
}

/**
 * Evidence maximisation by the fixed-point updates γ = Σ λᵢ/(α + λᵢ), α = γ / m_Nᵀm_N, β = (N − γ) / ‖y − Φm_N‖²,
 * where λᵢ are the eigenvalues of βΦᵀΦ.
 */
export function maximiseEvidence(
  phi: Matrix,
  y: number[],
  alpha: number,
  beta: number,
): { alpha: number; beta: number; gamma: number; iterations: number } {
  const m = phi[0].length
  const eig = symmetricEigenvalues(gram(phi, y).ptp).map((v) => Math.max(v, 0))
  let gamma = 0
  let iterations = 0
  for (; iterations < 500; iterations++) {
    const post = posterior(phi, y, alpha, beta, m)
    gamma = eig.reduce((s, mu) => s + (beta * mu) / (alpha + beta * mu), 0)
    const norm = post.mean.reduce((s, v) => s + v * v, 0)
    const resid = phi.reduce((acc, row, k) => acc + (y[k] - row.reduce((s, v, i) => s + v * post.mean[i], 0)) ** 2, 0)
    const nextAlpha = gamma / Math.max(norm, 1e-12)
    const nextBeta = (y.length - gamma) / Math.max(resid, 1e-12)
    const done = Math.abs(nextAlpha - alpha) < 1e-9 * alpha && Math.abs(nextBeta - beta) < 1e-9 * beta
    alpha = nextAlpha
    beta = nextBeta
    if (done) break
  }
  return { alpha, beta, gamma, iterations }
}

/** Effective number of well-determined parameters γ = Σ λᵢ/(α + λᵢ), λᵢ eigenvalues of βΦᵀΦ. */
export function effectiveParameters(phi: Matrix, y: number[], alpha: number, beta: number): number {
  if (!phi.length) return 0
  return symmetricEigenvalues(gram(phi, y).ptp).reduce(
    (s, mu) => s + (beta * Math.max(mu, 0)) / (alpha + beta * Math.max(mu, 0)),
    0,
  )
}
