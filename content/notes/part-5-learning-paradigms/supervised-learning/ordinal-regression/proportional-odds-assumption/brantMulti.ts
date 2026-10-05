import { regularisedGammaP, sigmoid } from 'aifn-compute/numerics/special'
/**
 * Brant's test for inputs of any dimension D: a separate binary logit per cumulative split, the joint covariance of
 * their slopes, and a Wald test that all K − 1 slope vectors are equal, on D(K − 2) degrees of freedom.
 */

type Mat = number[][]

const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => Array(c).fill(0) as number[])
const mul = (a: Mat, b: Mat): Mat => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
const transpose = (a: Mat): Mat => a[0].map((_, j) => a.map((row) => row[j]))

/** Inverse of a symmetric positive definite matrix by Gauss–Jordan elimination with partial pivoting. */
function inverse(a: Mat): Mat {
  const n = a.length
  const m = a.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r
    ;[m[c], m[p]] = [m[p], m[c]]
    const d = m[c][c] || 1e-12
    for (let j = 0; j < 2 * n; j++) m[c][j] /= d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = m[r][c]
      for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j]
    }
  }
  return m.map((row) => row.slice(n))
}

/**
 * Binary logistic regression by Newton's method with a small ridge (λ) on the slopes, which keeps the estimates finite
 * when a split separates the data. Returns the coefficients and the inverse of the penalised information.
 */
function logit(x: number[][], t: number[], lambda: number) {
  const d = x[0].length
  let beta = Array(d).fill(0) as number[]
  let cov = zeros(d, d)
  for (let it = 0; it < 50; it++) {
    const g = beta.map((b, j) => (j === 0 ? 0 : -lambda * b))
    const h = zeros(d, d)
    for (let j = 1; j < d; j++) h[j][j] += lambda
    x.forEach((xi, i) => {
      const p = sigmoid(xi.reduce((s, v, j) => s + v * beta[j], 0))
      const w = p * (1 - p)
      for (let a = 0; a < d; a++) {
        g[a] += (t[i] - p) * xi[a]
        for (let b = 0; b < d; b++) h[a][b] += w * xi[a] * xi[b]
      }
    })
    cov = inverse(h)
    const step = cov.map((row) => row.reduce((s, v, j) => s + v * g[j], 0))
    beta = beta.map((b, j) => b + step[j])
    if (step.reduce((s, v) => s + Math.abs(v), 0) < 1e-9) break
  }
  return { beta, cov }
}

export type BrantResult = { chi2: number; df: number; p: number; slopes: number[][] }

/** Brant's test on inputs `x` (without the constant) and 0-based classes y with K classes. */
export function brantTest(xs: number[][], y: number[], k: number, lambda = 1e-2): BrantResult {
  const x = xs.map((v) => [1, ...v])
  const d = x[0].length
  const splits = Array.from({ length: k - 1 }, (_, j) =>
    logit(
      x,
      y.map((c) => (c > j ? 1 : 0)),
      lambda,
    ),
  )
  const pi = splits.map((s) => x.map((xi) => sigmoid(xi.reduce((acc, v, j) => acc + v * s.beta[j], 0))))
  const D = d - 1
  const size = D * (k - 1)
  // Joint covariance of all the estimates, A⁻¹ B A⁻¹: A⁻¹ is block-diagonal with each fit's inverse information, and B
  // is the empirical covariance of the stacked score contributions. Brant's closed form replaces B by its expectation
  // under the fitted probabilities; the empirical B is positive semi-definite by construction, so the statistic cannot
  // go negative when some split nearly separates the data.
  const full = d * (k - 1)
  const B = zeros(full, full)
  x.forEach((xi, i) => {
    const s = splits.flatMap((_, j) => xi.map((v) => ((y[i] > j ? 1 : 0) - pi[j][i]) * v))
    for (let a = 0; a < full; a++) for (let b = 0; b < full; b++) B[a][b] += s[a] * s[b]
  })
  const Ainv = zeros(full, full)
  splits.forEach((sp, j) => sp.cov.forEach((row, a) => row.forEach((v, b) => (Ainv[j * d + a][j * d + b] = v))))
  const cov = mul(mul(Ainv, B), Ainv)
  const slopeIndex = splits.flatMap((_, j) => Array.from({ length: D }, (_, a) => j * d + 1 + a))
  const sigma = slopeIndex.map((r) => slopeIndex.map((c) => cov[r][c]))
  const slopes = splits.map((s) => s.beta.slice(1))
  const stacked = slopes.flat()
  // Adjacent differences of the slope vectors: D(K − 2) contrasts.
  const C = zeros(D * (k - 2), size)
  for (let j = 0; j < k - 2; j++)
    for (let a = 0; a < D; a++) {
      C[j * D + a][j * D + a] = 1
      C[j * D + a][(j + 1) * D + a] = -1
    }
  const diff = C.map((row) => row.reduce((s, v, i) => s + v * stacked[i], 0))
  const S = mul(mul(C, sigma), transpose(C))
  const Sinv = inverse(S)
  const chi2 = diff.reduce((s, v, i) => s + v * Sinv[i].reduce((t, w, j) => t + w * diff[j], 0), 0)
  const df = D * (k - 2)
  return { chi2, df, p: 1 - regularisedGammaP(df / 2, chi2 / 2), slopes }
}
