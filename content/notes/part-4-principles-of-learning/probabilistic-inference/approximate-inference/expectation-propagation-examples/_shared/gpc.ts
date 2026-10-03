/**
 * Binary Gaussian process classification with a probit likelihood and a squared-exponential kernel: EP (Rasmussen and
 * Williams, algorithms 3.5 and 3.6) and the Laplace approximation (algorithm 3.1 with probit derivatives). Dense
 * linear algebra for the small datasets of the figures.
 */
import { logNormalCdf, probitTilted, vFn, wFn } from './ep.ts'

type Mat = number[][]

export const seKernel = (ell: number, sf: number) => (a: number, b: number) =>
  sf * sf * Math.exp((-0.5 * (a - b) ** 2) / (ell * ell))

const gram = (k: (a: number, b: number) => number, a: number[], b: number[]): Mat => a.map((u) => b.map((v) => k(u, v)))

/** Lower Cholesky factor of a symmetric positive-definite matrix. */
function cholesky(A: Mat): Mat {
  const n = A.length
  const L: Mat = A.map(() => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]
    }
  }
  return L
}

/** Solve A x = b given the Cholesky factor L of A. */
function cholSolve(L: Mat, b: number[]): number[] {
  const n = L.length
  const z = new Array<number>(n).fill(0)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k]
    z[i] = s / L[i][i]
  }
  const x = new Array<number>(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]
    x[i] = s / L[i][i]
  }
  return x
}

const matVec = (A: Mat, v: number[]) => A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0))

/**
 * A Gaussian approximation N(f | μ, Σ) whose precision is K⁻¹ + diag(s) with s ≥ 0, stored by its parts so that
 * predictions use B = I + S^½ K S^½, which is well conditioned (R&W section 3.4.3).
 */
type Fit = { K: Mat; sW: number[]; L: Mat }

function factor(K: Mat, s: number[]): Fit {
  const sW = s.map((v) => Math.sqrt(Math.max(v, 0)))
  const B = K.map((row, i) => row.map((v, j) => sW[i] * v * sW[j] + (i === j ? 1 : 0)))
  return { K, sW, L: cholesky(B) }
}

/** Σ = K − K S^½ B⁻¹ S^½ K, column by column. */
function covariance({ K, sW, L }: Fit): Mat {
  const n = K.length
  const cols = K.map((_, j) => {
    const kj = K.map((row) => row[j])
    const c = cholSolve(
      L,
      kj.map((v, i) => sW[i] * v),
    )
    return matVec(
      K,
      c.map((v, i) => sW[i] * v),
    )
  })
  return K.map((row, i) => row.map((v, j) => v - cols[j][i])).slice(0, n)
}

export type GpcEpState = { sweep: number; tau: number[]; nu: number[]; mean: number[]; variance: number[] }

/** EP for GP classification. Returns the state after each sweep; entry 0 is the prior. */
export function gpcEp(x: number[], y: number[], k: (a: number, b: number) => number, sweeps: number): GpcEpState[] {
  const n = x.length
  const K = gram(k, x, x)
  const tau = new Array<number>(n).fill(0)
  const nu = new Array<number>(n).fill(0)
  let Sigma = K.map((r) => [...r])
  let mu = new Array<number>(n).fill(0)
  const states: GpcEpState[] = [
    { sweep: 0, tau: [...tau], nu: [...nu], mean: [...mu], variance: Sigma.map((r, i) => r[i]) },
  ]
  for (let sweep = 1; sweep <= sweeps; sweep++) {
    for (let i = 0; i < n; i++) {
      const cavTau = 1 / Sigma[i][i] - tau[i]
      const cavNu = mu[i] / Sigma[i][i] - nu[i]
      const t = probitTilted(cavNu / cavTau, 1 / cavTau, y[i], 0, 1)
      const dTau = 1 / t.variance - cavTau - tau[i]
      tau[i] += dTau
      nu[i] = t.mean / t.variance - cavNu
      // Rank-one update of Σ for the change in site i's precision (R&W eq. 3.70).
      if (dTau !== 0) {
        const si = Sigma.map((r) => r[i])
        const c = 1 / (1 / dTau + si[i])
        Sigma = Sigma.map((r, a) => r.map((v, b) => v - c * si[a] * si[b]))
      }
      mu = matVec(Sigma, nu)
    }
    // Recompute Σ from scratch after each sweep, to stop rounding errors accumulating.
    Sigma = covariance(factor(K, tau))
    mu = matVec(Sigma, nu)
    states.push({ sweep, tau: [...tau], nu: [...nu], mean: [...mu], variance: Sigma.map((r, i) => r[i]) })
  }
  return states
}

export type LatentPrediction = { mean: number[]; variance: number[]; prob: number[] }

/** Predictive latent mean, latent variance and class probability at test inputs, for EP sites (τ̃, ν̃). */
export function gpcEpPredict(
  x: number[],
  k: (a: number, b: number) => number,
  tau: number[],
  nu: number[],
  xs: number[],
): LatentPrediction {
  const K = gram(k, x, x)
  const fit = factor(K, tau)
  const Knu = matVec(K, nu)
  const z = cholSolve(
    fit.L,
    Knu.map((v, i) => fit.sW[i] * v),
  ).map((v, i) => fit.sW[i] * v)
  const a = nu.map((v, i) => v - z[i])
  return predictWith(fit, k, x, xs, a)
}

function predictWith(fit: Fit, k: (a: number, b: number) => number, x: number[], xs: number[], a: number[]) {
  const mean: number[] = []
  const variance: number[] = []
  const prob: number[] = []
  for (const s of xs) {
    const ks = x.map((xi) => k(xi, s))
    const m = ks.reduce((acc, v, i) => acc + v * a[i], 0)
    const c = cholSolve(
      fit.L,
      ks.map((v, i) => fit.sW[i] * v),
    )
    const v = k(s, s) - ks.reduce((acc, kv, i) => acc + kv * fit.sW[i] * c[i], 0)
    mean.push(m)
    variance.push(v)
    prob.push(Math.exp(logNormalCdf(m / Math.sqrt(1 + v))))
  }
  return { mean, variance, prob }
}

/** Laplace approximation with the probit likelihood: Newton's method for the mode, then predictions (R&W 3.1, 3.2). */
export function gpcLaplace(x: number[], y: number[], k: (a: number, b: number) => number, xs: number[]) {
  const n = x.length
  const K = gram(k, x, x)
  let f = new Array<number>(n).fill(0)
  for (let it = 0; it < 100; it++) {
    const W = f.map((fi, i) => wFn(y[i] * fi))
    const g = f.map((fi, i) => y[i] * vFn(y[i] * fi))
    const fit = factor(K, W)
    const b = f.map((fi, i) => W[i] * fi + g[i])
    const c = cholSolve(
      fit.L,
      matVec(K, b).map((v, i) => fit.sW[i] * v),
    )
    const next = matVec(
      K,
      b.map((v, i) => v - fit.sW[i] * c[i]),
    )
    const change = Math.max(...next.map((v, i) => Math.abs(v - f[i])))
    f = next
    if (change < 1e-10) break
  }
  const W = f.map((fi, i) => wFn(y[i] * fi))
  const g = f.map((fi, i) => y[i] * vFn(y[i] * fi))
  const fit = factor(K, W)
  const Sigma = covariance(fit)
  return { mode: f, variance: Sigma.map((r, i) => r[i]), predict: predictWith(fit, k, x, xs, g) }
}
