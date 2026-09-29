/**
 * Gaussian-process ordinal regression with the Laplace approximation (Chu & Ghahramani 2005), for one input. The
 * likelihood of class y given the latent f is Φ((b_y − f)/σ) − Φ((b_{y−1} − f)/σ), with b_{−1} = −∞ and b_{K−1} = +∞.
 * The Newton iteration and predictive equations follow Rasmussen & Williams, Algorithms 3.1 and 3.2.
 */
import { rng } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'

type Mat = number[][]

export const kernel = (a: number, b: number, lengthscale: number, amplitude = 1.5) =>
  amplitude * amplitude * Math.exp(-((a - b) ** 2) / (2 * lengthscale * lengthscale))

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const logPdf = (z: number) => (Number.isFinite(z) ? -0.5 * z * z - LOG_SQRT_2PI : -Infinity)

/**
 * log Φ(z) for z ≤ 0 in the form of Abramowitz & Stegun 7.1.26, whose exp(−z²/2) factor is kept in the exponent so the
 * far tail neither cancels nor underflows.
 */
function logLowerTail(z: number): number {
  if (z === -Infinity) return -Infinity
  const t = 1 / (1 + (0.3275911 * -z) / Math.SQRT2)
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
  return Math.log(0.5 * poly) - 0.5 * z * z
}

/** log(Φ(hi) − Φ(lo)) for lo < hi, computed in whichever tail both ends share. */
function logInterval(lo: number, hi: number): number {
  if (hi <= 0) return logLowerTail(hi) + Math.log1p(-Math.exp(logLowerTail(lo) - logLowerTail(hi)))
  if (lo >= 0) return logLowerTail(-lo) + Math.log1p(-Math.exp(logLowerTail(-hi) - logLowerTail(-lo)))
  return Math.log(normalCdf(hi) - normalCdf(lo))
}

/** Log-likelihood of one observation and its first two derivatives in f, all computed in log space. */
function likelihood(f: number, y: number, b: number[], sigma: number) {
  const z1 = y < b.length ? (b[y] - f) / sigma : Infinity
  const z2 = y > 0 ? (b[y - 1] - f) / sigma : -Infinity
  const logP = logInterval(z2, z1)
  // N(z)/p and z N(z)/p for each end; an infinite end contributes nothing.
  const m1 = Math.exp(logPdf(z1) - logP)
  const m2 = Math.exp(logPdf(z2) - logP)
  const r = m1 - m2
  const zn = (Number.isFinite(z1) ? z1 * m1 : 0) - (Number.isFinite(z2) ? z2 * m2 : 0)
  return {
    log: logP,
    grad: -r / sigma,
    // −d²/df² log p = (r² + (z1 N(z1) − z2 N(z2))/p)/σ², positive: the ordinal likelihood is log-concave in f.
    w: Math.max((r * r + zn) / (sigma * sigma), 1e-10),
  }
}

function cholesky(a: Mat): Mat {
  const n = a.length
  const l: Mat = Array.from({ length: n }, () => Array(n).fill(0) as number[])
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      l[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / l[j][j]
    }
  return l
}

const forward = (l: Mat, b: number[]) => {
  const x: number[] = []
  for (let i = 0; i < b.length; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= l[i][k] * x[k]
    x.push(s / l[i][i])
  }
  return x
}

const backward = (l: Mat, b: number[]) => {
  const n = b.length
  const x = Array(n).fill(0) as number[]
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]
    for (let k = i + 1; k < n; k++) s -= l[k][i] * x[k]
    x[i] = s / l[i][i]
  }
  return x
}

export type GpOrdinalFit<T = number> = {
  /** The Newton iterate a with f = K a at the mode, for warm-starting a fit with nearby settings. */
  a: number[]
  /** The posterior mode of f at the training inputs. */
  mode: number[]
  /** The Laplace approximation to the log marginal likelihood log p(y | X, θ). */
  logEvidence: number
  /** Predictive mean and variance of the latent f at a new input. */
  latent: (x: T) => { mean: number; variance: number }
  /** Predictive class probabilities at a new input. */
  probs: (x: T) => number[]
}

/** The 1-D fit of the figure: a squared-exponential kernel with the given lengthscale. */
export const fitGpOrdinal = (x: number[], y: number[], b: number[], lengthscale: number, sigma: number) =>
  fitGpOrdinalWith(x, y, b, (u, v) => kernel(u, v, lengthscale), sigma)

/**
 * The Laplace fit for any input type and kernel. `warm` is the `a` of an earlier fit on the same inputs, a starting
 * point that usually leaves only a few Newton steps.
 */
export function fitGpOrdinalWith<T>(
  x: T[],
  y: number[],
  b: number[],
  kern: (u: T, v: T) => number,
  sigma: number,
  warm?: number[],
): GpOrdinalFit<T> {
  const n = x.length
  const K = x.map((u) => x.map((v) => kern(u, v)))
  let a = warm && warm.length === n ? warm.slice() : (Array(n).fill(0) as number[])
  let f = K.map((row) => row.reduce((s, k, j) => s + k * a[j], 0))
  let l: Mat = []
  let sw: number[] = []
  let terms = y.map((yi, i) => likelihood(f[i], yi, b, sigma))
  for (let it = 0; it < 40; it++) {
    sw = terms.map((t) => Math.sqrt(t.w))
    const B = K.map((row, i) => row.map((k, j) => (i === j ? 1 : 0) + sw[i] * k * sw[j]))
    l = cholesky(B)
    const bvec = f.map((fi, i) => terms[i].w * fi + terms[i].grad)
    const kb = K.map((row) => row.reduce((s, k, j) => s + k * bvec[j], 0))
    const inner = backward(
      l,
      forward(
        l,
        kb.map((v, i) => sw[i] * v),
      ),
    )
    const aNewton = bvec.map((v, i) => v - sw[i] * inner[i])
    // Newton can overshoot where the likelihood's curvature changes fast (small σ), so halve the step until the
    // objective Ψ(f) = log p(y | f) − ½ aᵀf, with f = K a, increases.
    const objective = (av: number[], fv: number[], tv: typeof terms) =>
      tv.reduce((s, t) => s + t.log, 0) - 0.5 * av.reduce((s, v, i) => s + v * fv[i], 0)
    const current = objective(a, f, terms)
    let step = 1
    let aNew = aNewton
    let fNew = f
    let termsNew = terms
    for (let halving = 0; halving < 30; halving++) {
      aNew = a.map((v, i) => v + step * (aNewton[i] - v))
      fNew = K.map((row) => row.reduce((s, k, j) => s + k * aNew[j], 0))
      termsNew = y.map((yi, i) => likelihood(fNew[i], yi, b, sigma))
      if (objective(aNew, fNew, termsNew) >= current - 1e-12) break
      step /= 2
    }
    const change = fNew.reduce((s, v, i) => s + Math.abs(v - f[i]), 0)
    f = fNew
    a = aNew
    terms = termsNew
    if (change < 1e-8 * n) break
  }
  // Refresh B's factor at the mode for the evidence and the predictive variance.
  sw = terms.map((t) => Math.sqrt(t.w))
  l = cholesky(K.map((row, i) => row.map((k, j) => (i === j ? 1 : 0) + sw[i] * k * sw[j])))
  const logEvidence =
    -0.5 * a.reduce((s, v, i) => s + v * f[i], 0) +
    terms.reduce((s, t) => s + t.log, 0) -
    l.reduce((s, row, i) => s + Math.log(row[i]), 0)
  const grads = terms.map((t) => t.grad)
  const latent = (xs: T) => {
    const ks = x.map((xi) => kern(xs, xi))
    const mean = ks.reduce((s, k, i) => s + k * grads[i], 0)
    const v = forward(
      l,
      ks.map((k, i) => sw[i] * k),
    )
    return { mean, variance: Math.max(kern(xs, xs) - v.reduce((s, vi) => s + vi * vi, 0), 1e-12) }
  }
  const probs = (xs: T) => {
    const { mean, variance } = latent(xs)
    const scale = Math.sqrt(sigma * sigma + variance)
    const cdf = [...b.map((t) => normalCdf((t - mean) / scale)), 1]
    return cdf.map((c, k) => c - (k > 0 ? cdf[k - 1] : 0))
  }
  return { a, mode: f, logEvidence, latent, probs }
}

/** 30 inputs on [−3, 3] with a smooth latent function, Gaussian noise of 0.3, and true thresholds (−1, 0, 1). */
export function gpOrdinalData(seed = 3) {
  const r = rng(seed)
  const x = Array.from({ length: 30 }, () => -3 + 6 * r.uniform()).sort((p, q) => p - q)
  const truth = (v: number) => 1.4 * Math.sin(1.2 * v) + 0.3 * v
  const y = x.map((v) => [-1, 0, 1].filter((t) => t < truth(v) + 0.3 * r.normal()).length)
  return { x, y, truth }
}
