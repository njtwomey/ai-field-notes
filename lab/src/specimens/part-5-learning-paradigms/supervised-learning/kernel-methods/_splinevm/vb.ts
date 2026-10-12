/**
 * Variational GAM with adaptive, ARD-like smoothing: a cubic spline term under any loss, whose flexibility beyond a
 * plain GAM has to be paid for. A proof of concept with local numerics (design: .scratch/splinevm/design.md).
 *
 * Model (u = x rescaled to [0, 1], fixed knot grid t_1 … t_M):
 *
 *   κ ~ Gamma(a_κ, b_κ)                         the GAM's shared smoothing level (vague)
 *   α_j | κ ~ Gamma(ν, ν κ)                     each knot's relevance; E[α_j | κ] = 1/κ
 *   w_j | α_j ~ N(0, 1/α_j)                      jump in f‴ at t_j
 *   f(u) = Σ_k c_k u^k + Σ_j w_j (u − t_j)₊³      (c vague)
 *   y_n | f ~ p_ℓ(y_n | f(u_n), s)               any loss read as a likelihood, with a scale s where it has one
 *
 * ν prices departures from the GAM: ν → ∞ pins every α_j to the shared level (a Bayesian penalised spline, the GAM
 * smoothing term); small ν lets single knots switch on or off ARD-style where the data pay for it.
 *
 * Variational Bayes: q(w) q(α) q(κ) with q(w) = N(μ, Σ), q(α_j) and q(κ) Gamma (closed form). The expected
 * log-likelihood is a sum of 1-D Gaussian expectations over f_n ~ N(m_n, v_n), by Gauss–Hermite quadrature for any
 * loss; its derivatives in m_n use loss values only (g = E[z ℓ]/√v, h = E[(z² − 1) ℓ]/v), so non-smooth losses work.
 * q(w): Σ = (E[A] + Φᵀ diag(−h) Φ)⁻¹ and a damped Newton step on μ (Opper and Archambeau, 2009).
 */
import { standardNormals, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

// ── Special functions and quadrature ──

const LOG2PI = Math.log(2 * Math.PI)

export function lgamma(x: number): number {
  // Lanczos (g = 7, n = 9).
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x)
  x -= 1
  let a = c[0]
  const t = x + 7.5
  for (let i = 1; i < 9; i++) a += c[i] / (x + i)
  return 0.5 * LOG2PI + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

export function digamma(x: number): number {
  let r = 0
  while (x < 6) {
    r -= 1 / x
    x += 1
  }
  const f = 1 / (x * x)
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
}

function erf(x: number): number {
  // Numerical Recipes erfc by Chebyshev fit, |error| < 1.2e-7.
  const z = Math.abs(x)
  const t = 1 / (1 + 0.5 * z)
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    )
  return x >= 0 ? 1 - r : r - 1
}
export const normalCdf = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2))

/** Gauss–Hermite nodes and weights for E[g(z)], z ~ N(0, 1) (Numerical Recipes `gauher`, rescaled). */
function gaussHermite(n: number) {
  const x = new Float64Array(n)
  const w = new Float64Array(n)
  const PIM4 = 0.7511255444649425
  const m = (n + 1) >> 1
  let z = 0
  for (let i = 0; i < m; i++) {
    if (i === 0) z = Math.sqrt(2 * n + 1) - 1.85575 * Math.pow(2 * n + 1, -0.16667)
    else if (i === 1) z -= (1.14 * Math.pow(n, 0.426)) / z
    else if (i === 2) z = 1.86 * z - 0.86 * x[0]
    else if (i === 3) z = 1.91 * z - 0.91 * x[1]
    else z = 2 * z - x[i - 2]
    let pp = 0
    for (let its = 0; its < 50; its++) {
      let p1 = PIM4
      let p2 = 0
      for (let j = 0; j < n; j++) {
        const p3 = p2
        p2 = p1
        p1 = z * Math.sqrt(2 / (j + 1)) * p2 - Math.sqrt(j / (j + 1)) * p3
      }
      pp = Math.sqrt(2 * n) * p2
      const z1 = z
      z = z1 - p1 / pp
      if (Math.abs(z - z1) <= 1e-14) break
    }
    x[i] = z
    x[n - 1 - i] = -z
    w[i] = 2 / (pp * pp)
    w[n - 1 - i] = w[i]
  }
  return {
    z: Array.from(x, (v) => v * Math.SQRT2),
    w: Array.from(w, (v) => v / Math.sqrt(Math.PI)),
  }
}
const GH = gaussHermite(24)

// ── Losses as likelihoods ──

export type LossKey = 'gaussian' | 'student' | 'laplace' | 'huber' | 'pinball' | 'logistic' | 'poisson'

export const LOSSES: { value: LossKey; label: string }[] = [
  { value: 'gaussian', label: 'Gaussian (squared error)' },
  { value: 'student', label: 'Student-t, ν = 4 (robust)' },
  { value: 'laplace', label: 'Laplace (absolute error)' },
  { value: 'huber', label: 'Huber, δ = 1.345' },
  { value: 'pinball', label: 'pinball (quantile τ)' },
  { value: 'logistic', label: 'logistic (Bernoulli)' },
  { value: 'poisson', label: 'Poisson (log link)' },
]

export type Loss = {
  key: LossKey
  /** log p(y | f, s), normalised. */
  ll: (y: number, f: number, s: number) => number
  scaled: boolean
  /** Not twice differentiable: q is floored at a small spread so the curvature stays finite. */
  kinked: boolean
  /** What to draw for f: the identity, the class probability, the rate. */
  link: (f: number) => number
}

const logSigmoid = (f: number) => (f >= 0 ? -Math.log1p(Math.exp(-f)) : f - Math.log1p(Math.exp(f)))
const HUBER = 1.345
const HUBER_LOGZ = Math.log(
  Math.sqrt(2 * Math.PI) * (2 * normalCdf(HUBER) - 1) + (2 * Math.exp(-(HUBER * HUBER) / 2)) / HUBER,
)
const NU_T = 4
const T_LOGC = lgamma((NU_T + 1) / 2) - lgamma(NU_T / 2) - 0.5 * Math.log(NU_T * Math.PI)

export function makeLoss(key: LossKey, tau = 0.5): Loss {
  const id = (f: number) => f
  switch (key) {
    case 'gaussian':
      return {
        key,
        scaled: true,
        kinked: false,
        link: id,
        ll: (y, f, s) => -0.5 * ((y - f) / s) ** 2 - Math.log(s) - 0.5 * LOG2PI,
      }
    case 'student':
      return {
        key,
        scaled: true,
        kinked: false,
        link: id,
        ll: (y, f, s) => T_LOGC - Math.log(s) - ((NU_T + 1) / 2) * Math.log1p(((y - f) / s) ** 2 / NU_T),
      }
    case 'laplace':
      return { key, scaled: true, kinked: true, link: id, ll: (y, f, s) => -Math.abs(y - f) / s - Math.log(2 * s) }
    case 'huber':
      return {
        key,
        scaled: true,
        kinked: false,
        link: id,
        ll: (y, f, s) => {
          const r = Math.abs((y - f) / s)
          return -(r <= HUBER ? 0.5 * r * r : HUBER * r - 0.5 * HUBER * HUBER) - Math.log(s) - HUBER_LOGZ
        },
      }
    case 'pinball':
      return {
        key,
        scaled: true,
        kinked: true,
        link: id,
        ll: (y, f, s) => {
          const r = (y - f) / s
          return Math.log(tau * (1 - tau)) - Math.log(s) - r * (tau - (r < 0 ? 1 : 0))
        },
      }
    case 'logistic':
      return {
        key,
        scaled: false,
        kinked: false,
        link: (f) => 1 / (1 + Math.exp(-f)),
        ll: (y, f) => (y > 0.5 ? logSigmoid(f) : logSigmoid(-f)),
      }
    case 'poisson':
      // The rate is capped at e^30 so that wide early q's do not overflow.
      return {
        key,
        scaled: false,
        kinked: false,
        link: Math.exp,
        ll: (y, f) => y * f - Math.exp(Math.min(f, 30)) - lgamma(y + 1),
      }
  }
}

/** E[ℓ], and its first and second derivatives in the mean m, for f ~ N(m, v). */
function expected(loss: Loss, y: number, m: number, v: number, s: number) {
  const sd = Math.sqrt(v)
  let e = 0
  let g = 0
  let h = 0
  for (let q = 0; q < GH.z.length; q++) {
    const z = GH.z[q]
    const l = loss.ll(y, m + sd * z, s)
    e += GH.w[q] * l
    g += GH.w[q] * z * l
    h += GH.w[q] * (z * z - 1) * l
  }
  return { e, g: g / sd, h: h / v }
}

// ── Randomness and data ──

function rng(seed: number | string) {
  const s: Stream = stream(seed)
  let u: Float64Array = new Float64Array(0)
  let ui = 0
  let z: Float64Array = new Float64Array(0)
  let zi = 0
  return {
    u() {
      if (ui >= u.length) {
        u = Float64Array.from(toFlat(uniform(s, 0, 1, { shape: [4096] }) as Tensor))
        ui = 0
      }
      return u[ui++]
    },
    z() {
      if (zi >= z.length) {
        z = standardNormals(s, 4096)
        zi = 0
      }
      return z[zi++]
    },
  }
}

export type Task = 'reg' | 'clf' | 'count'
export type NoiseKind = 'gaussian' | 'heavy' | 'outliers' | 'skewed'

export type DataSpec = {
  task: Task
  /** Regression: f(x) = A sin(2π k x) + H σ((x − c)/w). */
  amplitude: number
  cycles: number
  stepHeight: number
  stepAt: number
  stepWidth: number
  noiseKind: NoiseKind
  noise: number
  /** Classification and counts: the logit or log-rate m + A sin(2π k x). */
  mean: number
  linkAmplitude: number
  linkCycles: number
  seed: number
}

export const N_MAX = 800

export type Dataset = { xs: number[]; ys: number[]; truth: (x: number) => number }

/**
 * N_MAX points in a random order, so that the first n of them are a sample of size n and samples are nested; `truth`
 * is the mean of y at x on the response scale (the location for regression).
 */
export function makeData(spec: DataSpec): Dataset {
  const r = rng(`splinevm-vb-${spec.task}-${spec.seed}`)
  const sig = (v: number) => 1 / (1 + Math.exp(-v))
  const xs = Array.from({ length: N_MAX }, () => r.u())
  if (spec.task === 'reg') {
    const f = (x: number) =>
      spec.amplitude * Math.sin(2 * Math.PI * spec.cycles * x) +
      spec.stepHeight * sig((x - spec.stepAt) / Math.max(spec.stepWidth, 1e-4))
    const eps = () => {
      switch (spec.noiseKind) {
        case 'gaussian':
          return r.z()
        case 'heavy': {
          // Student-t with 2 degrees of freedom.
          const a = r.z()
          const b = r.z()
          const c = r.z()
          return a / Math.sqrt((b * b + c * c) / 2)
        }
        case 'outliers':
          return r.u() < 0.08 ? 8 * r.z() : r.z()
        case 'skewed':
          // Exponential(1) minus its mean: a long right tail.
          return -Math.log(1 - r.u()) - 1
      }
    }
    return { xs, ys: xs.map((x) => f(x) + spec.noise * eps()), truth: f }
  }
  const link = (x: number) => spec.mean + spec.linkAmplitude * Math.sin(2 * Math.PI * spec.linkCycles * x)
  if (spec.task === 'clf') return { xs, ys: xs.map((x) => (r.u() < sig(link(x)) ? 1 : 0)), truth: (x) => sig(link(x)) }
  const poisson = (rate: number) => {
    // Knuth for small rates, normal approximation beyond.
    if (rate > 60) return Math.max(0, Math.round(rate + Math.sqrt(rate) * r.z()))
    const L = Math.exp(-rate)
    let k = 0
    let p = 1
    do {
      k++
      p *= r.u()
    } while (p > L)
    return k - 1
  }
  return { xs, ys: xs.map((x) => poisson(Math.exp(link(x)))), truth: (x) => Math.exp(link(x)) }
}

// ── The model and its variational fit ──

export type VbOptions = {
  loss: LossKey
  tau: number
  /** Knots, starting on an evenly spaced grid. */
  knots: number
  /** ν: how far a knot's relevance may depart from the shared GAM level (large: plain GAM; small: ARD-like). */
  adaptivity: number
  /**
   * Learn the knot locations: the prior sd of each knot around its grid point, in grid spacings (t_j ~ N(g_j, (c h)²)).
   * 0 or absent keeps the grid fixed.
   */
  knotLeash?: number
  maxIter: number
}

export type VbState = {
  iteration: number
  /** The knot locations at this state (u units). */
  knots: number[]
  mu: Float64Array
  sigma: Float64Array
  /** E[α_j] per knot, and the GAM level 1/E[κ]. */
  alpha: number[]
  gamLevel: number
  scale: number
  /** The ELBO, plus log p(t) of the knot locations when they are learned. */
  elbo: number
  /** log p(t) of the knot locations, up to a constant (0 on a fixed grid): subtract it to compare ELBOs. */
  knotLogPrior: number
  /** Σ_n E[log p(y_n | f_n)]. */
  fit: number
  /** fit − ELBO: the KL from the prior, in nats. */
  complexity: number
  /** Bits learned about each knot weight, ½ log₂(1/(E[α_j] Σ_jj)), and γ_j = 1 − E[α_j] Σ_jj. */
  bits: number[]
  gamma: number[]
}

export type VbFit = {
  states: VbState[]
  /** The starting grid (u units); each state carries its own knots. */
  knots: number[]
  lo: number
  span: number
  loss: Loss
  ms: number
}

const P = 4
/** Shape of the hyperprior on κ: small, so it is vague on a log scale (its rate is set from the data scale). */
const A_K = 1e-3

function designRow(u: number, knots: readonly number[], out: Float64Array, offset: number) {
  out[offset] = 1
  out[offset + 1] = u
  out[offset + 2] = u * u
  out[offset + 3] = u * u * u
  for (let j = 0; j < knots.length; j++) {
    const d = u - knots[j]
    out[offset + P + j] = d > 0 ? d * d * d : 0
  }
}

/**
 * A symmetric positive-definite matrix factored for solves, its inverse and its log determinant. The truncated-cubic
 * basis is badly conditioned (columns for knots near the right end are tiny), so the matrix is first scaled to unit
 * diagonal, D⁻¹AD⁻¹ with D = diag(√A_kk), and the Cholesky factor is taken of that, with no added jitter: the scaling is
 * exact and changes nothing but the rounding. An earlier version floored each pivot at 10⁻¹² of its diagonal, which on
 * this basis distorted Σ badly.
 */
function factor(a: Float64Array, m: number) {
  const d = Array.from({ length: m }, (_, k) => Math.sqrt(Math.max(a[k * m + k], 1e-300)))
  const l = new Float64Array(m * m)
  for (let j = 0; j < m; j++) {
    let piv = 1
    for (let k = 0; k < j; k++) piv -= l[j * m + k] ** 2
    const ljj = Math.sqrt(Math.max(piv, 1e-300))
    l[j * m + j] = ljj
    for (let i = j + 1; i < m; i++) {
      let v = a[i * m + j] / (d[i] * d[j])
      for (let k = 0; k < j; k++) v -= l[i * m + k] * l[j * m + k]
      l[i * m + j] = v / ljj
    }
  }
  const solveScaled = (b: ArrayLike<number>) => {
    const z = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      let v = b[i]
      for (let k = 0; k < i; k++) v -= l[i * m + k] * z[k]
      z[i] = v / l[i * m + i]
    }
    const x = new Float64Array(m)
    for (let i = m - 1; i >= 0; i--) {
      let v = z[i]
      for (let k = i + 1; k < m; k++) v -= l[k * m + i] * x[k]
      x[i] = v / l[i * m + i]
    }
    return x
  }
  let logDet = 0
  for (let k = 0; k < m; k++) logDet += 2 * Math.log(l[k * m + k]) + 2 * Math.log(d[k])
  return {
    /** A⁻¹ b */
    solve(b: ArrayLike<number>) {
      const x = solveScaled(Array.from(b, (v, k) => v / d[k]))
      return x.map((v, k) => v / d[k])
    },
    inverse() {
      const inv = new Float64Array(m * m)
      const e = new Float64Array(m)
      for (let c = 0; c < m; c++) {
        e.fill(0)
        e[c] = 1 / d[c]
        const col = solveScaled(e)
        for (let r = 0; r < m; r++) inv[r * m + c] = col[r] / d[r]
      }
      return inv
    },
    /** log det A */
    logDet,
  }
}
type Factor = ReturnType<typeof factor>

/** Entropy of Gamma(a, b) (rate b). */
const gammaEntropy = (a: number, b: number) => a - Math.log(b) + lgamma(a) + (1 - a) * digamma(a)

export function fitVb(data: { xs: number[]; ys: number[] }, opts: VbOptions): VbFit {
  const t0 = performance.now()
  const loss = makeLoss(opts.loss, opts.tau)
  const n = data.xs.length
  const lo = Math.min(...data.xs)
  const span = Math.max(...data.xs) - lo || 1
  const M = opts.knots
  const grid = Array.from({ length: M }, (_, j) => (j + 1) / (M + 1))
  const m = P + M
  const us = data.xs.map((x) => (x - lo) / span)
  const buildPhi = (t: readonly number[]) => {
    const out = new Float64Array(n * m)
    us.forEach((u, i) => designRow(u, t, out, i * m))
    return out
  }
  let knots = grid.slice()
  let phi = buildPhi(knots)
  // Learned locations: t_j ~ N(g_j, (c h)²) around the grid point g_j, h the spacing; the log prior joins the ELBO.
  const leashSd = (opts.knotLeash ?? 0) * (1 / (M + 1))
  const logPriorKnots = (t: readonly number[]) =>
    leashSd > 0 ? t.reduce((a, v, j) => a - (v - grid[j]) ** 2 / (2 * leashSd * leashSd), 0) : 0
  let knotStep = 0.2 / (M + 1)
  const y = data.ys
  const nu = opts.adaptivity

  // Scale of the response, to start the scale s and the relevances.
  const ybar = y.reduce((a, b) => a + b, 0) / n
  const sdY = Math.sqrt(y.reduce((a, b) => a + (b - ybar) ** 2, 0) / n) || 1
  const alphaPoly = loss.key === 'gaussian' || loss.scaled ? 1e-6 / (sdY * sdY) : 1e-2
  const h = 1 / (M + 1)
  const R = loss.scaled ? 3 * sdY : 6
  const alpha0 = (h ** 3 / R) ** 2
  // κ behaves like a prior variance of the weights (E[α_j | κ] = 1/κ), and on this basis it is of order 1/α₀, which
  // can be 10⁷ or more. A Gamma(10⁻³, 10⁻³) hyperprior would put a factor e^(−10⁻³ κ) on it, a wall that drags the
  // smoothing level down and over-smooths. So the rate is scaled to the data: E[κ] = 1/α₀ a priori, vague around it.
  const B_K = A_K * alpha0

  // Variational state.
  let mu = new Float64Array(m)
  if (loss.scaled) mu[0] = ybar
  else if (loss.key === 'poisson') mu[0] = Math.log(Math.max(ybar, 0.1))
  const qa = new Array(M).fill(nu + 0.5)
  const qb = new Array(M).fill((nu + 0.5) / alpha0)
  // q(κ) starts with E[κ] = 1/α₀, so the relevances start at the shared level α₀.
  let qka = A_K + M * nu
  let qkb = qka * alpha0
  let s = loss.scaled ? sdY : 1
  let sigma = new Float64Array(m * m)
  const Ealpha = () => qa.map((a, j) => a / qb[j])
  const precisionOf = (k: number, ea: number[]) => (k < P ? alphaPoly : ea[k - P])

  /** m_n and v_n of f_n under q(w) = N(mu, sig). */
  const moments = (mu_: Float64Array, sig: Float64Array, ph: Float64Array = phi) => {
    const mm = new Float64Array(n)
    const vv = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let a = 0
      let v = 0
      for (let r = 0; r < m; r++) {
        const pr = ph[i * m + r]
        if (pr === 0) continue
        a += pr * mu_[r]
        let row = 0
        for (let c = 0; c < m; c++) row += sig[r * m + c] * ph[i * m + c]
        v += pr * row
      }
      mm[i] = a
      vv[i] = Math.max(v, 1e-12)
    }
    return { mm, vv }
  }
  const floorV = (v: number) => (loss.kinked ? Math.max(v, (0.02 * s) ** 2) : v)
  const expectedFit = (mm: Float64Array, vv: Float64Array, scale: number) => {
    let e = 0
    for (let i = 0; i < n; i++) e += expected(loss, y[i], mm[i], floorV(vv[i]), scale).e
    return e
  }

  /** The ELBO (up to constants that no variational parameter changes). */
  const elboOf = (mu_: Float64Array, sig: Float64Array, fac: Factor, fit: number) => {
    const ea = Ealpha()
    const eLogA = qa.map((a, j) => digamma(a) - Math.log(qb[j]))
    const eK = qka / qkb
    const eLogK = digamma(qka) - Math.log(qkb)
    // E[log p(w | α)] + H[q(w)], with H[q(w)] = ½ log det Σ = −½ log det(precision) (constants dropped).
    let wTerm = -0.5 * fac.logDet
    for (let k = 0; k < m; k++) {
      const prec = precisionOf(k, ea)
      const logPrec = k < P ? Math.log(alphaPoly) : eLogA[k - P]
      wTerm += 0.5 * logPrec - 0.5 * prec * (mu_[k] ** 2 + sig[k * m + k])
    }
    // E[log p(α | κ)] + H[q(α)]
    let aTerm = 0
    for (let j = 0; j < M; j++)
      aTerm +=
        nu * eLogK + nu * Math.log(nu) - lgamma(nu) + (nu - 1) * eLogA[j] - nu * eK * ea[j] + gammaEntropy(qa[j], qb[j])
    // E[log p(κ)] + H[q(κ)]
    const kTerm = A_K * Math.log(B_K) - lgamma(A_K) + (A_K - 1) * eLogK - B_K * eK + gammaEntropy(qka, qkb)
    return fit + wTerm + aTerm + kTerm + logPriorKnots(knots)
  }

  const states: VbState[] = []
  let fac: Factor = factor(
    Float64Array.from({ length: m * m }, (_, k) => (k % (m + 1) === 0 ? 1 : 0)),
    m,
  )
  const record = (iteration: number, fit: number) => {
    const ea = Ealpha()
    const elbo = elboOf(mu, sigma, fac, fit)
    const bits = ea.map((a, j) => Math.max(0, 0.5 * Math.log2(1 / Math.max(a * sigma[(P + j) * m + P + j], 1e-300))))
    states.push({
      iteration,
      knots: knots.slice(),
      knotLogPrior: logPriorKnots(knots),
      mu: mu.slice(),
      sigma: sigma.slice(),
      alpha: ea,
      gamLevel: qkb / qka,
      scale: s,
      elbo,
      fit,
      complexity: fit - elbo,
      bits,
      gamma: ea.map((a, j) => Math.min(1, Math.max(0, 1 - a * sigma[(P + j) * m + P + j]))),
    })
  }

  /** Gradient and precision of the objective in w: Φᵀg − Aμ and A + Φᵀ diag(−h) Φ, at f_n ~ N(m_n, v_n). */
  const newtonParts = (mm: Float64Array, vv: Float64Array, ph: Float64Array = phi, mu_: Float64Array = mu) => {
    const ea = Ealpha()
    const prec = new Float64Array(m * m)
    const grad = new Float64Array(m)
    for (let i = 0; i < n; i++) {
      const d = expected(loss, y[i], mm[i], floorV(vv[i]), s)
      const Hi = Math.max(-d.h, 1e-10)
      const row = i * m
      for (let r = 0; r < m; r++) {
        const pr = ph[row + r]
        if (pr === 0) continue
        grad[r] += pr * d.g
        for (let c = r; c < m; c++) prec[r * m + c] += Hi * pr * ph[row + c]
      }
    }
    for (let r = 0; r < m; r++) {
      prec[r * m + r] += precisionOf(r, ea)
      grad[r] -= precisionOf(r, ea) * mu_[r]
      for (let c = 0; c < r; c++) prec[r * m + c] = prec[c * m + r]
    }
    return { prec, grad }
  }

  /**
   * q(α_j) = Gamma(ν + ½, ν E[κ] + ½ E[w_j²]) and q(κ) = Gamma(a_κ + M ν, b_κ + ν Σ E[α_j]), solved jointly. For large ν
   * the two are so tightly coupled that alternating them moves E[κ] by a fraction 1/(2ν) per pass, and the GAM's
   * smoothing level never leaves its start. Their joint fixed point solves g(k) = k (b_κ + ν Σ_j E[α_j](k)) − (a_κ + Mν)
   * = 0 for k = E[κ], with E[α_j](k) = (ν + ½)/(ν k + ½ E[w_j²]); g increases in k, so bisection on log k finds it.
   */
  const updateRelevances = () => {
    const e = Array.from({ length: M }, (_, j) => mu[P + j] ** 2 + sigma[(P + j) * m + P + j])
    const g = (k: number) => {
      let sum = 0
      for (let j = 0; j < M; j++) sum += (nu + 0.5) / (nu * k + 0.5 * e[j])
      return k * (B_K + nu * sum) - (A_K + M * nu)
    }
    let lo = -700
    let hi = 700
    for (let it = 0; it < 200; it++) {
      const mid = (lo + hi) / 2
      if (g(Math.exp(mid)) < 0) lo = mid
      else hi = mid
    }
    const k = Math.exp((lo + hi) / 2)
    for (let j = 0; j < M; j++) {
      qa[j] = nu + 0.5
      qb[j] = nu * k + 0.5 * e[j]
    }
    qka = A_K + M * nu
    qkb = (A_K + M * nu) / k
  }

  // Start at the MAP point (Newton on the plain log-likelihood plus the prior at the starting relevances), with q(w)
  // the Laplace approximation there. Starting from the prior instead, q(w) is so wide that the expected curvature of a
  // logistic, Poisson or pinball loss is nearly zero and the first step overshoots.
  {
    // A kinked loss (absolute, pinball) is smoothed at a third of the noise scale for the start, as if Huberised;
    // smooth losses use the plain loss.
    const tiny = new Float64Array(n).fill(loss.kinked ? (sdY / 3) ** 2 : 1e-6 * (loss.scaled ? sdY * sdY : 1))
    const objective = (w: Float64Array) => {
      const ea = Ealpha()
      let o = 0
      for (let i = 0; i < n; i++) {
        let f = 0
        for (let k = 0; k < m; k++) f += phi[i * m + k] * w[k]
        o += loss.kinked ? expected(loss, y[i], f, tiny[i], s).e : loss.ll(y[i], f, s)
      }
      for (let k = 0; k < m; k++) o -= 0.5 * precisionOf(k, ea) * w[k] ** 2
      return o
    }
    let current = objective(mu)
    for (let it = 0; it < 40; it++) {
      const { mm } = moments(mu, new Float64Array(m * m))
      const { prec, grad } = newtonParts(mm, tiny)
      const step = factor(prec, m).solve(grad)
      let moved = false
      for (let damp = 1, tries = 0; tries < 20; tries++, damp /= 2) {
        const trial = mu.map((v, k) => v + damp * step[k])
        const o = objective(trial)
        if (o > current) {
          moved = o - current > 1e-8
          mu = trial
          current = o
          break
        }
      }
      if (!moved) break
    }
    const { mm } = moments(mu, new Float64Array(m * m))
    const { prec } = newtonParts(mm, tiny)
    fac = factor(prec, m)
    sigma = fac.inverse()
    updateRelevances()
    const mom = moments(mu, sigma)
    record(0, expectedFit(mom.mm, mom.vv, s))
  }

  for (let it = 1; it <= opts.maxIter; it++) {
    const before = states[states.length - 1].elbo
    // 1 · q(w): curvature and gradient of the expected log-likelihood at the current q, then a damped Newton step
    //     kept only when the ELBO does not fall.
    const { mm, vv } = moments(mu, sigma)
    const { prec, grad } = newtonParts(mm, vv)
    const facNew = factor(prec, m)
    const sigNew = facNew.inverse()
    const step = facNew.solve(grad)
    for (let damp = 1, tries = 0; tries < 12; tries++, damp /= 2) {
      const trial = mu.map((v, k) => v + damp * step[k])
      const mom = moments(trial, sigNew)
      const fit = expectedFit(mom.mm, mom.vv, s)
      const e = elboOf(trial, sigNew, facNew, fit)
      if (e >= before - 1e-9) {
        mu = trial
        sigma = sigNew
        fac = facNew
        break
      }
    }
    // 2 · q(α_j) = Gamma(ν + ½, ν E[κ] + ½ E[w_j²]); q(κ) = Gamma(a_κ + M ν, b_κ + ν Σ E[α_j]).
    updateRelevances()
    // 3 · The loss scale s: maximise the expected log-likelihood in log s (closed form for the Gaussian).
    const mom = moments(mu, sigma)
    if (loss.scaled) {
      if (loss.key === 'gaussian') {
        let ss = 0
        for (let i = 0; i < n; i++) ss += (y[i] - mom.mm[i]) ** 2 + mom.vv[i]
        s = Math.sqrt(ss / n)
      } else {
        // Golden-section search on log s.
        let a = Math.log(s) - 2
        let b = Math.log(s) + 2
        const gr = 0.6180339887
        let c = b - gr * (b - a)
        let d = a + gr * (b - a)
        const fv = (ls: number) => expectedFit(mom.mm, mom.vv, Math.exp(ls))
        let fc = fv(c)
        let fd = fv(d)
        for (let k = 0; k < 30; k++) {
          if (fc > fd) {
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = fv(c)
          } else {
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = fv(d)
          }
        }
        s = Math.exp((a + b) / 2)
      }
    }
    // 4 · Knot locations (when learned): one normalised gradient step on ELBO + log p(t), kept when it rises. Only the
    //     fit term and the knot prior depend on t; by Bonnet and Price, ∂/∂φ_n Σ E[ℓ] = g_n μ + h_n Σ φ_n, and
    //     ∂φ_nj/∂t_j = −3 (u_n − t_j)₊².
    if (leashSd > 0) {
      const now = moments(mu, sigma)
      const G = new Float64Array(n)
      const Hh = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const dv = expected(loss, y[i], now.mm[i], floorV(now.vv[i]), s)
        G[i] = dv.g
        Hh[i] = dv.h
      }
      const grad = new Array(M).fill(0)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < M; j++) {
          const dd = us[i] - knots[j]
          if (dd <= 0) continue
          const c = P + j
          let sphi = 0
          for (let k = 0; k < m; k++) sphi += sigma[c * m + k] * phi[i * m + k]
          grad[j] += (G[i] * mu[c] + Hh[i] * sphi) * (-3 * dd * dd)
        }
      for (let j = 0; j < M; j++) grad[j] -= (knots[j] - grid[j]) / (leashSd * leashSd)
      const gmax = Math.max(...grad.map(Math.abs))
      if (gmax > 0) {
        // A trial position is scored with q(w) refitted to it (one Newton step from the current mean, exact for the
        // Gaussian loss): the old weights are tuned to the old basis, so scoring a move with them rejects nearly all.
        const base = states[states.length - 1].elbo
        for (let tries = 0; tries < 8; tries++) {
          const trial = knots.map((t, j) => Math.min(1 - 1e-4, Math.max(1e-4, t + (knotStep * grad[j]) / gmax)))
          const ph = buildPhi(trial)
          const mt0 = moments(mu, sigma, ph)
          const parts = newtonParts(mt0.mm, mt0.vv, ph, mu)
          const facT = factor(parts.prec, m)
          const sigT = facT.inverse()
          const stepW = facT.solve(parts.grad)
          const muT = mu.map((v, k) => v + stepW[k])
          const mt = moments(muT, sigT, ph)
          const eT = elboOf(muT, sigT, facT, expectedFit(mt.mm, mt.vv, s)) - logPriorKnots(knots) + logPriorKnots(trial)
          if (eT > base) {
            knots = trial
            phi = ph
            mu = muT
            sigma = sigT
            fac = facT
            knotStep = Math.min(knotStep * 1.3, 1 / (M + 1))
            break
          }
          knotStep *= 0.5
        }
        knotStep = Math.max(knotStep, 1e-5)
      }
    }
    const momEnd = moments(mu, sigma)
    record(it, expectedFit(momEnd.mm, momEnd.vv, s))
    const now = states[states.length - 1].elbo
    const back = states.length > 10 ? states[states.length - 10].elbo : -Infinity
    if (Math.abs(now - before) < 1e-6 || now - back < 1e-3) break
  }
  return { states, knots: grid, lo, span, loss, ms: performance.now() - t0 }
}

/** The latent f at points x for a state: mean and variance (no noise). */
export function predictVb(fit: VbFit, st: VbState, x: number[]) {
  const m = P + st.knots.length
  const row = new Float64Array(m)
  const mean: number[] = []
  const variance: number[] = []
  for (const xv of x) {
    designRow((xv - fit.lo) / fit.span, st.knots, row, 0)
    let a = 0
    let v = 0
    for (let r = 0; r < m; r++) {
      if (row[r] === 0) continue
      a += row[r] * st.mu[r]
      let s = 0
      for (let c = 0; c < m; c++) s += st.sigma[r * m + c] * row[c]
      v += row[r] * s
    }
    mean.push(a)
    variance.push(Math.max(v, 0))
  }
  return { mean, variance }
}
