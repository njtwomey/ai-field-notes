/**
 * Simulation and test statistics for class-conditional label noise, shared by the label-noise notes.
 *
 * Follows Poyiadzi et al. (2022) for the parametric test (logistic regression, anchors with true posterior 1/2) and
 * Yang et al. (2024) for the local-likelihood test (local linear logistic regression, sandwich variance).
 * Labels are 1 / 0 here; α = P(ỹ = 0 | y = 1) and β = P(ỹ = 1 | y = 0).
 */
import { rng } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'

export type Pt = [number, number]
export type Sample = { X: Pt[]; y: number[]; u: number[] }

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))))

/** Clean data from a two-class mixture; `u` is one uniform per point, reused for flips (common random numbers). */
function sampleFrom(centres: [Pt[], Pt[]], n: number, seed: number): Sample {
  const r = rng(seed)
  const X: Pt[] = []
  const y: number[] = []
  const u: number[] = []
  for (let i = 0; i < n; i++) {
    const label = r.uniform() < 0.5 ? 1 : 0
    const comps = centres[label]
    const c = comps[Math.floor(r.uniform() * comps.length)]
    X.push([c[0] + r.normal(), c[1] + r.normal()])
    y.push(label)
    u.push(r.uniform())
  }
  return { X, y, u }
}

/** The paper's synthetic data: unit-variance Gaussians at (1, 1) (y = 1) and (−1, −1) (y = 0), equal priors. */
export const twoGaussians = (n: number, seed: number) => sampleFrom([[[-1, -1]], [[1, 1]]], n, seed)

/** Asymmetric XOR from Yang et al. (2024): class 1 at (4, 4) and (−2, −2), class 0 at (−1, 1) and (1, −1). */
export const XOR_CENTRES: [Pt[], Pt[]] = [
  [
    [-1, 1],
    [1, -1],
  ],
  [
    [4, 4],
    [-2, -2],
  ],
]
export const asymmetricXor = (n: number, seed: number) => sampleFrom(XOR_CENTRES, n, seed)

/** True posterior P(y = 1 | x) for the asymmetric XOR mixture. */
export function xorPosterior([a, b]: Pt): number {
  const d = (cs: Pt[]) => cs.reduce((s, c) => s + Math.exp(-0.5 * ((a - c[0]) ** 2 + (b - c[1]) ** 2)), 0)
  const p1 = d(XOR_CENTRES[1])
  const p0 = d(XOR_CENTRES[0])
  return p1 / (p1 + p0)
}

/** Flip clean labels: a positive becomes 0 when u < α, a negative becomes 1 when u < β. */
export const corrupt = (s: Sample, alpha: number, beta: number) =>
  s.y.map((y, i) => (y === 1 ? (s.u[i] < alpha ? 0 : 1) : s.u[i] < beta ? 1 : 0))

function inv3(m: number[][]): number[][] {
  const [[a, b, c], [d, e, f], [g, h, i]] = m
  const A = e * i - f * h
  const B = -(d * i - f * g)
  const C = d * h - e * g
  const det = a * A + b * B + c * C
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ]
}

const quad = (x: number[], M: number[][], z: number[]) =>
  x.reduce((s, xi, j) => s + xi * M[j].reduce((t, mjk, k) => t + mjk * z[k], 0), 0)

/**
 * Weighted logistic regression on a 3-column design by Newton's method, with a tiny ridge so that separable or
 * nearly empty neighbourhoods still give a finite fit. Returns the coefficients, the inverse Hessian and the
 * per-observation weighted score contributions (for sandwich variances).
 */
function logisticNewton(D: number[][], y: number[], w: number[], ridge = 1e-6) {
  let beta = [0, 0, 0]
  let H: number[][] = []
  for (let it = 0; it < 25; it++) {
    H = [
      [ridge, 0, 0],
      [0, ridge, 0],
      [0, 0, ridge],
    ]
    const g = beta.map((b) => -ridge * b)
    for (let i = 0; i < D.length; i++) {
      if (w[i] < 1e-12) continue
      const p = sigmoid(D[i][0] * beta[0] + D[i][1] * beta[1] + D[i][2] * beta[2])
      const wv = w[i] * p * (1 - p)
      for (let j = 0; j < 3; j++) {
        g[j] += w[i] * (y[i] - p) * D[i][j]
        for (let k = 0; k < 3; k++) H[j][k] += wv * D[i][j] * D[i][k]
      }
    }
    const Hi = inv3(H)
    const step = Hi.map((row) => row.reduce((s, v, k) => s + v * g[k], 0))
    beta = beta.map((b, j) => b + Math.max(-5, Math.min(5, step[j])))
    if (Math.max(...step.map(Math.abs)) < 1e-8) break
  }
  const scores = D.map((d, i) => {
    const p = sigmoid(d[0] * beta[0] + d[1] * beta[1] + d[2] * beta[2])
    return d.map((dj) => w[i] * (y[i] - p) * dj)
  })
  return { beta, Hinv: inv3(H), scores }
}

/** Global logistic regression on x → (1, x₁, x₂): the MLE and (XᵀDX)⁻¹. */
export function fitLogistic(X: Pt[], y: number[]) {
  const D = X.map(([a, b]) => [1, a, b])
  const { beta, Hinv } = logisticNewton(D, y, Array(X.length).fill(1))
  return { theta: beta, Hinv }
}

export type TestResult = { etaBar: number; se: number; z: number; p: number }

const twoSided = (z: number) => 2 * (1 - normalCdf(Math.abs(z)))

/**
 * Poyiadzi et al. (2022): mean fitted posterior over the anchors, with null variance (1/16) x̄ᵀ Ĥ x̄, where x̄ is the
 * mean augmented anchor and Ĥ = (XᵀDX)⁻¹.
 */
export function parametricTest(fit: ReturnType<typeof fitLogistic>, anchors: Pt[]): TestResult {
  const A = anchors.map(([a, b]) => [1, a, b])
  const etaBar =
    A.reduce((s, x) => s + sigmoid(x[0] * fit.theta[0] + x[1] * fit.theta[1] + x[2] * fit.theta[2]), 0) / A.length
  const xbar = [0, 1, 2].map((j) => A.reduce((s, x) => s + x[j], 0) / A.length)
  const se = Math.sqrt(quad(xbar, fit.Hinv, xbar) / 16)
  const z = (etaBar - 0.5) / se
  return { etaBar, se, z, p: twoSided(z) }
}

/**
 * Yang et al. (2024): a local linear logistic fit at each anchor with a Gaussian kernel of bandwidth h. The anchor's
 * estimate is the fitted intercept; the variance of the mean over anchors uses the sandwich B⁻¹ C B⁻¹ and its
 * cross-anchor analogue, scaled by 1/16 under the null.
 */
export function localTest(X: Pt[], y: number[], anchors: Pt[], h: number): TestResult {
  const fits = anchors.map(([a0, b0]) => {
    const D = X.map(([a, b]) => [1, a - a0, b - b0])
    const w = X.map(([a, b]) => Math.exp((-0.5 * ((a - a0) ** 2 + (b - b0) ** 2)) / (h * h)))
    const f = logisticNewton(D, y, w, 1e-3)
    return { eta: sigmoid(f.beta[0]), row0: f.Hinv[0], scores: f.scores }
  })
  const k = fits.length
  let v = 0
  for (let j = 0; j < k; j++) {
    for (let l = 0; l < k; l++) {
      // [B_j⁻¹ (Σ_i s_ij s_ilᵀ) B_l⁻¹]₀₀ = Σ_i (row0_j · s_ij)(row0_l · s_il)
      const sj = fits[j].scores
      const sl = fits[l].scores
      const rj = fits[j].row0
      const rl = fits[l].row0
      for (let i = 0; i < sj.length; i++) {
        const a = rj[0] * sj[i][0] + rj[1] * sj[i][1] + rj[2] * sj[i][2]
        if (a === 0) continue
        v += a * (rl[0] * sl[i][0] + rl[1] * sl[i][1] + rl[2] * sl[i][2])
      }
    }
  }
  const etaBar = fits.reduce((s, f) => s + f.eta, 0) / k
  const se = Math.sqrt(v / (16 * k * k))
  const z = (etaBar - 0.5) / se
  return { etaBar, se, z, p: twoSided(z) }
}

/** k anchors on the line x₂ = −x₁ with x₁ uniform on [−4, 4], where the two-Gaussian posterior is exactly 1/2. */
export function lineAnchors(k: number, seed: number): Pt[] {
  const r = rng(seed)
  return Array.from({ length: k }, () => {
    const t = -4 + 8 * r.uniform()
    return [t, -t] as Pt
  })
}

/**
 * Anchors with xorPosterior = 1/2 inside [−4, 4]²: from random points and directions, find a sign change of
 * η − 1/2 along the line and bisect it.
 */
export function xorAnchors(k: number, seed: number): Pt[] {
  const r = rng(seed)
  const out: Pt[] = []
  const f = (p: Pt) => xorPosterior(p) - 0.5
  const inBox = (p: Pt) => Math.abs(p[0]) <= 4 && Math.abs(p[1]) <= 4
  let guard = 0
  while (out.length < k && guard++ < 1000) {
    const p: Pt = [-4 + 8 * r.uniform(), -4 + 8 * r.uniform()]
    const ang = 2 * Math.PI * r.uniform()
    const d: Pt = [Math.cos(ang), Math.sin(ang)]
    const at = (t: number): Pt => [p[0] + t * d[0], p[1] + t * d[1]]
    const crossings: number[] = []
    for (let t = -6; t < 6; t += 0.1) {
      if (inBox(at(t)) && inBox(at(t + 0.1)) && f(at(t)) * f(at(t + 0.1)) < 0) crossings.push(t)
    }
    if (!crossings.length) continue
    let lo = crossings[Math.floor(r.uniform() * crossings.length)]
    let hi = lo + 0.1
    for (let it = 0; it < 40; it++) {
      const m = (lo + hi) / 2
      if (f(at(lo)) * f(at(m)) <= 0) hi = m
      else lo = m
    }
    out.push(at(lo))
  }
  return out
}

/** Points on the true η = 1/2 contour of the XOR mixture, for drawing: a scan along rows and columns of a grid. */
export function xorBoundary(): Pt[] {
  const pts: Pt[] = []
  const g = (t: number) => -5 + 0.05 * t
  for (let i = 0; i <= 200; i++) {
    for (let j = 0; j < 200; j++) {
      const a: Pt = [g(i), g(j)]
      const b: Pt = [g(i), g(j + 1)]
      const fa = xorPosterior(a) - 0.5
      const fb = xorPosterior(b) - 0.5
      if (fa * fb < 0) pts.push([a[0], a[1] + (0.05 * fa) / (fa - fb)])
      const c: Pt = [g(j), g(i)]
      const e: Pt = [g(j + 1), g(i)]
      const fc = xorPosterior(c) - 0.5
      const fe = xorPosterior(e) - 0.5
      if (fc * fe < 0) pts.push([c[0] + (0.05 * fc) / (fc - fe), c[1]])
    }
  }
  return pts
}

/** The x₂ values of the line θ₀ + θ₁x₁ + θ₂x₂ = c (c = logit of the threshold) at the given x₁. */
export const boundaryLine = (theta: number[], xs: number[], c = 0) =>
  xs.map((x) => (c - theta[0] - theta[1] * x) / theta[2])
