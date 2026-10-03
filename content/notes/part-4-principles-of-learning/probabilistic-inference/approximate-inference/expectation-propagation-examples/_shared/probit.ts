/**
 * Bayesian probit regression in two dimensions: w ~ N(0, s0 I), p(y_i | w) = Φ(y_i x_iᵀw). EP with rank-one sites on
 * the projections a_i = x_iᵀw, the Laplace approximation, and the exact posterior on a grid.
 */
import { logNormalCdf, probitTilted, vFn, wFn } from './ep.ts'

export type V2 = [number, number]
/** Symmetric 2×2 matrix [a, b; b, c] stored as [a, b, c]. */
export type S2 = [number, number, number]

const inv = ([a, b, c]: S2): S2 => {
  const d = a * c - b * b
  return [c / d, -b / d, a / d]
}
const mul = ([a, b, c]: S2, [x, y]: V2): V2 => [a * x + b * y, b * x + c * y]
const quad = (s: S2, x: V2) => {
  const [p, q] = mul(s, x)
  return p * x[0] + q * x[1]
}
const dot = (a: V2, b: V2) => a[0] * b[0] + a[1] * b[1]

export type ProbitStep = { sweep: number; site: number; ok: boolean; mean: V2; cov: S2 }

/** Posterior mean and covariance from the prior precision and the rank-one sites exp(−½τ a² + ν a), a = xᵀw. */
function posterior(X: V2[], tau: number[], nu: number[], priorVar: number) {
  const P: S2 = [1 / priorVar, 0, 1 / priorVar]
  const h: V2 = [0, 0]
  X.forEach((x, i) => {
    P[0] += tau[i] * x[0] * x[0]
    P[1] += tau[i] * x[0] * x[1]
    P[2] += tau[i] * x[1] * x[1]
    h[0] += nu[i] * x[0]
    h[1] += nu[i] * x[1]
  })
  const cov = inv(P)
  return { mean: mul(cov, h), cov }
}

/**
 * EP for probit regression. Site i touches w only through a_i = x_iᵀw, so the cavity, the tilted moments and the site
 * are all one-dimensional: remove site i from the marginal of a_i, match the moments of that marginal times Φ(y_i a_i),
 * and store the difference in natural parameters.
 */
export function epProbit(X: V2[], y: number[], priorVar: number, sweeps: number, damping = 1): ProbitStep[] {
  const n = X.length
  const tau = new Array<number>(n).fill(0)
  const nu = new Array<number>(n).fill(0)
  const steps: ProbitStep[] = []
  let q = posterior(X, tau, nu, priorVar)
  for (let sweep = 0; sweep < sweeps; sweep++) {
    for (let i = 0; i < n; i++) {
      const s2 = quad(q.cov, X[i])
      const m = dot(q.mean, X[i])
      const cavTau = 1 / s2 - tau[i]
      const cavNu = m / s2 - nu[i]
      if (!(cavTau > 0)) {
        steps.push({ sweep, site: i, ok: false, ...q })
        continue
      }
      const t = probitTilted(cavNu / cavTau, 1 / cavTau, y[i], 0, 1)
      tau[i] = damping * (1 / t.variance - cavTau) + (1 - damping) * tau[i]
      nu[i] = damping * (t.mean / t.variance - cavNu) + (1 - damping) * nu[i]
      q = posterior(X, tau, nu, priorVar)
      steps.push({ sweep, site: i, ok: true, ...q })
    }
  }
  return steps
}

/** Laplace: Newton's method on the log posterior, which is concave, then the inverse negative Hessian at the mode. */
export function laplaceProbit(X: V2[], y: number[], priorVar: number): { mean: V2; cov: S2 } {
  let w: V2 = [0, 0]
  let H: S2 = [1 / priorVar, 0, 1 / priorVar]
  for (let it = 0; it < 100; it++) {
    const g: V2 = [-w[0] / priorVar, -w[1] / priorVar]
    H = [1 / priorVar, 0, 1 / priorVar]
    X.forEach((x, i) => {
      const a = y[i] * dot(x, w)
      const v = vFn(a)
      const c = wFn(a)
      g[0] += y[i] * v * x[0]
      g[1] += y[i] * v * x[1]
      H[0] += c * x[0] * x[0]
      H[1] += c * x[0] * x[1]
      H[2] += c * x[1] * x[1]
    })
    const step = mul(inv(H), g)
    w = [w[0] + step[0], w[1] + step[1]]
    if (Math.hypot(step[0], step[1]) < 1e-10) break
  }
  return { mean: w, cov: inv(H) }
}

/** Unnormalised exact log posterior on a grid; rows are w2, columns w1. */
export function exactProbitGrid(X: V2[], y: number[], priorVar: number, w1: number[], w2: number[]) {
  return w2.map((b) =>
    w1.map(
      (a) =>
        -(a * a + b * b) / (2 * priorVar) + X.reduce((s, x, i) => s + logNormalCdf(y[i] * (x[0] * a + x[1] * b)), 0),
    ),
  )
}

/** Predictive probability of y = +1 at x under a Gaussian posterior: Φ(mᵀx / √(1 + xᵀSx)). */
export const predictive = (mean: V2, cov: S2, x: V2) =>
  Math.exp(logNormalCdf(dot(mean, x) / Math.sqrt(1 + quad(cov, x))))

/** Points on the k-standard-deviation ellipse of a 2-D Gaussian. */
export function ellipse(mean: V2, [a, b, c]: S2, k = 2, n = 90): { x: number[]; y: number[] } {
  // Cholesky factor L of the covariance maps the unit circle onto the 1-sd ellipse.
  const l11 = Math.sqrt(a)
  const l21 = b / l11
  const l22 = Math.sqrt(Math.max(c - l21 * l21, 0))
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i <= n; i++) {
    const t = (2 * Math.PI * i) / n
    const u = Math.cos(t)
    const v = Math.sin(t)
    xs.push(mean[0] + k * l11 * u)
    ys.push(mean[1] + k * (l21 * u + l22 * v))
  }
  return { x: xs, y: ys }
}
