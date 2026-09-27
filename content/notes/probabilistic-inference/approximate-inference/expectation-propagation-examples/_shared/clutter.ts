/**
 * Minka's clutter problem in one dimension: θ ~ N(0, 100), and each x_i is drawn from N(θ, 1) with probability 1 − w or
 * from the clutter N(0, 10) with probability w. Exact posterior on a grid, EP, Laplace and mean-field VB.
 */
import { clutterTilted, ep1d, grid, normalLogPdf, normalPdf, normaliseOnGrid, toNat, type Moments } from './ep.ts'

export const PRIOR_VAR = 100
export const CLUTTER_VAR = 10

const logLik = (x: number, theta: number, w: number) =>
  Math.log((1 - w) * normalPdf(x, theta, 1) + w * normalPdf(x, 0, CLUTTER_VAR))

/** Posterior probability that each point is signal, given θ. */
const signalProb = (x: number, theta: number, w: number) => {
  const s = (1 - w) * normalPdf(x, theta, 1)
  return s / (s + w * normalPdf(x, 0, CLUTTER_VAR))
}

/** The exact posterior on a wide grid; the tails follow the broad prior, so the grid must reach far. */
export function clutterExact(xs: number[], w: number, thetas = grid(-40, 40, 2001)) {
  const logf = thetas.map((t) => normalLogPdf(t, 0, PRIOR_VAR) + xs.reduce((s, x) => s + logLik(x, t, w), 0))
  return { thetas, ...normaliseOnGrid(thetas, logf) }
}

export function clutterEp(xs: number[], w: number, sweeps: number, damping = 1) {
  return ep1d(
    toNat({ mean: 0, variance: PRIOR_VAR }),
    xs.length,
    (i, c) => clutterTilted(xs[i], w, CLUTTER_VAR, c.mean, c.variance),
    {
      sweeps,
      damping,
    },
  )
}

/** Laplace: Newton's method on the log posterior from the best grid point; variance from the curvature there. */
export function clutterLaplace(xs: number[], w: number): Moments {
  const coarse = grid(-20, 20, 801)
  let theta = coarse[0]
  let best = -Infinity
  for (const t of coarse) {
    const lp = normalLogPdf(t, 0, PRIOR_VAR) + xs.reduce((s, x) => s + logLik(x, t, w), 0)
    if (lp > best) [best, theta] = [lp, t]
  }
  let hess = -1 / PRIOR_VAR
  for (let it = 0; it < 50; it++) {
    const r = xs.map((x) => signalProb(x, theta, w))
    const grad = -theta / PRIOR_VAR + xs.reduce((s, x, i) => s + r[i] * (x - theta), 0)
    hess = -1 / PRIOR_VAR + xs.reduce((s, x, i) => s - r[i] + r[i] * (1 - r[i]) * (x - theta) ** 2, 0)
    if (hess >= 0) break
    const next = theta - grad / hess
    if (Math.abs(next - theta) < 1e-10) break
    theta = next
  }
  return { mean: theta, variance: hess < 0 ? -1 / hess : NaN }
}

/**
 * Mean-field variational Bayes: q(θ) q(c_1) … q(c_n), with c_i the indicator that x_i is signal. Coordinate ascent from
 * the sample mean with unit variance.
 */
export function clutterVb(xs: number[], w: number): Moments {
  let mean = xs.reduce((a, b) => a + b, 0) / xs.length
  let variance = 1
  for (let it = 0; it < 500; it++) {
    const q = xs.map((x) => {
      const l1 = Math.log(1 - w) + normalLogPdf(x, mean, 1) - 0.5 * variance
      const l2 = Math.log(w) + normalLogPdf(x, 0, CLUTTER_VAR)
      return 1 / (1 + Math.exp(l2 - l1))
    })
    const k = q.reduce((a, b) => a + b, 0)
    variance = 1 / (k + 1 / PRIOR_VAR)
    const next = variance * xs.reduce((s, x, i) => s + q[i] * x, 0)
    const done = Math.abs(next - mean) < 1e-12
    mean = next
    if (done) break
  }
  return { mean, variance }
}

/** The exact factor t_i(θ) and the site that replaces it, Z q_new / cavity, on a grid of θ. */
export function clutterFactorAndSite(
  x: number,
  w: number,
  thetas: number[],
  cavity: Moments,
  next: Moments,
  logZ: number,
) {
  return {
    factor: thetas.map((t) => Math.exp(logLik(x, t, w))),
    site: thetas.map((t) =>
      Math.exp(logZ + normalLogPdf(t, next.mean, next.variance) - normalLogPdf(t, cavity.mean, cavity.variance)),
    ),
  }
}
