/**
 * Expectation propagation primitives shared by the EP worked examples: normal-distribution functions that stay accurate
 * in the tails, Gaussian sites in natural parameters, closed-form tilted moments for step, probit and clutter factors,
 * and a generic one-dimensional EP loop with damping that records every site update.
 *
 * Plain TypeScript with no imports, so the numbers can be checked against a Python reference outside the site.
 */

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

export const normalPdf = (x: number, mean = 0, variance = 1) =>
  Math.exp((-0.5 * (x - mean) ** 2) / variance - 0.5 * Math.log(variance) - LOG_SQRT_2PI)

export const normalLogPdf = (x: number, mean = 0, variance = 1) =>
  (-0.5 * (x - mean) ** 2) / variance - 0.5 * Math.log(variance) - LOG_SQRT_2PI

/**
 * log erfc(z), from the Chebyshev fit in Numerical Recipes (relative error below 1.2e-7 everywhere). Kept in log form
 * so that log Φ(t) and φ(t)/Φ(t) stay finite far into the lower tail, where Φ(t) underflows.
 */
function logErfc(z: number): number {
  const a = Math.abs(z)
  const t = 1 / (1 + 0.5 * a)
  const poly =
    -a * a -
    1.26551223 +
    t *
      (1.00002368 +
        t *
          (0.37409196 +
            t *
              (0.09678418 +
                t *
                  (-0.18628806 +
                    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
  const logTail = Math.log(t) + poly
  return z >= 0 ? logTail : Math.log(2 - Math.exp(logTail))
}

/** log Φ(t), the log of the standard normal cdf. */
export const logNormalCdf = (t: number) => Math.log(0.5) + logErfc(-t / Math.SQRT2)
export const normalCdf = (t: number) => Math.exp(logNormalCdf(t))

/** v(t) = φ(t)/Φ(t): the mean shift of a standard normal truncated to (−t, ∞), in units of its sd. */
export const vFn = (t: number) => Math.exp(normalLogPdf(t) - logNormalCdf(t))
/** w(t) = v(t)(v(t) + t): the fraction of variance removed by the same truncation. Lies in (0, 1). */
export const wFn = (t: number) => {
  const v = vFn(t)
  return v * (v + t)
}

/** An unnormalised Gaussian in natural parameters: precision τ and precision-weighted mean ν = τμ. */
export type Nat = { tau: number; nu: number }
export type Moments = { mean: number; variance: number }

export const toMoments = ({ tau, nu }: Nat): Moments => ({ mean: nu / tau, variance: 1 / tau })
export const toNat = ({ mean, variance }: Moments): Nat => ({ tau: 1 / variance, nu: mean / variance })
export const times = (a: Nat, b: Nat): Nat => ({ tau: a.tau + b.tau, nu: a.nu + b.nu })
export const divide = (a: Nat, b: Nat): Nat => ({ tau: a.tau - b.tau, nu: a.nu - b.nu })
/** Damped site update: a convex combination of natural parameters, i.e. a weighted geometric mean of the sites. */
export const damp = (next: Nat, old: Nat, step: number): Nat => ({
  tau: step * next.tau + (1 - step) * old.tau,
  nu: step * next.nu + (1 - step) * old.nu,
})
export const ONE: Nat = { tau: 0, nu: 0 }

/** Result of multiplying a Gaussian cavity by one factor: the tilted distribution's log normaliser and moments. */
export type Tilted = { logZ: number; mean: number; variance: number }

/** N(x | μ, σ²) · 𝟙(x > ε): the truncated Gaussian, with t = (μ − ε)/σ. */
export function stepTilted(mean: number, variance: number, eps = 0): Tilted {
  const s = Math.sqrt(variance)
  const t = (mean - eps) / s
  return { logZ: logNormalCdf(t), mean: mean + s * vFn(t), variance: variance * (1 - wFn(t)) }
}

/**
 * N(θ | μ, σ²) · Φ(y(θ − c)/s): a probit factor with sign y = ±1, offset c and noise variance s². This is the step
 * factor applied to θ plus Gaussian noise, so it has the step's moments with σ² + s² in place of σ².
 */
export function probitTilted(mean: number, variance: number, y = 1, c = 0, s2 = 1): Tilted {
  const d = Math.sqrt(s2 + variance)
  const z = (y * (mean - c)) / d
  const v = vFn(z)
  return {
    logZ: logNormalCdf(z),
    mean: mean + (y * variance * v) / d,
    variance: variance - (variance * variance * v * (v + z)) / (s2 + variance),
  }
}

/** Minka's clutter factor (1 − w) N(x | θ, 1) + w N(x | 0, clutterVar), times a Gaussian cavity on θ. */
export function clutterTilted(x: number, w: number, clutterVar: number, mean: number, variance: number) {
  const zSignal = (1 - w) * normalPdf(x, mean, variance + 1)
  const zClutter = w * normalPdf(x, 0, clutterVar)
  const Z = zSignal + zClutter
  // r: the posterior probability, under the cavity, that x is signal rather than clutter.
  const r = zSignal / Z
  const d = x - mean
  return {
    logZ: Math.log(Z),
    r,
    mean: mean + (variance * r * d) / (variance + 1),
    variance:
      variance -
      (r * variance * variance) / (variance + 1) +
      (r * (1 - r) * variance ** 2 * d * d) / (variance + 1) ** 2,
  }
}

/** One recorded site update of a one-dimensional EP run. */
export type Ep1dStep = {
  sweep: number
  site: number
  /** False when the cavity had non-positive precision and the update was skipped. */
  ok: boolean
  cavity: Moments
  tilted: Tilted
  /** Sites and posterior after this update. */
  sites: Nat[]
  q: Nat
}

/**
 * One-dimensional EP for a prior times n factors. `tilted(i, cavity)` returns the moments of cavity · t_i. Records
 * every update, so a figure can step through them. The sites start at 1 (τ = ν = 0), so the first sweep is ADF.
 */
export function ep1d(
  prior: Nat,
  n: number,
  tilted: (i: number, cavity: Moments) => Tilted,
  { sweeps = 10, damping = 1, order }: { sweeps?: number; damping?: number; order?: number[] } = {},
): Ep1dStep[] {
  let sites: Nat[] = Array.from({ length: n }, () => ONE)
  const visit = order ?? Array.from({ length: n }, (_, i) => i)
  const total = () => sites.reduce(times, prior)
  const steps: Ep1dStep[] = []
  for (let sweep = 0; sweep < sweeps; sweep++) {
    for (const i of visit) {
      const cav = divide(total(), sites[i])
      if (!(cav.tau > 0)) {
        steps.push({
          sweep,
          site: i,
          ok: false,
          cavity: { mean: NaN, variance: NaN },
          tilted: { logZ: NaN, mean: NaN, variance: NaN },
          sites,
          q: total(),
        })
        continue
      }
      const cavity = toMoments(cav)
      const t = tilted(i, cavity)
      const next = divide(toNat(t), cav)
      sites = sites.map((s, j) => (j === i ? damp(next, s, damping) : s))
      steps.push({ sweep, site: i, ok: true, cavity, tilted: t, sites, q: total() })
    }
  }
  return steps
}

/** Assumed density filtering: one pass, each factor absorbed once into the running Gaussian. */
export function adf(prior: Moments, order: number[], tilted: (i: number, q: Moments) => Tilted): Moments[] {
  const out: Moments[] = [prior]
  let q = prior
  for (const i of order) {
    const t = tilted(i, q)
    q = { mean: t.mean, variance: t.variance }
    out.push(q)
  }
  return out
}

/** Evenly spaced points, inclusive. */
export function grid(lo: number, hi: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

/** Normalise log-density values on an even grid; returns the density and its mean and variance (trapezoid rule). */
export function normaliseOnGrid(xs: number[], logf: number[]): { density: number[] } & Moments {
  const top = Math.max(...logf)
  const f = logf.map((v) => Math.exp(v - top))
  const h = xs[1] - xs[0]
  const trap = (g: number[]) => h * (g.reduce((a, b) => a + b, 0) - 0.5 * (g[0] + g[g.length - 1]))
  const Z = trap(f)
  const density = f.map((v) => v / Z)
  const mean = trap(density.map((p, i) => p * xs[i]))
  const variance = trap(density.map((p, i) => p * (xs[i] - mean) ** 2))
  return { density, mean, variance }
}

/**
 * Moments of cavity · exp(power · logFactor), by the trapezoid rule on ±10 cavity standard deviations. Used when the
 * tilted moments have no closed form, e.g. for a factor raised to a power.
 */
export function tiltedByQuadrature(
  cavity: Moments,
  logFactor: (theta: number) => number,
  power = 1,
  points = 801,
): Tilted {
  const s = Math.sqrt(cavity.variance)
  const thetas = grid(cavity.mean - 10 * s, cavity.mean + 10 * s, points)
  const logf = thetas.map((t) => normalLogPdf(t, cavity.mean, cavity.variance) + power * logFactor(t))
  const top = Math.max(...logf)
  const h = thetas[1] - thetas[0]
  let z = 0
  let m1 = 0
  let m2 = 0
  logf.forEach((l, i) => {
    const f = Math.exp(l - top)
    z += f
    m1 += f * thetas[i]
    m2 += f * thetas[i] * thetas[i]
  })
  const mean = m1 / z
  return { logZ: top + Math.log(z * h), mean, variance: m2 / z - mean * mean }
}

/**
 * Power EP (Minka 2004) in one dimension: each update removes only a fraction `power` of site i, multiplies in the
 * factor raised to that power, projects, and rescales the change by 1/power. power = 1 is ordinary EP. Its fixed points
 * minimise local α-divergences with α = power.
 */
export function powerEp1d(
  prior: Nat,
  n: number,
  logFactor: (i: number, theta: number) => number,
  { sweeps = 10, damping = 1, power = 1 }: { sweeps?: number; damping?: number; power?: number } = {},
): Ep1dStep[] {
  let sites: Nat[] = Array.from({ length: n }, () => ONE)
  const total = () => sites.reduce(times, prior)
  const steps: Ep1dStep[] = []
  for (let sweep = 0; sweep < sweeps; sweep++) {
    for (let i = 0; i < n; i++) {
      const q = total()
      const cav = { tau: q.tau - power * sites[i].tau, nu: q.nu - power * sites[i].nu }
      if (!(cav.tau > 0)) {
        steps.push({
          sweep,
          site: i,
          ok: false,
          cavity: { mean: NaN, variance: NaN },
          tilted: { logZ: NaN, mean: NaN, variance: NaN },
          sites,
          q,
        })
        continue
      }
      const cavity = toMoments(cav)
      const t = tiltedByQuadrature(cavity, (theta) => logFactor(i, theta), power)
      const proj = toNat(t)
      const next = { tau: (proj.tau - cav.tau) / power, nu: (proj.nu - cav.nu) / power }
      sites = sites.map((s, j) => (j === i ? damp(next, s, damping) : s))
      steps.push({ sweep, site: i, ok: true, cavity, tilted: t, sites, q: total() })
    }
  }
  return steps
}
