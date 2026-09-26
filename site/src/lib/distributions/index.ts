/**
 * Numerical definitions of probability distributions, shared by every `distribution` note and by widgets that need a
 * pmf, pdf, cdf or quantile. Formulas and prose live in the notes; this file only computes.
 *
 * To add a distribution: add an entry here, then write its note from docs/templates/distribution.mdx and place
 * `<DistributionExplorer id="…" />` in it.
 */
import { logChoose, logFactorial, normalCdf, normalPdf, normalQuantile } from '@/lib/math/special'
import { bernoulli, geometric, hypergeometric, negativeBinomial } from './discrete'
import { continuous } from './continuous'

export type DistributionParam = {
  key: string
  /** Label for the slider, e.g. 'rate λ'. */
  label: string
  min: number
  max: number
  step: number
  value: number
}

export type Params = Record<string, number>

export type Distribution = {
  id: string
  name: string
  discrete: boolean
  params: DistributionParam[]
  /** pmf for discrete distributions, pdf for continuous ones. */
  density: (x: number, p: Params) => number
  cdf: (x: number, p: Params) => number
  /** Inverse cdf, where it has a closed form. Used by inverse-transform sampling. */
  quantile?: (u: number, p: Params) => number
  mean: (p: Params) => number
  variance: (p: Params) => number
  /** Range to plot for these parameters. */
  range: (p: Params) => [number, number]
}

const poisson: Distribution = {
  id: 'poisson',
  name: 'Poisson',
  discrete: true,
  params: [{ key: 'lambda', label: 'rate λ', min: 0.1, max: 30, step: 0.1, value: 4 }],
  density: (k, { lambda }) =>
    k < 0 || !Number.isInteger(k) ? 0 : Math.exp(k * Math.log(lambda) - lambda - logFactorial(k)),
  cdf: (x, p) => {
    let total = 0
    for (let k = 0; k <= Math.floor(x); k++) total += poisson.density(k, p)
    return Math.min(total, 1)
  },
  mean: ({ lambda }) => lambda,
  variance: ({ lambda }) => lambda,
  range: ({ lambda }) => [0, Math.max(10, Math.ceil(lambda + 5 * Math.sqrt(lambda)))],
}

const binomial: Distribution = {
  id: 'binomial',
  name: 'Binomial',
  discrete: true,
  params: [
    { key: 'n', label: 'trials n', min: 1, max: 200, step: 1, value: 20 },
    { key: 'p', label: 'success probability p', min: 0.01, max: 0.99, step: 0.01, value: 0.3 },
  ],
  density: (k, { n, p }) => {
    if (k < 0 || k > n || !Number.isInteger(k)) return 0
    if (p === 0) return k === 0 ? 1 : 0
    if (p === 1) return k === n ? 1 : 0
    return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))
  },
  cdf: (x, params) => {
    let total = 0
    for (let k = 0; k <= Math.min(Math.floor(x), params.n); k++) total += binomial.density(k, params)
    return Math.min(total, 1)
  },
  mean: ({ n, p }) => n * p,
  variance: ({ n, p }) => n * p * (1 - p),
  range: ({ n }) => [0, n],
}

const exponential: Distribution = {
  id: 'exponential',
  name: 'Exponential',
  discrete: false,
  params: [{ key: 'rate', label: 'rate λ', min: 0.1, max: 5, step: 0.1, value: 1 }],
  density: (x, { rate }) => (x < 0 ? 0 : rate * Math.exp(-rate * x)),
  cdf: (x, { rate }) => (x < 0 ? 0 : 1 - Math.exp(-rate * x)),
  quantile: (u, { rate }) => -Math.log(1 - u) / rate,
  mean: ({ rate }) => 1 / rate,
  variance: ({ rate }) => 1 / (rate * rate),
  range: ({ rate }) => [0, 6 / rate],
}

const gaussian: Distribution = {
  id: 'gaussian',
  name: 'Gaussian',
  discrete: false,
  params: [
    { key: 'mu', label: 'mean μ', min: -5, max: 5, step: 0.1, value: 0 },
    { key: 'sigma', label: 'standard deviation σ', min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  density: (x, { mu, sigma }) => normalPdf((x - mu) / sigma) / sigma,
  cdf: (x, { mu, sigma }) => normalCdf((x - mu) / sigma),
  quantile: (u, { mu, sigma }) => mu + sigma * normalQuantile(u),
  mean: ({ mu }) => mu,
  variance: ({ sigma }) => sigma * sigma,
  range: ({ mu, sigma }) => [mu - 4 * sigma, mu + 4 * sigma],
}

const cauchy: Distribution = {
  id: 'cauchy',
  name: 'Cauchy',
  discrete: false,
  params: [
    { key: 'loc', label: 'location x₀', min: -5, max: 5, step: 0.1, value: 0 },
    { key: 'scale', label: 'scale γ', min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  density: (x, { loc, scale }) => 1 / (Math.PI * scale * (1 + ((x - loc) / scale) ** 2)),
  cdf: (x, { loc, scale }) => 0.5 + Math.atan((x - loc) / scale) / Math.PI,
  quantile: (u, { loc, scale }) => loc + scale * Math.tan(Math.PI * (u - 0.5)),
  mean: () => NaN,
  variance: () => NaN,
  range: ({ loc, scale }) => [loc - 10 * scale, loc + 10 * scale],
}

const gumbel: Distribution = {
  id: 'gumbel',
  name: 'Gumbel',
  discrete: false,
  params: [
    { key: 'loc', label: 'location μ', min: -5, max: 5, step: 0.1, value: 0 },
    { key: 'scale', label: 'scale β', min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  density: (x, { loc, scale }) => {
    const z = (x - loc) / scale
    return Math.exp(-(z + Math.exp(-z))) / scale
  },
  cdf: (x, { loc, scale }) => Math.exp(-Math.exp(-(x - loc) / scale)),
  quantile: (u, { loc, scale }) => loc - scale * Math.log(-Math.log(u)),
  mean: ({ loc, scale }) => loc + scale * 0.5772156649,
  variance: ({ scale }) => (Math.PI ** 2 / 6) * scale * scale,
  range: ({ loc, scale }) => [loc - 3 * scale, loc + 7 * scale],
}

const logistic: Distribution = {
  id: 'logistic',
  name: 'Logistic',
  discrete: false,
  params: [
    { key: 'loc', label: 'location μ', min: -5, max: 5, step: 0.1, value: 0 },
    { key: 'scale', label: 'scale s', min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  density: (x, { loc, scale }) => {
    const e = Math.exp(-(x - loc) / scale)
    return e / (scale * (1 + e) ** 2)
  },
  cdf: (x, { loc, scale }) => 1 / (1 + Math.exp(-(x - loc) / scale)),
  quantile: (u, { loc, scale }) => loc + scale * Math.log(u / (1 - u)),
  mean: ({ loc }) => loc,
  variance: ({ scale }) => (Math.PI * scale) ** 2 / 3,
  range: ({ loc, scale }) => [loc - 8 * scale, loc + 8 * scale],
}

export const distributions: Record<string, Distribution> = Object.fromEntries(
  [
    poisson,
    binomial,
    exponential,
    gaussian,
    cauchy,
    gumbel,
    logistic,
    bernoulli,
    geometric,
    negativeBinomial,
    hypergeometric,
    ...continuous,
  ].map((d) => [d.id, d]),
)

export function distribution(id: string): Distribution {
  const d = distributions[id]
  if (!d) throw new Error(`unknown distribution "${id}" (see site/src/lib/distributions)`)
  return d
}

export const defaults = (d: Distribution): Params => Object.fromEntries(d.params.map((p) => [p.key, p.value]))
