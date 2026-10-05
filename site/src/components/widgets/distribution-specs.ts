/**
 * The distributions a `distribution` note can show in `DistributionExplorer`: for each id, the aifn family built from
 * the note's parameterisation, the sliders (label, range, default) and the x range to plot. The numbers (pmf, pdf, cdf,
 * quantiles, moments) all come from `aifn/probability/distributions`; this file holds presentation only.
 *
 * To add a distribution: add an entry here, then write its note from docs/templates/distribution.mdx and place
 * `<DistributionExplorer id="…" />` in it.
 */
import {
  Bernoulli,
  Beta,
  Binomial,
  Cauchy,
  ChiSquare,
  Exponential,
  FisherSnedecor,
  Gamma,
  Geometric,
  Gumbel,
  Hypergeometric,
  InverseGamma,
  Laplace,
  Logistic,
  LogNormal,
  NegativeBinomial,
  Normal,
  Poisson,
  StudentT,
  type Univariate,
} from 'aifn-compute/probability/distributions'

export type ParamSpec = { key: string; label: string; min: number; max: number; step: number; value: number }
export type Params = Record<string, number>

export type DistributionSpec = {
  name: string
  params: ParamSpec[]
  make: (p: Params) => Univariate<number>
  /** The x range to plot for these parameters (integers for a discrete distribution). */
  range: (p: Params, d: Univariate<number>) => [number, number]
}

const param = (key: string, label: string, min: number, max: number, step: number, value: number): ParamSpec => ({
  key,
  label,
  min,
  max,
  step,
  value,
})
const loc = (label = 'location μ') => param('loc', label, -5, 5, 0.1, 0)
const scale = (label: string) => param('scale', label, 0.1, 5, 0.1, 1)

/** Up to the `u`-quantile, starting just above 0 where the density may be infinite. */
const positive =
  (u: number) =>
  (_: Params, d: Univariate<number>): [number, number] => {
    const hi = d.quantile(u)
    return [hi * 1e-3, hi]
  }

/** Sliders move independently, so K and n are clamped to the population size N. */
const population = ({ N, K, n }: Params) => ({ N, K: Math.min(K, N), n: Math.min(n, N) })

const specs: Record<string, DistributionSpec> = {
  poisson: {
    name: 'Poisson',
    params: [param('lambda', 'rate λ', 0.1, 30, 0.1, 4)],
    make: ({ lambda }) => Poisson(lambda),
    range: ({ lambda }) => [0, Math.max(10, Math.ceil(lambda + 5 * Math.sqrt(lambda)))],
  },
  binomial: {
    name: 'Binomial',
    params: [param('n', 'trials n', 1, 200, 1, 20), param('p', 'success probability p', 0.01, 0.99, 0.01, 0.3)],
    make: ({ n, p }) => Binomial(n, p),
    range: ({ n }) => [0, n],
  },
  bernoulli: {
    name: 'Bernoulli',
    params: [param('p', 'success probability p', 0, 1, 0.01, 0.3)],
    make: ({ p }) => Bernoulli(p),
    range: () => [0, 1],
  },
  geometric: {
    name: 'Geometric',
    params: [param('p', 'success probability p', 0.02, 1, 0.01, 0.25)],
    make: ({ p }) => Geometric(p),
    range: ({ p }) => [1, p === 1 ? 5 : Math.max(10, Math.ceil(Math.log(0.001) / Math.log(1 - p)))],
  },
  'negative-binomial': {
    name: 'Negative binomial',
    params: [param('r', 'successes r', 0.5, 30, 0.5, 3), param('p', 'success probability p', 0.02, 0.99, 0.01, 0.4)],
    make: ({ r, p }) => NegativeBinomial(r, p),
    range: (_, d) => [0, Math.max(10, Math.ceil(d.mean() + 5 * d.stddev()))],
  },
  hypergeometric: {
    name: 'Hypergeometric',
    params: [
      param('N', 'population N', 2, 200, 1, 50),
      param('K', 'successes in population K', 0, 200, 1, 20),
      param('n', 'draws n', 1, 200, 1, 10),
    ],
    make: (p) => {
      const { N, K, n } = population(p)
      return Hypergeometric(N, K, n)
    },
    range: (p) => {
      const { N, K, n } = population(p)
      return [Math.max(0, n - (N - K)), Math.min(n, K)]
    },
  },
  exponential: {
    name: 'Exponential',
    params: [param('rate', 'rate λ', 0.1, 5, 0.1, 1)],
    make: ({ rate }) => Exponential(rate),
    range: ({ rate }) => [0, 6 / rate],
  },
  gaussian: {
    name: 'Gaussian',
    params: [param('mu', 'mean μ', -5, 5, 0.1, 0), param('sigma', 'standard deviation σ', 0.1, 5, 0.1, 1)],
    make: ({ mu, sigma }) => Normal(mu, sigma),
    range: ({ mu, sigma }) => [mu - 4 * sigma, mu + 4 * sigma],
  },
  cauchy: {
    name: 'Cauchy',
    params: [loc('location x₀'), scale('scale γ')],
    make: ({ loc, scale }) => Cauchy(loc, scale),
    range: ({ loc, scale }) => [loc - 10 * scale, loc + 10 * scale],
  },
  gumbel: {
    name: 'Gumbel',
    params: [loc(), scale('scale β')],
    make: ({ loc, scale }) => Gumbel(loc, scale),
    range: ({ loc, scale }) => [loc - 3 * scale, loc + 7 * scale],
  },
  logistic: {
    name: 'Logistic',
    params: [loc(), scale('scale s')],
    make: ({ loc, scale }) => Logistic(loc, scale),
    range: ({ loc, scale }) => [loc - 8 * scale, loc + 8 * scale],
  },
  laplace: {
    name: 'Laplace',
    params: [loc(), scale('scale b')],
    make: ({ loc, scale }) => Laplace(loc, scale),
    range: ({ loc, scale }) => [loc - 6 * scale, loc + 6 * scale],
  },
  gamma: {
    name: 'Gamma',
    params: [param('shape', 'shape α', 0.5, 20, 0.1, 2), param('rate', 'rate β', 0.1, 5, 0.1, 1)],
    make: ({ shape, rate }) => Gamma(shape, rate),
    range: positive(0.99),
  },
  'inverse-gamma': {
    name: 'Inverse gamma',
    params: [param('shape', 'shape α', 0.5, 20, 0.1, 3), param('scale', 'scale β', 0.1, 10, 0.1, 2)],
    make: ({ shape, scale }) => InverseGamma(shape, scale),
    // The right tail is polynomial, so stop at the 95th percentile or the plot is mostly empty tail.
    range: positive(0.95),
  },
  beta: {
    name: 'Beta',
    params: [param('a', 'α', 0.2, 20, 0.1, 2), param('b', 'β', 0.2, 20, 0.1, 5)],
    make: ({ a, b }) => Beta(a, b),
    // Open interval: for α or β below 1 the density is infinite at an endpoint.
    range: () => [0.002, 0.998],
  },
  'student-t': {
    name: 'Student t',
    params: [param('nu', 'degrees of freedom ν', 0.5, 30, 0.5, 3)],
    make: ({ nu }) => StudentT(nu, 0, 1),
    range: () => [-6, 6],
  },
  'log-normal': {
    name: 'Log-normal',
    params: [param('mu', 'log-mean μ', -2, 2, 0.1, 0), param('sigma', 'log-standard deviation σ', 0.1, 2, 0.05, 0.5)],
    make: ({ mu, sigma }) => LogNormal(mu, sigma),
    range: ({ mu }, d) => [0.001 * Math.exp(mu), d.quantile(0.98)],
  },
  'chi-squared': {
    name: 'χ²',
    params: [param('k', 'degrees of freedom k', 1, 50, 1, 4)],
    make: ({ k }) => ChiSquare(k),
    range: positive(0.995),
  },
  f: {
    name: 'F',
    params: [
      param('d1', 'numerator degrees of freedom d₁', 1, 50, 1, 5),
      param('d2', 'denominator degrees of freedom d₂', 1, 50, 1, 10),
    ],
    make: ({ d1, d2 }) => FisherSnedecor(d1, d2),
    range: (p, d) => {
      const [lo, hi] = positive(0.98)(p, d)
      return [lo, Math.min(hi, 8)]
    },
  },
}

export function distributionSpec(id: string): DistributionSpec {
  const d = specs[id]
  if (!d) throw new Error(`unknown distribution "${id}" (see site/src/components/widgets/distribution-specs.ts)`)
  return d
}
