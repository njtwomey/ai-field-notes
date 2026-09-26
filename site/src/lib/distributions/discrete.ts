/** Discrete distributions beyond the Poisson and binomial in index.ts. Registered in the list at the end of index.ts. */
import { logChoose, logFactorial, logGamma } from '@/lib/math/special'
import type { Distribution, Params } from './index'

/** Sum of the pmf over the integers from `lo` to ⌊x⌋. */
function cdfBySum(d: Distribution, lo: number, x: number, p: Params): number {
  let total = 0
  for (let k = lo; k <= Math.floor(x); k++) total += d.density(k, p)
  return Math.min(total, 1)
}

const isCount = (k: number) => k >= 0 && Number.isInteger(k)

export const bernoulli: Distribution = {
  id: 'bernoulli',
  name: 'Bernoulli',
  discrete: true,
  params: [{ key: 'p', label: 'success probability p', min: 0, max: 1, step: 0.01, value: 0.3 }],
  density: (k, { p }) => (k === 1 ? p : k === 0 ? 1 - p : 0),
  cdf: (x, { p }) => (x < 0 ? 0 : x < 1 ? 1 - p : 1),
  quantile: (u, { p }) => (u <= 1 - p ? 0 : 1),
  mean: ({ p }) => p,
  variance: ({ p }) => p * (1 - p),
  range: () => [0, 1],
}

/** Trials up to and including the first success, on {1, 2, …}. */
export const geometric: Distribution = {
  id: 'geometric',
  name: 'Geometric',
  discrete: true,
  params: [{ key: 'p', label: 'success probability p', min: 0.02, max: 1, step: 0.01, value: 0.25 }],
  density: (k, { p }) => (k >= 1 && Number.isInteger(k) ? p * (1 - p) ** (k - 1) : 0),
  cdf: (x, { p }) => (x < 1 ? 0 : 1 - (1 - p) ** Math.floor(x)),
  quantile: (u, { p }) => (p === 1 ? 1 : Math.max(1, Math.ceil(Math.log(1 - u) / Math.log(1 - p)))),
  mean: ({ p }) => 1 / p,
  variance: ({ p }) => (1 - p) / (p * p),
  range: ({ p }) => [1, p === 1 ? 5 : Math.max(10, Math.ceil(Math.log(0.001) / Math.log(1 - p)))],
}

/** Failures before the r-th success, on {0, 1, 2, …}. r may be any positive real (the gamma–Poisson mixture). */
export const negativeBinomial: Distribution = {
  id: 'negative-binomial',
  name: 'Negative binomial',
  discrete: true,
  params: [
    { key: 'r', label: 'successes r', min: 0.5, max: 30, step: 0.5, value: 3 },
    { key: 'p', label: 'success probability p', min: 0.02, max: 0.99, step: 0.01, value: 0.4 },
  ],
  density: (k, { r, p }) =>
    isCount(k) ? Math.exp(logGamma(k + r) - logGamma(r) - logFactorial(k) + r * Math.log(p) + k * Math.log(1 - p)) : 0,
  cdf: (x, p) => cdfBySum(negativeBinomial, 0, x, p),
  mean: ({ r, p }) => (r * (1 - p)) / p,
  variance: ({ r, p }) => (r * (1 - p)) / (p * p),
  range: (params) => {
    const m = negativeBinomial.mean(params)
    return [0, Math.max(10, Math.ceil(m + 5 * Math.sqrt(negativeBinomial.variance(params))))]
  },
}

/** Sliders move independently, so K and n are clamped to the population size N. */
const population = ({ N, K, n }: Params) => ({ N, K: Math.min(K, N), n: Math.min(n, N) })

/** Successes in n draws without replacement from N items of which K are successes. */
export const hypergeometric: Distribution = {
  id: 'hypergeometric',
  name: 'Hypergeometric',
  discrete: true,
  params: [
    { key: 'N', label: 'population N', min: 2, max: 200, step: 1, value: 50 },
    { key: 'K', label: 'successes in population K', min: 0, max: 200, step: 1, value: 20 },
    { key: 'n', label: 'draws n', min: 1, max: 200, step: 1, value: 10 },
  ],
  density: (k, params) => {
    const { N, K, n } = population(params)
    if (!isCount(k) || k > K || k > n || n - k > N - K) return 0
    return Math.exp(logChoose(K, k) + logChoose(N - K, n - k) - logChoose(N, n))
  },
  cdf: (x, p) => cdfBySum(hypergeometric, 0, x, p),
  mean: (params) => {
    const { N, K, n } = population(params)
    return (n * K) / N
  },
  variance: (params) => {
    const { N, K, n } = population(params)
    return N === 1 ? 0 : ((n * K) / N) * (1 - K / N) * ((N - n) / (N - 1))
  },
  range: (params) => {
    const { N, K, n } = population(params)
    return [Math.max(0, n - (N - K)), Math.min(n, K)]
  },
}
