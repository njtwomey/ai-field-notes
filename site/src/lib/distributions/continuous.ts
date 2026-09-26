/**
 * Continuous distributions beyond the originals in index.ts: gamma, inverse gamma, beta, Student t, Laplace,
 * log-normal, χ² and F. Registered in index.ts.
 */
import {
  incompleteBeta,
  incompleteGamma,
  invertCdf,
  logGamma,
  normalCdf,
  normalQuantile,
  studentTCdf,
} from '@/lib/math/special'
import type { Distribution } from './index'

/** Plot up to the 99th percentile (or `u`), starting just above 0 where the density may be infinite. */
function positiveRange(cdf: (x: number) => number, guess: number, u = 0.99): [number, number] {
  const hi = invertCdf(cdf, u, 0, Math.max(guess, 1e-3))
  return [hi * 1e-3, hi]
}

const logBeta = (a: number, b: number) => logGamma(a) + logGamma(b) - logGamma(a + b)

export const gamma: Distribution = {
  id: 'gamma',
  name: 'Gamma',
  discrete: false,
  params: [
    { key: 'shape', label: 'shape α', min: 0.5, max: 20, step: 0.1, value: 2 },
    { key: 'rate', label: 'rate β', min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  density: (x, { shape, rate }) =>
    x <= 0 ? 0 : Math.exp(shape * Math.log(rate) + (shape - 1) * Math.log(x) - rate * x - logGamma(shape)),
  cdf: (x, { shape, rate }) => incompleteGamma(shape, rate * x),
  mean: ({ shape, rate }) => shape / rate,
  variance: ({ shape, rate }) => shape / (rate * rate),
  range: (p) => positiveRange((x) => gamma.cdf(x, p), gamma.mean(p)),
}

export const inverseGamma: Distribution = {
  id: 'inverse-gamma',
  name: 'Inverse gamma',
  discrete: false,
  params: [
    { key: 'shape', label: 'shape α', min: 0.5, max: 20, step: 0.1, value: 3 },
    { key: 'scale', label: 'scale β', min: 0.1, max: 10, step: 0.1, value: 2 },
  ],
  density: (x, { shape, scale }) =>
    x <= 0 ? 0 : Math.exp(shape * Math.log(scale) - (shape + 1) * Math.log(x) - scale / x - logGamma(shape)),
  cdf: (x, { shape, scale }) => (x <= 0 ? 0 : 1 - incompleteGamma(shape, scale / x)),
  mean: ({ shape, scale }) => (shape > 1 ? scale / (shape - 1) : NaN),
  variance: ({ shape, scale }) => (shape > 2 ? (scale * scale) / ((shape - 1) ** 2 * (shape - 2)) : NaN),
  // The right tail is polynomial, so stop at the 95th percentile or the plot is mostly empty tail.
  range: (p) => positiveRange((x) => inverseGamma.cdf(x, p), p.scale / (p.shape + 1), 0.95),
}

export const beta: Distribution = {
  id: 'beta',
  name: 'Beta',
  discrete: false,
  params: [
    { key: 'a', label: 'α', min: 0.2, max: 20, step: 0.1, value: 2 },
    { key: 'b', label: 'β', min: 0.2, max: 20, step: 0.1, value: 5 },
  ],
  density: (x, { a, b }) =>
    x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta(a, b)),
  cdf: (x, { a, b }) => incompleteBeta(x, a, b),
  mean: ({ a, b }) => a / (a + b),
  variance: ({ a, b }) => (a * b) / ((a + b) ** 2 * (a + b + 1)),
  // Open interval: for α or β below 1 the density is infinite at an endpoint.
  range: () => [0.002, 0.998],
}

export const studentT: Distribution = {
  id: 'student-t',
  name: 'Student t',
  discrete: false,
  params: [{ key: 'nu', label: 'degrees of freedom ν', min: 0.5, max: 30, step: 0.5, value: 3 }],
  density: (x, { nu }) =>
    Math.exp(
      logGamma((nu + 1) / 2) -
        logGamma(nu / 2) -
        0.5 * Math.log(nu * Math.PI) -
        ((nu + 1) / 2) * Math.log1p((x * x) / nu),
    ),
  cdf: (x, { nu }) => studentTCdf(x, nu),
  mean: ({ nu }) => (nu > 1 ? 0 : NaN),
  variance: ({ nu }) => (nu > 2 ? nu / (nu - 2) : nu > 1 ? Infinity : NaN),
  range: () => [-6, 6],
}

export const laplace: Distribution = {
  id: 'laplace',
  name: 'Laplace',
  discrete: false,
  params: [
    { key: 'loc', label: 'location μ', min: -5, max: 5, step: 0.1, value: 0 },
    { key: 'scale', label: 'scale b', min: 0.1, max: 5, step: 0.1, value: 1 },
  ],
  density: (x, { loc, scale }) => Math.exp(-Math.abs(x - loc) / scale) / (2 * scale),
  cdf: (x, { loc, scale }) => (x < loc ? 0.5 * Math.exp((x - loc) / scale) : 1 - 0.5 * Math.exp(-(x - loc) / scale)),
  quantile: (u, { loc, scale }) => (u < 0.5 ? loc + scale * Math.log(2 * u) : loc - scale * Math.log(2 - 2 * u)),
  mean: ({ loc }) => loc,
  variance: ({ scale }) => 2 * scale * scale,
  range: ({ loc, scale }) => [loc - 6 * scale, loc + 6 * scale],
}

export const logNormal: Distribution = {
  id: 'log-normal',
  name: 'Log-normal',
  discrete: false,
  params: [
    { key: 'mu', label: 'log-mean μ', min: -2, max: 2, step: 0.1, value: 0 },
    { key: 'sigma', label: 'log-standard deviation σ', min: 0.1, max: 2, step: 0.05, value: 0.5 },
  ],
  density: (x, { mu, sigma }) =>
    x <= 0 ? 0 : Math.exp(-((Math.log(x) - mu) ** 2) / (2 * sigma * sigma)) / (x * sigma * Math.sqrt(2 * Math.PI)),
  cdf: (x, { mu, sigma }) => (x <= 0 ? 0 : normalCdf((Math.log(x) - mu) / sigma)),
  quantile: (u, { mu, sigma }) => Math.exp(mu + sigma * normalQuantile(u)),
  mean: ({ mu, sigma }) => Math.exp(mu + (sigma * sigma) / 2),
  variance: ({ mu, sigma }) => (Math.exp(sigma * sigma) - 1) * Math.exp(2 * mu + sigma * sigma),
  range: ({ mu, sigma }) => [0.001 * Math.exp(mu), Math.exp(mu + sigma * normalQuantile(0.98))],
}

export const chiSquared: Distribution = {
  id: 'chi-squared',
  name: 'χ²',
  discrete: false,
  params: [{ key: 'k', label: 'degrees of freedom k', min: 1, max: 50, step: 1, value: 4 }],
  density: (x, { k }) =>
    x <= 0 ? 0 : Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.LN2 - logGamma(k / 2)),
  cdf: (x, { k }) => incompleteGamma(k / 2, x / 2),
  mean: ({ k }) => k,
  variance: ({ k }) => 2 * k,
  range: (p) => positiveRange((x) => chiSquared.cdf(x, p), p.k, 0.995),
}

export const fDist: Distribution = {
  id: 'f',
  name: 'F',
  discrete: false,
  params: [
    { key: 'd1', label: 'numerator degrees of freedom d₁', min: 1, max: 50, step: 1, value: 5 },
    { key: 'd2', label: 'denominator degrees of freedom d₂', min: 1, max: 50, step: 1, value: 10 },
  ],
  density: (x, { d1, d2 }) =>
    x <= 0
      ? 0
      : Math.exp(
          (d1 / 2) * Math.log(d1 / d2) +
            (d1 / 2 - 1) * Math.log(x) -
            ((d1 + d2) / 2) * Math.log1p((d1 * x) / d2) -
            logBeta(d1 / 2, d2 / 2),
        ),
  cdf: (x, { d1, d2 }) => (x <= 0 ? 0 : incompleteBeta((d1 * x) / (d1 * x + d2), d1 / 2, d2 / 2)),
  mean: ({ d2 }) => (d2 > 2 ? d2 / (d2 - 2) : NaN),
  variance: ({ d1, d2 }) =>
    d2 > 4 ? (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) ** 2 * (d2 - 4)) : d2 > 2 ? Infinity : NaN,
  // Heavy right tail for small d₂: cap the plot so the body stays visible.
  range: (p) => {
    const [lo, hi] = positiveRange((x) => fDist.cdf(x, p), 1, 0.98)
    return [lo, Math.min(hi, 8)]
  },
}

export const continuous = [gamma, inverseGamma, beta, studentT, laplace, logNormal, chiSquared, fDist]
