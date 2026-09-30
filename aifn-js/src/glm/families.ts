/**
 * Link functions and exponential-dispersion families for generalised linear models (Nelder and Wedderburn, 1972;
 * McCullagh and Nelder, 1989, "Generalized Linear Models", 2nd ed., ch. 2 and Table 2.1). Links are compositions of
 * `aifn/tensor` and `aifn/special` primitives, so they accept numbers, tensors and traced values and are
 * differentiable. Variance functions and unit deviances likewise.
 */

import { Bernoulli, Binomial, Gamma, NegativeBinomial, Normal, Poisson, xlogy } from 'aifn/distributions'
import type { Distribution } from 'aifn/estimators'
import { normal, uniform, type Stream } from 'aifn/random'
import { logChoose, logGamma, normalCdf, normalPdf, normalQuantile, sigmoid, logit } from 'aifn/special'
import {
  add,
  div,
  exp,
  fromData,
  log,
  mul,
  neg,
  pow,
  sqrt,
  square,
  sub,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn/tensor'

// ── Links ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The name of a link function. */
export type LinkName = 'identity' | 'log' | 'logit' | 'probit' | 'cloglog' | 'inverse' | 'inverse-squared' | 'sqrt'

/** A link g: the linear predictor is η = g(μ). */
export interface Link {
  readonly name: LinkName
  /** η = g(μ). */
  link(mu: Value): Value
  /** μ = g⁻¹(η). */
  inverse(eta: Value): Value
  /** dμ/dη at η. */
  derivative(eta: Value): Value
}

const LINKS: Record<LinkName, Omit<Link, 'name'>> = {
  identity: { link: (m) => m, inverse: (e) => e, derivative: onesLike },
  log: { link: log, inverse: exp, derivative: exp },
  logit: { link: logit, inverse: sigmoid, derivative: (e) => mul(sigmoid(e), sigmoid(neg(e))) },
  probit: { link: normalQuantile, inverse: normalCdf, derivative: normalPdf },
  cloglog: {
    link: (m) => log(neg(log(sub(1, m)))),
    inverse: (e) => sub(1, exp(neg(exp(e)))),
    derivative: (e) => exp(sub(e, exp(e))),
  },
  inverse: { link: (m) => div(1, m), inverse: (e) => div(1, e), derivative: (e) => neg(div(1, square(e))) },
  'inverse-squared': {
    link: (m) => div(1, square(m)),
    inverse: (e) => div(1, sqrt(e)),
    derivative: (e) => mul(-0.5, pow(e, -1.5)),
  },
  sqrt: { link: sqrt, inverse: square, derivative: (e) => mul(2, e) },
}

/** 1 in the shape of v (a number, tensor or traced value). */
function onesLike(v: Value): Value {
  return add(mul(0, v), 1)
}

/** The link function by name. */
export function link(name: LinkName): Link {
  const l = LINKS[name]
  if (!l) throw new Error(`link: unknown link "${name}"`)
  return { name, ...l }
}

// ── Families ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The name of a family. */
export type FamilyName = 'gaussian' | 'binomial' | 'poisson' | 'gamma' | 'inverse-gaussian' | 'negative-binomial'

/**
 * An exponential-dispersion family: Var(y) = φ V(μ)/w for prior weight w. The deviance is Σ wᵢ d(yᵢ, μᵢ) with the unit
 * deviance d. Binomial responses are proportions with the number of trials as the prior weight (as R's `glm`).
 */
export interface Family {
  readonly name: FamilyName
  /** Parameters fixed when the family was made (e.g. the negative binomial's θ). */
  readonly params: Record<string, number>
  /** The default link (the canonical one except for the negative binomial, whose conventional link is log). */
  readonly defaultLink: LinkName
  /** φ when it is known (1 for binomial, Poisson and negative binomial); null when it is estimated. */
  readonly dispersion: number | null
  /** V(μ). */
  variance(mu: Value): Value
  /** d(y, μ), the unit deviance (twice the log-likelihood ratio of the saturated model for one observation). */
  unitDeviance(y: Value, mu: Value): Value
  /** True when every μ lies in the family's mean space. */
  validMean(mu: Tensor): boolean
  /** A starting mean for IRLS from the responses and prior weights (as R's `mustart`). */
  initialMean(y: Tensor, weights: Tensor): Tensor
  /** The log-likelihood Σ wᵢ log p(yᵢ | μᵢ, φ) (binomial: the log-likelihood of the counts wᵢyᵢ). */
  logLikelihood(y: Tensor, mu: Tensor, dispersion: number, weights: Tensor): number
  /** The distribution of a new response given its mean (and dispersion; prior weights as trials or precision). */
  predictive(mu: Tensor, dispersion: number, weights?: Tensor): Distribution
}

const all = (mu: Tensor, ok: (m: number) => boolean) => toFlat(mu).every(ok)
const zip = (a: Tensor, b: Tensor, f: (x: number, y: number, i: number) => number) => {
  const x = toFlat(a)
  const y = toFlat(b)
  return fromData(
    Float64Array.from(x, (v, i) => f(v, y[i], i)),
    a.shape,
  )
}
const total = (t: Tensor) => toFlat(t).reduce((s, v) => s + v, 0)

/** The Gaussian family: V(μ) = 1, d = (y − μ)², canonical link identity. */
export function gaussian(): Family {
  return {
    name: 'gaussian',
    params: {},
    defaultLink: 'identity',
    dispersion: null,
    variance: (mu) => onesLike(mu),
    unitDeviance: (y, mu) => square(sub(y, mu)),
    validMean: (mu) => all(mu, Number.isFinite),
    initialMean: (y) => y,
    logLikelihood: (y, mu, phi, w) =>
      total(
        zip(
          y,
          mu,
          (a, b, i) => toFlat(w)[i] * -0.5 * (Math.log((2 * Math.PI * phi) / toFlat(w)[i]) + (a - b) ** 2 / phi),
        ),
      ),
    predictive: (mu, phi, w) =>
      Normal(
        mu,
        w
          ? zip(w, w, (v) => Math.sqrt(phi / v))
          : fromData(new Float64Array(mu.shape[0]).fill(Math.sqrt(phi)), mu.shape),
      ),
  }
}

/**
 * The binomial family for proportions y ∈ [0, 1] with wᵢ trials (Bernoulli when every w = 1): V(μ) = μ(1 − μ),
 * d = 2[y log(y/μ) + (1 − y) log((1 − y)/(1 − μ))], canonical link logit.
 */
export function binomial(): Family {
  return {
    name: 'binomial',
    params: {},
    defaultLink: 'logit',
    dispersion: 1,
    variance: (mu) => mul(mu, sub(1, mu)),
    unitDeviance: (y, mu) =>
      mul(2, add(sub(xlogy(y, y), xlogy(y, mu)), sub(xlogy(sub(1, y), sub(1, y)), xlogy(sub(1, y), sub(1, mu))))),
    validMean: (mu) => all(mu, (m) => m > 0 && m < 1),
    initialMean: (y, w) => zip(y, w, (a, b) => (b * a + 0.5) / (b + 1)),
    logLikelihood: (y, mu, _phi, w) => {
      const wf = toFlat(w)
      return total(
        zip(y, mu, (a, b, i) => {
          const m = wf[i]
          const k = Math.round(m * a)
          return (
            (logChoose(m, k) as number) + (k === 0 ? 0 : k * Math.log(b)) + (m - k === 0 ? 0 : (m - k) * Math.log1p(-b))
          )
        }),
      )
    },
    predictive: (mu, _phi, w) => (w && !toFlat(w).every((v) => v === 1) ? Binomial(w, mu) : Bernoulli(mu)),
  }
}

/** The Poisson family: V(μ) = μ, d = 2[y log(y/μ) − (y − μ)], canonical link log. */
export function poisson(): Family {
  return {
    name: 'poisson',
    params: {},
    defaultLink: 'log',
    dispersion: 1,
    variance: (mu) => mu,
    unitDeviance: (y, mu) => mul(2, sub(sub(xlogy(y, y), xlogy(y, mu)), sub(y, mu))),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => zip(y, y, (a) => a + 0.1),
    logLikelihood: (y, mu, _phi, w) =>
      total(zip(y, mu, (a, b, i) => toFlat(w)[i] * (xlogy(a, b) - b - (logGamma(a + 1) as number)))),
    predictive: (mu) => Poisson(mu),
  }
}

/** The gamma family: V(μ) = μ², d = 2[−log(y/μ) + (y − μ)/μ], canonical link inverse; shape 1/φ. */
export function gamma(): Family {
  return {
    name: 'gamma',
    params: {},
    defaultLink: 'inverse',
    dispersion: null,
    variance: (mu) => square(mu),
    unitDeviance: (y, mu) => mul(2, add(neg(log(div(y, mu))), div(sub(y, mu), mu))),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => y,
    logLikelihood: (y, mu, phi, w) =>
      total(
        zip(y, mu, (a, b, i) => {
          const shape = toFlat(w)[i] / phi
          return shape * Math.log((shape * a) / b) - (shape * a) / b - Math.log(a) - (logGamma(shape) as number)
        }),
      ),
    predictive: (mu, phi, w) => {
      const shape = w ? zip(w, w, (v) => v / phi) : fromData(new Float64Array(mu.shape[0]).fill(1 / phi), mu.shape)
      return Gamma(
        shape,
        zip(shape, mu, (s, m) => s / m),
      )
    },
  }
}

/** The inverse Gaussian family: V(μ) = μ³, d = (y − μ)²/(μ²y), canonical link 1/μ². */
export function inverseGaussian(): Family {
  return {
    name: 'inverse-gaussian',
    params: {},
    defaultLink: 'inverse-squared',
    dispersion: null,
    variance: (mu) => pow(mu, 3),
    unitDeviance: (y, mu) => div(square(sub(y, mu)), mul(square(mu), y)),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => y,
    logLikelihood: (y, mu, phi, w) =>
      total(
        zip(y, mu, (a, b, i) => {
          const p = phi / toFlat(w)[i]
          return -0.5 * (Math.log(2 * Math.PI * p * a ** 3) + (a - b) ** 2 / (p * b * b * a))
        }),
      ),
    predictive: (mu, phi, w) =>
      inverseGaussianPredictive(
        mu,
        w ? zip(w, w, (v) => v / phi) : fromData(new Float64Array(mu.shape[0]).fill(1 / phi), mu.shape),
      ),
  }
}

/**
 * The negative binomial family with fixed shape θ > 0 (NB2): V(μ) = μ + μ²/θ,
 * d = 2[y log(y/μ) − (y + θ) log((y + θ)/(μ + θ))], default link log (Hilbe, 2011, "Negative Binomial Regression").
 * The predictive counts failures before θ successes with p = θ/(θ + μ), so its mean is μ.
 */
export function negativeBinomial(theta: number): Family {
  if (!(theta > 0)) throw new Error('negativeBinomial: θ must be positive')
  return {
    name: 'negative-binomial',
    params: { theta },
    defaultLink: 'log',
    dispersion: 1,
    variance: (mu) => add(mu, div(square(mu), theta)),
    unitDeviance: (y, mu) =>
      mul(
        2,
        sub(
          sub(xlogy(y, y), xlogy(y, mu)),
          sub(xlogy(add(y, theta), add(y, theta)), xlogy(add(y, theta), add(mu, theta))),
        ),
      ),
    validMean: (mu) => all(mu, (m) => m > 0 && Number.isFinite(m)),
    initialMean: (y) => zip(y, y, (a) => a + 0.1),
    logLikelihood: (y, mu, _phi, w) =>
      total(
        zip(y, mu, (a, b, i) => {
          const ll =
            (logGamma(a + theta) as number) -
            (logGamma(theta) as number) -
            (logGamma(a + 1) as number) +
            theta * Math.log(theta / (theta + b)) +
            xlogy(a, b / (theta + b))
          return toFlat(w)[i] * ll
        }),
      ),
    predictive: (mu) =>
      NegativeBinomial(
        theta,
        zip(mu, mu, (m) => theta / (theta + m)),
      ),
  }
}

/** A family by name (the negative binomial needs θ). */
export function family(name: FamilyName, params: { theta?: number } = {}): Family {
  switch (name) {
    case 'gaussian':
      return gaussian()
    case 'binomial':
      return binomial()
    case 'poisson':
      return poisson()
    case 'gamma':
      return gamma()
    case 'inverse-gaussian':
      return inverseGaussian()
    case 'negative-binomial':
      return negativeBinomial(params.theta ?? 1)
  }
  throw new Error(`family: unknown family "${name}"`)
}

// ── Inverse Gaussian predictive ──────────────────────────────────────────────────────────────────────────────────

/**
 * A batch of inverse Gaussian laws IG(μ, λ) with mean μ and shape λ (variance μ³/λ); `aifn/distributions` has none
 * yet, so this minimal object satisfies the estimators' distribution contract. Draws by Michael, Schucany and Haas
 * (1976).
 */
function inverseGaussianPredictive(mu: Tensor, lambda: Tensor): Distribution & { variance(): Tensor } {
  const m = toFlat(mu)
  const l = toFlat(lambda)
  return {
    name: 'InverseGaussian',
    batchShape: mu.shape,
    eventShape: [],
    logProb: (x: Tensor) => {
      const xs = toFlat(x)
      return fromData(
        Float64Array.from(m, (mi, i) => {
          const xi = xs.length === 1 ? xs[0] : xs[i]
          return 0.5 * Math.log(l[i] / (2 * Math.PI * xi ** 3)) - (l[i] * (xi - mi) ** 2) / (2 * mi * mi * xi)
        }),
        mu.shape,
      )
    },
    sample: (s: Stream, { shape }: { shape: readonly number[] }) => {
      const reps = shape.reduce((a, b) => a * b, 1)
      const out = new Float64Array(reps * m.length)
      const z = toFlat(normal(s.child('normal'), 0, 1, { shape: [reps * m.length] }) as Tensor)
      const u = toFlat(uniform(s.child('uniform'), 0, 1, { shape: [reps * m.length] }) as Tensor)
      for (let k = 0; k < out.length; k++) {
        const i = k % m.length
        const v = z[k] * z[k]
        const mi = m[i]
        const x = mi + (mi * mi * v) / (2 * l[i]) - (mi / (2 * l[i])) * Math.sqrt(4 * mi * l[i] * v + mi * mi * v * v)
        out[k] = u[k] <= mi / (mi + x) ? x : (mi * mi) / x
      }
      return fromData(out, [...shape, ...mu.shape])
    },
    mean: () => mu,
    variance: () =>
      fromData(
        Float64Array.from(m, (mi, i) => mi ** 3 / l[i]),
        mu.shape,
      ),
  }
}
