/**
 * Known truth for synthetic datasets. When the generating process has a closed form, a dataset carries it in
 * `meta.truth`, so a figure can draw the Bayes-optimal boundary and curves beside a fitted model's.
 *
 * A classification truth is built from class-conditional densities p(x | y = j) and clean class priors πⱼ, followed by
 * a list of label operations: noise matrices T (T[i][j] = P(observed j | label i)) and class reweightings w (from
 * resampling by label). The joint of x and the observed label is then
 *
 *   p(x, ỹ) ∝ ((π ∘ p(x)) T₁ ∘ w₁ …)_ỹ,
 *
 * applied left to right, and the Bayes posterior is that vector normalised. Modifiers compose by editing this model.
 */

import { normalCdf } from 'aifn/special'

/** A point, one number per feature. */
export type Row = ArrayLike<number>

/** A label operation applied after the clean labels are drawn. */
export type LabelOp =
  | { kind: 'noise'; matrix: number[][] }
  | {
      kind: 'weights'
      /** Relative weight of each observed class (resampling by label). */
      weights: number[]
    }

/**
 * Points drawn from the population's marginal of x with importance weights summing to one, for Monte Carlo estimates
 * (the Bayes error when no closed form applies).
 */
export interface Reference {
  rows: Float64Array[]
  weights: Float64Array
}

/** The parts a classification truth is built from; modifiers edit these. */
export interface ClassModel {
  classes: number
  /** Clean class proportions πⱼ in the population. */
  priors: number[]
  /** log p(x | clean y = j) for each class j (up to a term shared by all classes). −∞ outside a class's support. */
  logDensity: (x: Row) => number[]
  /** Label operations in order. */
  ops: LabelOp[]
  /** A weighted sample of the population's x (lazy; computed on first use of the Bayes error). */
  reference: () => Reference
  /** The Bayes error of the clean problem for given priors, when it has a closed form. */
  closedForm?: (priors: number[]) => number
  /** True once x has been reweighted (covariate shift), so class proportions must be estimated. */
  shifted?: boolean
  /** What the densities are, for captions: e.g. "two Gaussian classes". */
  family: string
}

/**
 * The Bayes-optimal classifier of a synthetic classification problem. Functions take one point (one number per
 * feature) and refer to the observed labels `y` of the dataset, after any label noise or resampling.
 */
export interface ClassificationTruth {
  kind: 'classification'
  classes: number
  /** Clean class proportions in the population. */
  priors: readonly number[]
  /** Observed class proportions in the population. */
  readonly prevalence: readonly number[]
  /** log p(x | clean y = j), per class, up to a term shared by all classes. */
  logDensity(x: Row): number[]
  /** P(clean y = j | x), per class. */
  cleanPosterior(x: Row): number[]
  /** P(observed y = j | x), per class: the Bayes posterior. NaN where x has zero density under every class. */
  posterior(x: Row): number[]
  /** P(y = 1 | x): the positive-class posterior (class 1 against the rest when there are more than two classes). */
  probability(x: Row): number
  /** The log odds of class 1: a Bayes-optimal score for ROC and precision–recall curves (any increasing map works). */
  score(x: Row): number
  /** The Bayes error E[1 − maxⱼ P(y = j | x)], the lowest error any classifier can reach. Computed on first access. */
  readonly bayesError: number
  /** How `bayesError` is computed: exactly, or by Monte Carlo over a reference sample of the population. */
  readonly bayesErrorMethod: 'closed form' | 'monte carlo'
  /** Standard error of a Monte Carlo `bayesError` (0 in closed form). */
  readonly bayesErrorSe: number
  /** The model the truth is built from (for modifiers). */
  model: ClassModel
}

/**
 * The truth of a synthetic regression problem y = m(x) + ε, ε ~ N(0, σ(x)²).
 */
export interface RegressionTruth {
  kind: 'regression'
  /** The regression function m(x) = E[y | x]. */
  mean(x: Row): number
  /** The noise standard deviation (at the left end of the input range when it varies). */
  noiseSd: number
  /** σ(x), the noise standard deviation at x. */
  noiseSdAt(x: Row): number
  /** The Bayes risk under squared loss, E[σ(x)²]: the lowest mean squared error any predictor can reach. */
  bayesRisk: number
  /** Fraction of targets replaced by gross outliers (see `withOutliers`), 0 by default. */
  outlierFraction: number
}

export type Truth = ClassificationTruth | RegressionTruth

/** Number of reference points drawn for Monte Carlo Bayes errors. */
export const REFERENCE_SIZE = 6000

/** log Σ exp(a). */
export function logSumExp(a: ArrayLike<number>): number {
  let m = -Infinity
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]
  if (m === -Infinity) return -Infinity
  let s = 0
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - m)
  return m + Math.log(s)
}

function lazy<T>(f: () => T): () => T {
  let cached: { value: T } | undefined
  return () => (cached ??= { value: f() }).value
}

/** Apply the label operations to an unnormalised vector over clean classes. */
function applyOps(p: number[], ops: readonly LabelOp[]): number[] {
  let v = p
  for (const op of ops) {
    if (op.kind === 'weights') v = v.map((a, j) => a * op.weights[j])
    else {
      const m = op.matrix
      const out = new Array<number>(m[0].length).fill(0)
      for (let i = 0; i < v.length; i++) if (v[i] !== 0) for (let j = 0; j < out.length; j++) out[j] += v[i] * m[i][j]
      v = out
    }
  }
  return v
}

function normalise(v: number[]): number[] {
  const s = v.reduce((a, b) => a + b, 0)
  return v.map((a) => a / s)
}

/** The clean log joint log πⱼ + log p(x | j). */
function logJoint(model: ClassModel, x: Row): number[] {
  const ld = model.logDensity(x)
  return ld.map((l, j) => Math.log(model.priors[j]) + l)
}

function posteriorOf(model: ClassModel, x: Row): number[] {
  const a = logJoint(model, x)
  const m = Math.max(...a)
  if (!(m > -Infinity)) return a.map(() => NaN)
  return normalise(
    applyOps(
      a.map((v) => Math.exp(v - m)),
      model.ops,
    ),
  )
}

/**
 * The closed-form Bayes error, when the model has one and its label operations allow it: any class reweightings
 * (which change the effective priors), optionally followed by one symmetric binary flip at rate ρ ≤ 1/2, which maps an
 * error e to ρ + (1 − 2ρ) e.
 */
function closedFormError(model: ClassModel): number | undefined {
  if (!model.closedForm || model.shifted) return undefined
  let priors = [...model.priors]
  let flip: number | undefined
  for (const [i, op] of model.ops.entries()) {
    if (op.kind === 'weights') priors = priors.map((p, j) => p * op.weights[j])
    else {
      const m = op.matrix
      const symmetricBinary = m.length === 2 && i === model.ops.length - 1 && m[0][1] === m[1][0] && m[0][1] <= 0.5
      if (!symmetricBinary) return undefined
      flip = m[0][1]
    }
  }
  const e = model.closedForm(normalise(priors))
  return flip === undefined ? e : flip + (1 - 2 * flip) * e
}

/** Build the truth object from a model. */
export function classificationTruth(model: ClassModel): ClassificationTruth {
  const monteCarlo = lazy(() => {
    const { rows, weights } = model.reference()
    let mean = 0
    const errs = rows.map((r, i) => {
      const p = posteriorOf(model, r)
      const e = 1 - Math.max(...p)
      mean += weights[i] * (Number.isNaN(e) ? 0 : e)
      return Number.isNaN(e) ? 0 : e
    })
    // Standard error of a self-normalised weighted mean: sqrt(Σ wᵢ² (eᵢ − ē)²).
    let v = 0
    errs.forEach((e, i) => (v += weights[i] * weights[i] * (e - mean) ** 2))
    return { error: mean, se: Math.sqrt(v) }
  })
  const exact = lazy(() => closedFormError(model))
  const prevalence = lazy(() => {
    if (!model.shifted) return normalise(applyOps([...model.priors], model.ops))
    const { rows, weights } = model.reference()
    const out = new Array<number>(model.classes).fill(0)
    rows.forEach((r, i) => posteriorOf(model, r).forEach((p, j) => (out[j] += weights[i] * (Number.isNaN(p) ? 0 : p))))
    return normalise(out)
  })
  const positive = (x: Row) => {
    const p = posteriorOf(model, x)
    return p[1]
  }
  return {
    kind: 'classification',
    classes: model.classes,
    priors: model.priors,
    get prevalence() {
      return prevalence()
    },
    logDensity: model.logDensity,
    cleanPosterior: (x) => {
      const a = logJoint(model, x)
      const z = logSumExp(a)
      return a.map((v) => Math.exp(v - z))
    },
    posterior: (x) => posteriorOf(model, x),
    probability: positive,
    score: (x) => {
      const a = logJoint(model, x)
      if (model.ops.length === 0 && model.classes === 2) return a[1] - a[0]
      if (model.ops.length === 0) return a[1] - logSumExp(a.filter((_, j) => j !== 1))
      const p = posteriorOf(model, x)
      return Math.log(p[1]) - Math.log(1 - p[1])
    },
    get bayesError() {
      return exact() ?? monteCarlo().error
    },
    get bayesErrorMethod() {
      return exact() === undefined ? 'monte carlo' : 'closed form'
    },
    get bayesErrorSe() {
      return exact() === undefined ? monteCarlo().se : 0
    },
    model,
  }
}

// ── Class-conditional densities ─────────────────────────────────────────────────────────────────────────────────────

const LOG_2PI = Math.log(2 * Math.PI)

/** Φ(z), the standard normal cdf. */
export function phi(z: number): number {
  return normalCdf(z) as number
}

/** Lower Cholesky factor of a small symmetric positive-definite matrix (rows). Throws when it is not. */
export function choleskyRows(a: readonly (readonly number[])[]): number[][] {
  const d = a.length
  const l = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (let i = 0; i < d; i++)
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      if (i === j) {
        if (!(s > 0)) throw new RangeError('covariance is not positive definite')
        l[i][i] = Math.sqrt(s)
      } else l[i][j] = s / l[j][j]
    }
  return l
}

/** A Gaussian log density with precomputed Cholesky factor. */
export function gaussianLogDensity(mean: readonly number[], covariance: readonly (readonly number[])[]) {
  const d = mean.length
  const l = choleskyRows(covariance)
  let logDet = 0
  for (let i = 0; i < d; i++) logDet += 2 * Math.log(l[i][i])
  const z = new Float64Array(d)
  return (x: Row): number => {
    let q = 0
    for (let i = 0; i < d; i++) {
      let s = x[i] - mean[i]
      for (let k = 0; k < i; k++) s -= l[i][k] * z[k]
      z[i] = s / l[i][i]
      q += z[i] * z[i]
    }
    return -0.5 * (q + logDet + d * LOG_2PI)
  }
}

/** Squared Mahalanobis distance (a − b)ᵀ Σ⁻¹ (a − b). */
export function mahalanobis2(a: readonly number[], b: readonly number[], covariance: readonly (readonly number[])[]) {
  const l = choleskyRows(covariance)
  const d = a.length
  const z = new Array<number>(d).fill(0)
  let q = 0
  for (let i = 0; i < d; i++) {
    let s = a[i] - b[i]
    for (let k = 0; k < i; k++) s -= l[i][k] * z[k]
    z[i] = s / l[i][i]
    q += z[i] * z[i]
  }
  return q
}

/**
 * The Bayes error of two Gaussian classes with a shared covariance at Mahalanobis distance Δ and priors (π₀, π₁). The
 * log likelihood ratio L is N(±Δ²/2, Δ²) under each class and the Bayes rule says 1 when L > t = log(π₀/π₁), so
 * the error is π₁ Φ((t − Δ²/2)/Δ) + π₀ Φ((−t − Δ²/2)/Δ) (e.g. Duda, Hart and Stork, 2001, §2.8.3).
 */
export function twoGaussianBayesError(delta: number, priors: readonly number[]): number {
  const [p0, p1] = priors
  if (p0 === 0 || p1 === 0) return 0
  if (delta === 0) return Math.min(p0, p1)
  const t = Math.log(p0 / p1)
  return p1 * phi((t - (delta * delta) / 2) / delta) + p0 * phi((-t - (delta * delta) / 2) / delta)
}

/**
 * The log density of points uniform along a curve c(u), u uniform on [0, 1], blurred by isotropic Gaussian noise of
 * standard deviation `sd`: log (1/M) Σₘ N(x; c(uₘ), sd² I), a midpoint rule over M nodes. M is chosen so that nodes
 * are at most sd/2 apart along the curve, which keeps the rule accurate to well under a percent.
 */
export function curveLogDensity(curve: (u: number) => [number, number], sd: number): (x: Row) => number {
  let length = 0
  let prev = curve(0)
  for (let i = 1; i <= 256; i++) {
    const p = curve(i / 256)
    length += Math.hypot(p[0] - prev[0], p[1] - prev[1])
    prev = p
  }
  const m = Math.min(2000, Math.max(64, Math.ceil((2 * length) / sd)))
  const cx = new Float64Array(m)
  const cy = new Float64Array(m)
  for (let i = 0; i < m; i++) [cx[i], cy[i]] = curve((i + 0.5) / m)
  const terms = new Float64Array(m)
  const norm = -Math.log(m) - LOG_2PI - 2 * Math.log(sd)
  const k = -1 / (2 * sd * sd)
  return (x) => {
    for (let i = 0; i < m; i++) terms[i] = k * ((x[0] - cx[i]) ** 2 + (x[1] - cy[i]) ** 2)
    return norm + logSumExp(terms)
  }
}

/** Rows of a matrix stored row-major. */
export function rowsOf(data: Float64Array, n: number, d: number): Float64Array[] {
  return Array.from({ length: n }, (_, i) => data.slice(i * d, (i + 1) * d))
}

/** A reference with equal weights. */
export function equalReference(rows: Float64Array[]): Reference {
  return { rows, weights: new Float64Array(rows.length).fill(1 / rows.length) }
}

/** A regression truth with homoscedastic noise unless `sdAt` is given. */
export function regressionTruth(
  mean: (x: Row) => number,
  noiseSd: number,
  options: { sdAt?: (x: Row) => number; bayesRisk?: number } = {},
): RegressionTruth {
  return {
    kind: 'regression',
    mean,
    noiseSd,
    noiseSdAt: options.sdAt ?? (() => noiseSd),
    bayesRisk: options.bayesRisk ?? noiseSd * noiseSd,
    outlierFraction: 0,
  }
}
