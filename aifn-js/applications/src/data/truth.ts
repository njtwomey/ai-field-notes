/**
 * Known truth for synthetic datasets. When the generating process has a closed form, a dataset carries it in
 * `meta.truth` as a model (`kind: 'model'`) with the contract's capabilities: the Bayes rule (`decide`), the Bayes
 * posterior or the conditional law of y (`predictive`), the regression function (`expect`), and the lowest risk any
 * predictor can reach (`bayesRisk`). A figure draws the Bayes-optimal boundary and curves beside a fitted model's by
 * calling the same methods on both.
 *
 * A classification truth is built from class-conditional densities p(x | y = j), evaluated with the registered
 * distributions of `aifn/probability/distributions`, and clean class priors πⱼ, followed by a list of label
 * operations: noise matrices T (T[i][j] = P(observed j | label i)) and class reweightings w (from resampling by
 * label). The joint of x and the observed label is then
 *
 *   p(x, ỹ) ∝ ((π ∘ p(x)) T₁ ∘ w₁ …)_ỹ,
 *
 * applied left to right, and the Bayes posterior is that vector normalised (Duda, Hart and Stork, 2001, "Pattern
 * Classification", §2.2). Modifiers compose by editing this model. Every method takes a batch of points, an [n, d]
 * float64 matrix.
 */

import type { Scores, Size, Truth as TruthContract } from 'aifn/foundation/contracts'
import { fromData, stack, type Tensor } from 'aifn/foundation/tensor'
import { solve } from 'aifn/numerics/linalg'
import { gaussHermite } from 'aifn/numerics/quadrature'
import { normalCdf } from 'aifn/numerics/special'
import { Bernoulli, Categorical, MultivariateNormal, Normal, type AnyUnivariate } from 'aifn/probability/distributions'

/** A point, one number per feature (used inside regression functions). */
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
 * Points drawn from the population's marginal of x ([m, d]) with importance weights summing to one, for Monte Carlo
 * estimates (the Bayes error when no closed form applies).
 */
export interface Reference {
  x: Tensor
  weights: Float64Array
}

/** The parts a classification truth is built from; modifiers edit these. */
export interface ClassModel {
  classes: Size
  /** Clean class proportions πⱼ in the population. */
  priors: number[]
  /**
   * log p(x | clean y = j) for every row of x ([n, d]) and class j: an [n, k] matrix, up to a term per row shared by
   * all classes. −∞ outside a class's support.
   */
  logDensity: (x: Tensor) => Tensor
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
 * The Bayes-optimal classifier of a synthetic classification problem, as a model. Methods take points ([n, d]) and
 * refer to the observed labels `y` of the dataset, after any label noise or resampling.
 */
export interface ClassificationTruth extends TruthContract, Scores<Tensor> {
  readonly task: 'classification'
  /** What the class densities are, e.g. "two moons". */
  readonly name: string
  readonly classes: Size
  /** Clean class proportions in the population. */
  readonly priors: readonly number[]
  /** Observed class proportions in the population. */
  readonly prevalence: readonly number[]
  /** log p(x | clean y = j) per row and class ([n, k]), up to a term shared by all classes. */
  logDensity(x: Tensor): Tensor
  /** P(clean y = j | x) per row and class ([n, k]). */
  cleanPosterior(x: Tensor): Tensor
  /** P(observed y = j | x) per row and class ([n, k]): the Bayes posterior. NaN where x has zero density under every class. */
  posterior(x: Tensor): Tensor
  /**
   * The Bayes posterior as a distribution over the batch: a Bernoulli over class 1 for two classes, else a
   * Categorical. Rows with zero density under every class get the uniform law.
   */
  predictive(x: Tensor): AnyUnivariate
  /** The Bayes rule: the most probable observed class per row (int32 [n]); −1 where the posterior is undefined. */
  decide(x: Tensor): Tensor
  /** E[f(ỹ) | x] under the posterior ([n]); the mean class index without `f` (P(ỹ = 1 | x) for two classes). */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** The log odds of class 1 against the rest ([n]): a Bayes-optimal score for ROC and precision–recall curves. */
  score(x: Tensor): Tensor
  /** The Bayes error E[1 − maxⱼ P(ỹ = j | x)], the lowest error any classifier can reach (computed on first access). */
  readonly bayesError: number
  /** How `bayesError` is computed: exactly, or by Monte Carlo over a reference sample of the population. */
  readonly bayesErrorMethod: 'closed form' | 'monte carlo'
  /** Standard error of a Monte Carlo `bayesError` (0 in closed form). */
  readonly bayesErrorSe: number
  /** The model the truth is built from (for modifiers). */
  readonly model: ClassModel
}

/** The parts a regression truth is built from: y = m(x) + ε, ε ~ N(0, σ(x)²). Modifiers edit these. */
export interface RegressionModel {
  /** The regression function m(x) = E[y | x] at one point. */
  mean: (x: Row) => number
  /** σ(x), the noise standard deviation at one point. */
  sdAt: (x: Row) => number
  /** The noise standard deviation (at the left end of the input range when it varies). */
  noiseSd: number
  /** The Bayes risk under squared loss, E[σ(x)²]. */
  bayesRisk: number
  /** Fraction of targets replaced by gross outliers (see `withOutliers`), 0 by default. */
  outlierFraction: number
  /** What the regression function is, for captions. */
  family: string
}

/** The truth of a synthetic regression problem y = m(x) + ε, ε ~ N(0, σ(x)²), as a model. */
export interface RegressionTruth extends TruthContract {
  readonly task: 'regression'
  readonly name: string
  /** The regression function m(x) per row ([n]); the same as `expect`. */
  mean(x: Tensor): Tensor
  /** σ(x) per row ([n]). */
  noiseSdAt(x: Tensor): Tensor
  /** The conditional law of y: Normal(m(x), σ(x)) over the batch. Outliers (if any) are not part of it. */
  predictive(x: Tensor): AnyUnivariate
  /** The point prediction that minimises squared loss: m(x) ([n]). */
  decide(x: Tensor): Tensor
  /** E[f(y) | x] ([n]); m(x) without `f`, else by 32-point Gauss–Hermite quadrature. */
  expect(x: Tensor, f?: (y: number) => number): Tensor
  /** The noise standard deviation (at the left end of the input range when it varies). */
  readonly noiseSd: number
  /** Fraction of targets replaced by gross outliers. */
  readonly outlierFraction: number
  readonly model: RegressionModel
}

export type Truth = ClassificationTruth | RegressionTruth

/** Number of reference points drawn for Monte Carlo Bayes errors. */
export const REFERENCE_SIZE = 6000

// ── Batches ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The rows of an [n, d] matrix as a fresh row-major Float64Array, with n and d. */
export function points(x: Tensor): { data: Float64Array; n: Size; d: Size } {
  if (x.shape.length !== 2) throw new RangeError(`truth: points must be an [n, d] matrix, got rank ${x.shape.length}`)
  const [n, d] = x.shape
  const data = new Float64Array(n * d)
  const [s0, s1] = x.strides
  const src = x.data
  for (let i = 0; i < n; i++) for (let c = 0; c < d; c++) data[i * d + c] = Number(src[x.offset + i * s0 + c * s1])
  return { data, n, d }
}

/** An [n, d] float64 matrix from row-major data. */
export function pointsFrom(data: Float64Array, n: Size, d: Size): Tensor {
  return fromData(data, [n, d])
}

/** Row-major values of a tensor (any shape) as a fresh Float64Array. */
function flat(t: Tensor): Float64Array {
  if (t.shape.length === 2) return points(t).data
  if (t.shape.length === 1)
    return Float64Array.from({ length: t.shape[0] }, (_, i) => Number(t.data[t.offset + i * t.strides[0]]))
  return Float64Array.of(Number(t.data[t.offset]))
}

/** Per-row log densities of k classes (each [n]) as one [n, k] matrix. */
export function classColumns(columns: readonly Tensor[]): Tensor {
  return stack(columns, 1)
}

function lazy<T>(f: () => T): () => T {
  let cached: { value: T } | undefined
  return () => (cached ??= { value: f() }).value
}

/** log Σ exp over a short list (a row of class scores). */
function logSumExpRow(a: ArrayLike<number>, from = 0, to = a.length): number {
  let m = -Infinity
  for (let i = from; i < to; i++) if (a[i] > m) m = a[i]
  if (m === -Infinity) return -Infinity
  let s = 0
  for (let i = from; i < to; i++) s += Math.exp(a[i] - m)
  return m + Math.log(s)
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

/** The clean log joint log πⱼ + log p(x | j), [n × k] row-major. */
function logJoint(model: ClassModel, x: Tensor): { a: Float64Array; n: Size; k: Size } {
  const ld = model.logDensity(x)
  const k = model.classes
  const a = flat(ld)
  const n = a.length / k
  const logPrior = model.priors.map(Math.log)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) a[i * k + j] += logPrior[j]
  return { a, n, k }
}

/** The observed-label posterior, [n × k] row-major (NaN rows where every class has zero density). */
function posteriorData(model: ClassModel, x: Tensor): { p: Float64Array; n: Size; k: Size } {
  const { a, n, k } = logJoint(model, x)
  const kOut = model.ops.reduce((c, op) => (op.kind === 'noise' ? op.matrix[0].length : c), k)
  const p = new Float64Array(n * kOut)
  for (let i = 0; i < n; i++) {
    const row = Array.from(a.subarray(i * k, (i + 1) * k))
    const m = Math.max(...row)
    if (!(m > -Infinity)) {
      p.fill(NaN, i * kOut, (i + 1) * kOut)
      continue
    }
    p.set(
      normalise(
        applyOps(
          row.map((v) => Math.exp(v - m)),
          model.ops,
        ),
      ),
      i * kOut,
    )
  }
  return { p, n, k: kOut }
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

/** Build the truth model from a class model. */
export function classificationTruth(model: ClassModel): ClassificationTruth {
  const monteCarlo = lazy(() => {
    const { x, weights } = model.reference()
    const { p, n, k } = posteriorData(model, x)
    let mean = 0
    const errs = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const e = 1 - Math.max(...p.subarray(i * k, (i + 1) * k))
      errs[i] = Number.isNaN(e) ? 0 : e
      mean += weights[i] * errs[i]
    }
    // Standard error of a self-normalised weighted mean: sqrt(Σ wᵢ² (eᵢ − ē)²).
    let v = 0
    errs.forEach((e, i) => (v += weights[i] * weights[i] * (e - mean) ** 2))
    return { error: mean, se: Math.sqrt(v) }
  })
  const exact = lazy(() => closedFormError(model))
  const prevalence = lazy(() => {
    if (!model.shifted) return normalise(applyOps([...model.priors], model.ops))
    const { x, weights } = model.reference()
    const { p, n, k } = posteriorData(model, x)
    const out = new Array<number>(k).fill(0)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < k; j++) out[j] += weights[i] * (Number.isNaN(p[i * k + j]) ? 0 : p[i * k + j])
    return normalise(out)
  })
  const posterior = (x: Tensor) => {
    const { p, n, k } = posteriorData(model, x)
    return fromData(p, [n, k])
  }
  return {
    kind: 'model',
    task: 'classification',
    name: model.family,
    classes: model.classes,
    priors: model.priors,
    get prevalence() {
      return prevalence()
    },
    logDensity: model.logDensity,
    cleanPosterior: (x) => {
      const { a, n, k } = logJoint(model, x)
      for (let i = 0; i < n; i++) {
        const z = logSumExpRow(a, i * k, (i + 1) * k)
        for (let j = 0; j < k; j++) a[i * k + j] = Math.exp(a[i * k + j] - z)
      }
      return fromData(a, [n, k])
    },
    posterior,
    predictive: (x) => {
      const { p, n, k } = posteriorData(model, x)
      for (let i = 0; i < n; i++) if (Number.isNaN(p[i * k])) p.fill(1 / k, i * k, (i + 1) * k)
      if (k === 2)
        return Bernoulli(
          fromData(
            Float64Array.from({ length: n }, (_, i) => p[2 * i + 1]),
            [n],
          ),
        )
      return Categorical(fromData(p, [n, k]))
    },
    decide: (x) => {
      const { p, n, k } = posteriorData(model, x)
      const out = new Int32Array(n)
      for (let i = 0; i < n; i++) {
        let best = -1
        for (let j = 0; j < k; j++)
          if (!Number.isNaN(p[i * k + j]) && (best < 0 || p[i * k + j] > p[i * k + best])) best = j
        out[i] = best
      }
      return fromData(out, [n])
    },
    expect: (x, f = (y) => y) => {
      const { p, n, k } = posteriorData(model, x)
      const fs = Array.from({ length: k }, (_, j) => f(j))
      const out = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[i] += p[i * k + j] * fs[j]
      return fromData(out, [n])
    },
    score: (x) => {
      const { a, n, k } = logJoint(model, x)
      const out = new Float64Array(n)
      if (model.ops.length === 0) {
        for (let i = 0; i < n; i++) {
          const row = Array.from(a.subarray(i * k, (i + 1) * k))
          out[i] = row[1] - logSumExpRow(row.filter((_, j) => j !== 1))
        }
      } else {
        const { p, k: kOut } = posteriorData(model, x)
        for (let i = 0; i < n; i++) out[i] = Math.log(p[i * kOut + 1]) - Math.log(1 - p[i * kOut + 1])
      }
      return fromData(out, [n])
    },
    get bayesError() {
      return exact() ?? monteCarlo().error
    },
    get bayesRisk() {
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

// ── Class-conditional densities, from the registered distributions ──────────────────────────────────────────────────

/** Φ(z), the standard normal cdf. */
export function phi(z: number): number {
  return normalCdf(z) as number
}

const matrixOf = (rows: readonly (readonly number[])[]): Tensor =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])
const vectorOf = (v: readonly number[]): Tensor => fromData(Float64Array.from(v), [v.length])

/** The Gaussian classes N(meanⱼ, Σⱼ): log p(x | j) for every row, [n, k]. Throws when a covariance is not positive definite. */
export function gaussianClasses(
  means: readonly (readonly number[])[],
  covariances: readonly (readonly (readonly number[])[])[],
): (x: Tensor) => Tensor {
  const laws = means.map((m, j) => MultivariateNormal(vectorOf(m), { covariance: matrixOf(covariances[j]) }))
  return (x) => classColumns(laws.map((law) => law.logProb(x) as Tensor))
}

/** Squared Mahalanobis distance (a − b)ᵀ Σ⁻¹ (a − b). */
export function mahalanobis2(a: readonly number[], b: readonly number[], covariance: readonly (readonly number[])[]) {
  const diff = a.map((v, i) => v - b[i])
  const z = flat(solve(matrixOf(covariance), vectorOf(diff)))
  return diff.reduce((s, v, i) => s + v * z[i], 0)
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

/** Rows evaluated per block against the nodes of a curve density, so a block holds at most ~2¹⁸ node terms. */
const BLOCK_TERMS = 1 << 18

/**
 * The log density of points uniform along a curve c(u), u uniform on [0, 1], blurred by isotropic Gaussian noise of
 * standard deviation `sd`: log (1/M) Σₘ N(x; c(uₘ), sd² I), a midpoint rule over M nodes, evaluated as a batch of
 * bivariate normals. M is chosen so that nodes are at most sd/2 apart along the curve, which keeps the rule accurate
 * to well under a percent. Returns the density of every row of x ([n]).
 */
export function curveLogDensity(curve: (u: number) => [number, number], sd: number): (x: Tensor) => Tensor {
  let length = 0
  let prev = curve(0)
  for (let i = 1; i <= 256; i++) {
    const p = curve(i / 256)
    length += Math.hypot(p[0] - prev[0], p[1] - prev[1])
    prev = p
  }
  const m = Math.min(2000, Math.max(64, Math.ceil((2 * length) / sd)))
  const nodes = new Float64Array(2 * m)
  for (let i = 0; i < m; i++) [nodes[2 * i], nodes[2 * i + 1]] = curve((i + 0.5) / m)
  const law = MultivariateNormal(fromData(nodes, [m, 2]), {
    covariance: fromData(Float64Array.of(sd * sd, 0, 0, sd * sd), [2, 2]),
  })
  const logM = Math.log(m)
  return (x) => {
    const { data, n } = points(x)
    const out = new Float64Array(n)
    const block = Math.max(1, Math.floor(BLOCK_TERMS / m))
    for (let start = 0; start < n; start += block) {
      const rows = Math.min(block, n - start)
      const terms = flat(law.logProb(fromData(data.slice(2 * start, 2 * (start + rows)), [rows, 1, 2])) as Tensor)
      for (let i = 0; i < rows; i++) out[start + i] = logSumExpRow(terms, i * m, (i + 1) * m) - logM
    }
    return fromData(out, [n])
  }
}

/** A reference sample with equal weights. */
export function equalReference(x: Tensor): Reference {
  const n = x.shape[0]
  return { x, weights: new Float64Array(n).fill(1 / n) }
}

// ── Regression ───────────────────────────────────────────────────────────────────────────────────────────────────────

const HERMITE = lazy(() => {
  const rule = gaussHermite(32, { probabilists: true })
  const w = flat(rule.weights)
  const total = w.reduce((a, b) => a + b, 0)
  return { nodes: flat(rule.nodes), weights: w.map((v) => v / total) }
})

/** Build the truth model from a regression model. */
export function regressionTruthOf(model: RegressionModel): RegressionTruth {
  const perRow = (x: Tensor, f: (row: Float64Array) => number) => {
    const { data, n, d } = points(x)
    return fromData(
      Float64Array.from({ length: n }, (_, i) => f(data.subarray(i * d, (i + 1) * d))),
      [n],
    )
  }
  const mean = (x: Tensor) => perRow(x, model.mean)
  return {
    kind: 'model',
    task: 'regression',
    name: model.family,
    mean,
    noiseSdAt: (x) => perRow(x, model.sdAt),
    predictive: (x) => Normal(mean(x), perRow(x, model.sdAt)),
    decide: mean,
    expect: (x, f) => {
      if (!f) return mean(x)
      const { nodes, weights } = HERMITE()
      return perRow(x, (row) => {
        const [m, s] = [model.mean(row), model.sdAt(row)]
        return nodes.reduce((acc, z, q) => acc + weights[q] * f(m + s * z), 0)
      })
    },
    noiseSd: model.noiseSd,
    bayesRisk: model.bayesRisk,
    outlierFraction: model.outlierFraction,
    model,
  }
}

/** A regression truth with homoscedastic noise unless `sdAt` is given. */
export function regressionTruth(
  mean: (x: Row) => number,
  noiseSd: number,
  options: { sdAt?: (x: Row) => number; bayesRisk?: number; family?: string } = {},
): RegressionTruth {
  return regressionTruthOf({
    mean,
    sdAt: options.sdAt ?? (() => noiseSd),
    noiseSd,
    bayesRisk: options.bayesRisk ?? noiseSd * noiseSd,
    outlierFraction: 0,
    family: options.family ?? 'regression function',
  })
}

/** A regression truth with some parts of its model replaced. */
export function remodelRegression(t: RegressionTruth, edit: Partial<RegressionModel>): RegressionTruth {
  return regressionTruthOf({ ...t.model, ...edit })
}
