/**
 * The distribution contract predictive outputs satisfy, helpers that read class probabilities and expectations from
 * any distribution, and the predictive constructors the reference estimators use (on `aifn/distributions`).
 *
 * `Distribution` here is a minimal structural subset of `aifn/distributions`' protocol (`name`, `batchShape`,
 * `eventShape`, `logProb`, `sample(s, { shape })`, `mean()`, and `cdf`/`quantile`/`variance()` for univariate laws),
 * so that its objects and small hand-made predictives (e.g. pushed-forward ones in `aifn/compose`) both satisfy it.
 */

import { eigh } from 'aifn/linalg'
import { Bernoulli, Categorical, Normal, type Univariate } from 'aifn/distributions'
import type { Stream } from 'aifn/random'
import { normalCdf } from 'aifn/special'
import { fromData, isTensor, type Tensor, type Value } from 'aifn/tensor'
import { sizeOf, values } from './util'

/**
 * The minimal distribution contract: a subset of `aifn/distributions`' protocol. Results are `Value`s (numbers,
 * tensors or traced values); `asTensor` reads an untraced result as a tensor.
 */
export interface Distribution {
  /** The family's name, e.g. `Normal`. */
  readonly name: string
  /** Shape of the batch of independent distributions, e.g. [N] for one per input row. */
  readonly batchShape: readonly number[]
  /** Shape of one draw; [] for a scalar outcome. */
  readonly eventShape: readonly number[]
  /** log p(x), elementwise over the batch (x broadcasts against it). */
  logProb(x: Tensor): Value
  /** Draws of shape [...shape, ...batchShape, ...eventShape]. */
  sample(s: Stream, options: { shape: readonly number[] }): Tensor
  /** The mean, shape [...batchShape, ...eventShape]. */
  mean(): Value
}

/** A scalar-valued distribution with a CDF, a quantile function and a variance. */
export interface UnivariateDistribution extends Distribution {
  cdf(x: Tensor): Value
  quantile(p: Tensor): Value
  variance(): Value
}

/** A distribution over classes 0 … K−1: `aifn/distributions`' Bernoulli (K = 2) or Categorical. */
export interface ClassDistribution extends Distribution {
  readonly name: 'Bernoulli' | 'Categorical'
}

/** An untraced `Value` as a tensor (a number becomes a scalar tensor). */
export function asTensor(v: Value): Tensor {
  if (typeof v === 'number') return fromData(Float64Array.of(v), [])
  if (isTensor(v)) return v
  throw new Error('asTensor: a traced value; evaluate distributions outside a gradient tape here')
}

/** The mean of a distribution as a tensor. */
export function meanOf(d: Distribution): Tensor {
  return asTensor(d.mean())
}

/** The variance of a distribution as a tensor; throws when it has none. */
export function varianceOf(d: Distribution): Tensor {
  const v = (d as Partial<UnivariateDistribution>).variance
  if (typeof v !== 'function') throw new Error(`varianceOf: ${d.name} has no variance`)
  return asTensor(v.call(d))
}

/** True when `d` has a CDF and a quantile function and a scalar event. */
export function isUnivariate(d: Distribution): d is UnivariateDistribution {
  const u = d as Partial<UnivariateDistribution>
  return d.eventShape.length === 0 && typeof u.cdf === 'function' && typeof u.quantile === 'function'
}

/** True when `d` is a Bernoulli or categorical law (by name). */
export function isClassDistribution(d: Distribution): d is ClassDistribution {
  return d.name === 'Bernoulli' || d.name === 'Categorical'
}

/**
 * Class probabilities as an [N, K] matrix (flattened, row-major) from a Bernoulli (K = 2: [1 − p, p]) or categorical
 * distribution over a batch of N, read through `logProb` so that either parameterisation (probabilities or logits)
 * works. Throws for any other distribution.
 */
export function classProbabilities(d: Distribution): { probs: Float64Array; n: number; k: number } {
  if (!isClassDistribution(d)) throw new Error(`classProbabilities: ${d.name} has no class probabilities`)
  const n = sizeOf(d.batchShape)
  const upper = (d as { support?: { upper?: unknown } }).support?.upper
  const k = d.name === 'Bernoulli' ? 2 : typeof upper === 'number' ? upper + 1 : NaN
  if (!Number.isInteger(k)) throw new Error('classProbabilities: unknown number of classes')
  const out = new Float64Array(n * k)
  for (let c = 0; c < k; c++) {
    const lp = values(asTensor(d.logProb(fromData(Float64Array.of(c), []))))
    for (let i = 0; i < n; i++) out[i * k + c] = Math.exp(lp.length === 1 ? lp[0] : lp[i])
  }
  return { probs: out, n, k }
}

// ── Predictive constructors ──────────────────────────────────────────────────────────────────────────────────────

/** A batch of Gaussians N(loc, scale²) from `aifn/distributions` (params `{ loc, scale }`). */
export type GaussianPredictive = Univariate<Tensor>

/** A batch of Bernoulli laws from `aifn/distributions` (params `{ probs }`). */
export type BernoulliPredictive = Univariate<Tensor>

/** A batch of categorical laws from `aifn/distributions` (params `{ probs }`, [..., K]). */
export type CategoricalPredictive = Univariate<Tensor>

/** A batch of Gaussians with means `loc` and standard deviations `scale` (same shape): `Normal(loc, scale)`. */
export function gaussianPredictive(loc: Tensor, scale: Tensor): GaussianPredictive {
  return Normal(loc, scale)
}

/** Bernoulli laws with P(y = 1) = `probs`: `Bernoulli(probs)`. */
export function bernoulliPredictive(probs: Tensor): BernoulliPredictive {
  return Bernoulli(probs)
}

/** Categorical laws with class probabilities `probs` [..., K]: `Categorical(probs)`. */
export function categoricalPredictive(probs: Tensor): CategoricalPredictive {
  return Categorical(probs)
}

// ── Expectations ─────────────────────────────────────────────────────────────────────────────────────────────────────

let hermite: { nodes: Float64Array; weights: Float64Array } | null = null

/**
 * Probabilists' Gauss–Hermite rule with 32 nodes, for E[g(Z)] with Z ~ N(0, 1), by the Golub–Welsch algorithm
 * (Golub and Welsch, 1969, "Calculation of Gauss quadrature rules", Math. Comp. 23): the nodes are the eigenvalues of
 * the Jacobi matrix of the Hermite recurrence (off-diagonal √k) and the weights the squared first components of its
 * eigenvectors.
 */
export function gaussHermite(): { nodes: Float64Array; weights: Float64Array } {
  if (hermite) return hermite
  const n = 32
  const J = new Float64Array(n * n)
  for (let k = 1; k < n; k++) J[k * n + (k - 1)] = J[(k - 1) * n + k] = Math.sqrt(k)
  const { values: lambda, vectors } = eigh(fromData(J, [n, n]))
  const nodes = values(lambda).slice()
  const v = values(vectors)
  const weights = Float64Array.from({ length: n }, (_, j) => v[j] * v[j])
  hermite = { nodes, weights }
  return hermite
}

/**
 * E[f(y)] under `d`, elementwise over its batch (shape `batchShape`). Without `f`, the mean. For class distributions,
 * the finite sum Σₖ f(k) pₖ. For other univariate laws, 32-point Gauss–Hermite quadrature on normal scores:
 * E f(y) = E f(Q(Φ(Z))) with Z ~ N(0, 1) and Q the quantile function, exact for a Gaussian and f a polynomial of degree
 * < 64 up to the skipped outermost nodes (|z| > 8.1, total weight below 1e-15).
 */
export function expectation(d: Distribution, f?: (y: number) => number): Tensor {
  if (!f) return meanOf(d)
  if (isClassDistribution(d)) {
    const { probs, n, k } = classProbabilities(d)
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) out[i] += f(c) * probs[i * k + c]
    return fromData(out, d.batchShape)
  }
  if (!isUnivariate(d)) throw new Error('expectation: needs a class distribution or a univariate one with a quantile')
  const { nodes, weights } = gaussHermite()
  const n = sizeOf(d.batchShape)
  const out = new Float64Array(n)
  for (let j = 0; j < nodes.length; j++) {
    // Skip the outermost nodes, where Φ(z) or 1 − Φ(z) is below the float64 resolution of 1 (their weights are
    // below 1e-17), so that quantiles are never asked for at 0 or 1.
    if (normalCdf(-Math.abs(nodes[j])) < Number.EPSILON) continue
    const p = normalCdf(nodes[j])
    const q = values(asTensor(d.quantile(fromData(Float64Array.of(p), []))))
    for (let i = 0; i < n; i++) out[i] += weights[j] * f(q.length === 1 ? q[0] : q[i])
  }
  return fromData(out, d.batchShape)
}
