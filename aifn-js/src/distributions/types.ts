/**
 * The distribution protocol: plain objects whose parameters are public fields and whose derived quantities are
 * methods. See `aifn-js/README.md` (Distributions) and `docs/aifn-plan.md` §5.2.
 *
 * Kinds. Parameters and arguments may be numbers, tensors or traced values. A result is a number when every input is a
 * number, a tensor when any is a tensor, and traced when any is traced (so log-densities are differentiable in both
 * the parameters and the value). `Kind<P>` computes that kind from the union of input types.
 */

import type { Stream } from 'aifn/random'
import type { Raw, Tensor, Traced, Value } from 'aifn/tensor'

/**
 * The kind of a result computed from inputs of types `P` (a union): `Value` when that is only known at run time,
 * `Traced` when any input is traced, a number when all are numbers, otherwise a tensor.
 */
export type Kind<P> = Value extends P
  ? Value
  : [Extract<P, Traced>] extends [never]
    ? [P] extends [number]
      ? number
      : Tensor
    : Traced

/**
 * The kind of a result that reduces an event axis (multivariate log-densities): a number for one event or a tensor for
 * a batch (known only at run time, so `Raw`), or traced when any input is traced.
 */
export type EventKind<P> = Value extends P ? Value : [Extract<P, Traced>] extends [never] ? Raw : Traced

/** The kind of a draw: a number when every parameter is a number, otherwise a tensor. */
export type SampleKind<P> = [P] extends [number] ? number : Tensor

/** Options of `sample`. */
export type SampleOptions = {
  /**
   * The sample shape. Draws have shape `[...shape, ...batchShape, ...eventShape]` (PyTorch's convention). Omitted:
   * one draw per batch member (a number for an unbatched univariate distribution).
   */
  shape?: readonly number[]
}

/**
 * Where a distribution puts its mass. `real` is the real line; `interval` is [lower, upper] (either end may be
 * infinite; `lower` and `upper` may be tensors for a batch); `integers` are the integers in [lower, upper]; `simplex`,
 * `real-vector`, `positive-definite` and `count-vector` (non-negative integer vectors summing to n) are event spaces
 * of multivariate distributions; `circle` is an interval of length 2π on which the density is periodic.
 */
export type Support =
  | { type: 'real' }
  /** Finite ends are closed unless marked open (as `Transformed` marks the ends of an image, e.g. (0, 1)). */
  | { type: 'interval'; lower: Value; upper: Value; lowerOpen?: boolean; upperOpen?: boolean }
  | { type: 'integers'; lower: Value; upper: Value }
  | { type: 'circle'; lower: Value; upper: Value }
  | { type: 'simplex' }
  | { type: 'real-vector' }
  | { type: 'positive-definite' }
  | { type: 'count-vector'; total: Value }

/**
 * Exponential-family structure: log p(x) = Σᵢ ηᵢ · Tᵢ(x) − A(η) + log h(x), where each ηᵢ · Tᵢ(x) is summed over
 * the event axes. Used by `pgm` and `ep` for conjugate updates and message passing.
 */
export interface ExponentialFamily {
  /** The natural parameters η, one entry per sufficient statistic, each with the batch (and event) shape. */
  naturalParams(): Value[]
  /** The sufficient statistics T(x), in the order of `naturalParams`. */
  sufficientStats(x: Value): Value[]
  /** The log-partition function A(η), with the batch shape. */
  logPartition(): Value
  /** log h(x), the base measure. */
  logBaseMeasure(x: Value): Value
}

/** What every distribution has. */
interface Common {
  /** The family's name, e.g. `Normal`; `kl` dispatches on it. */
  readonly name: string
  /** The parameters as given (after conversion to the documented parameterisation), by name. */
  readonly params: Readonly<Record<string, Value>>
  /** The shape of a batch of independent distributions (the parameters' broadcast shape). */
  readonly batchShape: readonly number[]
  /** The shape of one draw: `[]` for univariate distributions, `[d]` for vectors, `[d, d]` for matrices. */
  readonly eventShape: readonly number[]
  readonly support: Support
  /** True for distributions over integers (mass functions). */
  readonly discrete: boolean
  /** Exponential-family structure, for the members of that family. */
  readonly expFamily?: ExponentialFamily
  /**
   * Draws, from the stream only, of shape `[...shape, ...batchShape, ...eventShape]` (a number for an unbatched
   * univariate distribution without `shape`). Parameters must not be traced (reparameterise explicitly instead).
   */
  sample(s: Stream, options?: SampleOptions): number | Tensor
}

/**
 * A univariate distribution (event shape `[]`) of any parameter kind. Every function of a value broadcasts the value
 * against the batch. `prob` is the density (continuous) or the mass (discrete). Methods that have no closed form for
 * a family (the entropy of a mixture) throw, and undefined moments (the mean of a Cauchy) are NaN, as documented on
 * each constructor.
 */
export interface AnyUnivariate extends Common {
  readonly eventShape: readonly []
  logProb(x: Value): Value
  prob(x: Value): Value
  /** P(X ≤ x). */
  cdf(x: Value): Value
  logcdf(x: Value): Value
  /** P(X > x), computed directly where that is more accurate than 1 − cdf. */
  survival(x: Value): Value
  logSurvival(x: Value): Value
  /** The inverse cdf: the smallest x with cdf(x) ≥ p. */
  quantile(p: Value): Value
  mean(): Value
  variance(): Value
  stddev(): Value
  /** Differential entropy (continuous) or entropy (discrete), in nats. */
  entropy(): Value
  mode(): Value
}

/** A univariate distribution whose parameters have types `P`: results have the kinds `Kind` computes. */
export interface TypedUnivariate<P extends Value> extends AnyUnivariate {
  logProb<X extends Value>(x: X): Kind<P | X>
  prob<X extends Value>(x: X): Kind<P | X>
  cdf<X extends Value>(x: X): Kind<P | X>
  logcdf<X extends Value>(x: X): Kind<P | X>
  survival<X extends Value>(x: X): Kind<P | X>
  logSurvival<X extends Value>(x: X): Kind<P | X>
  quantile<X extends Value>(p: X): Kind<P | X>
  mean(): Kind<P>
  variance(): Kind<P>
  stddev(): Kind<P>
  entropy(): Kind<P>
  mode(): Kind<P>
  sample(s: Stream): SampleKind<P>
  sample(s: Stream, options: SampleOptions & { shape: readonly number[] }): Tensor
  sample(s: Stream, options?: SampleOptions): number | Tensor
}

/**
 * A univariate distribution. `Univariate` alone accepts any (its methods return `Value`); `Univariate<number>`, as
 * returned by `Normal(0, 1)`, returns numbers for numbers, tensors for tensors and traced values for traced ones.
 */
export type Univariate<P extends Value = Value> = Value extends P ? AnyUnivariate : TypedUnivariate<P>

/** A distribution over vectors or matrices, of any parameter kind. `logProb` reduces the event axes. */
export interface AnyMultivariate extends Common {
  logProb(x: Value): Value
  prob(x: Value): Value
  /** The mean, with shape `[...batchShape, ...eventShape]`. */
  mean(): Value
  /** The elementwise variance, with shape `[...batchShape, ...eventShape]`. */
  variance(): Value
  stddev(): Value
  /** The covariance matrix of a vector-valued distribution, `[...batchShape, d, d]`. */
  covariance(): Value
  entropy(): Value
  mode(): Value
}

/** A multivariate distribution whose parameters have types `P`. */
export interface TypedMultivariate<P extends Value> extends AnyMultivariate {
  logProb<X extends Value>(x: X): EventKind<P | X>
  prob<X extends Value>(x: X): EventKind<P | X>
  mean(): Kind<P | Tensor>
  variance(): Kind<P | Tensor>
  stddev(): Kind<P | Tensor>
  covariance(): Kind<P | Tensor>
  entropy(): EventKind<P>
  mode(): Kind<P | Tensor>
  sample(s: Stream, options?: SampleOptions): Tensor
}

/** A multivariate distribution; `Multivariate` alone accepts any. */
export type Multivariate<P extends Value = Value> = Value extends P ? AnyMultivariate : TypedMultivariate<P>

/** Any distribution. */
export type Distribution = AnyUnivariate | AnyMultivariate
