/**
 * The shapes every dataset shares, and small private helpers for building them. Datasets are plain data: tensors plus
 * a description used in captions.
 */

import { copy, fromData, isTensor, type Tensor } from 'aifn/tensor'
import type { Truth } from './truth'

/** One step of how a dataset was made: the generator or modifier (`op`) and its parameters. */
export interface RecipeStep {
  op: string
  params: Record<string, unknown>
}

/** What a dataset is and where it came from. */
export interface DatasetMeta {
  /** A short name, e.g. `moons`. */
  name: string
  /** One or two sentences for a caption: how the data were made and what the labels mean. */
  description: string
  /** What the dataset is for. */
  task: 'classification' | 'regression' | 'clustering' | 'manifold' | 'sequence' | 'images' | 'recommendation'
  /** One name per column of `x`. */
  featureNames: string[]
  /** One name per class, for integer labels `y`. */
  labelNames?: string[]
  /** The name of a real-valued target `y`. */
  targetName?: string
  /** A citation for real data or for the generator's recipe. */
  source?: string
  url?: string
  /** The key of the stream the data were drawn from (synthetic data only). */
  stream?: string
  /**
   * The known generating process, where it has a closed form: the Bayes posterior and error for classification, the
   * regression function and noise sd for regression. Modifiers keep it consistent (label noise, prevalence, shift).
   */
  truth?: Truth
  /** The steps that made the dataset, generator first, each modifier appending itself. */
  recipe?: RecipeStep[]
  /** Labels before `withLabelNoise` (int32, length n); `y` holds the observed, possibly flipped, labels. */
  cleanLabels?: Tensor
  /** 1 for rows replaced by `withOutliers` (int32, length n). */
  outliers?: Tensor
  /** 1 where `withMissing` removed an entry (int32, n × d); those entries of `x` are NaN. */
  missing?: Tensor
  /** `x` before `withMissing`, for drawing where the missing values were. */
  complete?: Tensor
}

/**
 * A dataset: features `x` (n × d, float64), optional labels or targets `y` (length n: int32 class indices, or float64
 * targets), and metadata. Generators may add named extras, e.g. `t`, a continuous coordinate along a manifold for
 * colouring, or `f`, the noise-free regression function at `x`.
 */
export interface Dataset {
  x: Tensor
  y?: Tensor
  /** A continuous coordinate per point (position along a roll or spiral), for colouring. */
  t?: Tensor
  /** The noise-free target at each point (regression). */
  f?: Tensor
  meta: DatasetMeta
}

/** A float64 matrix [rows, cols] from a row-major array. */
export function matrix(data: Float64Array, rows: number, cols: number): Tensor {
  return fromData(data, [rows, cols])
}

/** A float64 vector. */
export function vector(data: ArrayLike<number>): Tensor {
  return fromData(Float64Array.from(data))
}

/** An int32 vector. */
export function labels(data: ArrayLike<number>): Tensor {
  return fromData(Int32Array.from(data))
}

/** Values of a tensor or array as a fresh Float64Array (row-major). */
export function values(x: Tensor | ArrayLike<number>): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/** Split `n` into `k` near-equal group sizes (the first `n mod k` groups get one extra), as scikit-learn does. */
export function splitSizes(n: number, k: number): number[] {
  const base = Math.floor(n / k)
  return Array.from({ length: k }, (_, j) => base + (j < n % k ? 1 : 0))
}

export function checkCount(n: number, what: string): void {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`${what}: n must be a non-negative integer, got ${n}`)
}
