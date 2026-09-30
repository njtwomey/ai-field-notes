/**
 * The shared layer of `aifn/losses`: the `Loss` type and its metadata, the reduction over examples, and the
 * conversions that read targets (labels, relevance grades) into constant tensors. Private to the module except for the
 * types, `defineLoss`, `isLoss` and `oneHot`, which `index.ts` re-exports.
 */

import {
  fromData,
  isTensor,
  isTraced,
  mean,
  shapeOfValue,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn/tensor'

// ── Loss metadata ────────────────────────────────────────────────────────────────────────────────────────────────────

/** The family a loss belongs to, as the site's notes group them. */
export type LossFamily = 'classification' | 'regression' | 'ranking' | 'retrieval' | 'representation' | 'divergence'

/**
 * What a loss reads from a model, so a caller can feed it the right output:
 *
 * - `logits`: unnormalised log-odds or class scores (before a sigmoid or softmax).
 * - `probabilities`: predicted probabilities.
 * - `margins`: real-valued scores f(x) paired with labels y ∈ {−1, +1} (the margin is y·f(x)).
 * - `values`: point predictions of a real target; `distribution`: predictive distribution parameters (mean and scale,
 *   a rate).
 * - `scores`: scores of the items of a list, with relevance grades.
 * - `embeddings`: vectors compared by a similarity or a distance.
 * - `distributions`: two distribution objects or probability vectors.
 */
export type LossInput =
  'logits' | 'probabilities' | 'margins' | 'values' | 'distribution' | 'scores' | 'embeddings' | 'distributions'

/** Metadata that lets a loss be used generically: listed, labelled and fed the right input. */
export type LossInfo = {
  /** The export name, e.g. `huber`. Unique across the module. */
  readonly key: string
  /** A display name, e.g. "Huber loss". */
  readonly name: string
  readonly family: LossFamily
  /** What the loss reads (see `LossInput`). */
  readonly inputs: LossInput
  /** The slug of the site note that defines it. */
  readonly note: string
  /** What its population minimiser estimates or what it optimises, in a few words, e.g. "the conditional median". */
  readonly target?: string
}

/** Any function returning a value (a `never[]` parameter list accepts every signature). */
export type LossFunction = (...args: never[]) => Value

/** A loss: a differentiable function of predictions returning a number (or per-example values), with `info`. */
export type Loss<F extends LossFunction = LossFunction> = F & { readonly info: LossInfo }

/** Attach metadata to a loss function (the function itself is returned, with `info` added). */
export function defineLoss<F extends LossFunction>(info: LossInfo, f: F): Loss<F> {
  return Object.assign(f, { info })
}

/** True when `x` is a loss defined with `defineLoss`. */
export function isLoss(x: unknown): x is Loss {
  return (
    typeof x === 'function' &&
    'info' in x &&
    typeof (x as { info?: { family?: unknown } }).info === 'object' &&
    typeof (x as { info: { family?: unknown } }).info.family === 'string'
  )
}

// ── Reduction ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * How per-example losses are combined: `mean` (the default, as in PyTorch), `sum`, or `none` (the per-example values,
 * one per example or list).
 */
export type Reduction = 'mean' | 'sum' | 'none'

/** Options every loss takes. */
export type ReductionOptions = { reduction?: Reduction }

/** Combine per-example losses. A number (no examples axis) is returned as it is. */
export function reduce(v: Value, reduction: Reduction = 'mean'): Value {
  if (reduction === 'none' || shapeOfValue(v).length === 0) return v
  return reduction === 'sum' ? sum(v) : mean(v)
}

// ── Constant inputs ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Targets, labels or grades: a number, an array of numbers, or a tensor (any dtype). Never differentiated. */
export type Target = number | ArrayLike<number> | Tensor

/** A target as a float64 constant (a number stays a number). Traced targets are read as constants. */
export function constant(t: Target | Value): number | Tensor {
  if (typeof t === 'number') return t
  if (isTraced(t)) return unwrap(t)
  if (isTensor(t)) return t.dtype === 'float64' ? t : fromData(Float64Array.from(t.data), t.shape)
  return fromData(Float64Array.from(t as ArrayLike<number>), [(t as ArrayLike<number>).length])
}

/** A target's values as a flat Float64Array (row-major). */
export function flatValues(t: Target | Value): Float64Array {
  const c = constant(t)
  return typeof c === 'number' ? new Float64Array([c]) : Float64Array.from(toFlat(c))
}

/**
 * One-hot rows for integer class labels: a float64 tensor of shape [...labels shape, K] with 1 at each label's class.
 * Labels must be integers in [0, K).
 *
 * @example oneHot([2, 0], 3) // [[0, 0, 1], [1, 0, 0]]
 */
export function oneHot(labels: Target, K: number): Tensor {
  const c = constant(labels)
  const shape = typeof c === 'number' ? [] : c.shape
  const ys = flatValues(labels)
  const out = new Float64Array(ys.length * K)
  ys.forEach((y, i) => {
    if (!Number.isInteger(y) || y < 0 || y >= K) throw new RangeError(`oneHot: label ${y} is not a class in [0, ${K})`)
    out[i * K + y] = 1
  })
  return fromData(out, [...shape, K])
}

/** Throw unless a value has the expected rank(s). */
export function expectRank(x: Value, ranks: readonly number[], what: string): number[] {
  const shape = shapeOfValue(x)
  if (!ranks.includes(shape.length)) {
    throw new Error(`losses: ${what} needs rank ${ranks.join(' or ')}, got shape [${shape.join(', ')}]`)
  }
  return shape
}
