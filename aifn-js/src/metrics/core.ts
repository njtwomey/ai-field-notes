/**
 * The shared layer of `aifn/metrics`: the `Metric` type and its metadata, and the conversions every metric uses to
 * read its inputs (numbers, labels, rows) and to return tensors. Private to the module except for the types and
 * `defineMetric`, which `index.ts` re-exports.
 */

import { copy, fromData, isTensor, type Tensor } from 'aifn/tensor'

// ── Metric metadata ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a metric reads from a model or a prediction, so that a caller (the lab, `estimators.evaluate`) can pick the
 * right output to feed it:
 *
 * - `labels`: predicted classes (a decision); `sets`: predicted label sets (multi-label, as 0/1 rows).
 * - `scores`: real-valued scores whose order matters (a ranking of cases), such as a margin or a probability.
 * - `probabilities`: predicted class probabilities (a vector of P(positive) or an n × K matrix).
 * - `distribution`: a predictive distribution (Gaussian parameters, samples, quantiles or an interval).
 * - `values`: point predictions of a real target (regression and forecasting).
 * - `ranking`: relevance grades of a ranked list, or relevance and scores to rank by.
 * - `partitions`: two cluster labellings of the same items; `features`: data vectors (internal clustering indices,
 *   generative-model features, distances).
 * - `ratings`: several raters' categories or scores for the same items (agreement), or a contingency table.
 * - `boxes`, `masks`, `points`: detection and segmentation geometry.
 * - `sequences`: text (strings or tokens); `images`: pixel arrays; `signals`: waveforms; `vectors`: embeddings.
 * - `exposure`: counts or popularities of items shown to users (beyond-accuracy recommendation metrics).
 */
export type InputKind =
  | 'labels'
  | 'sets'
  | 'scores'
  | 'probabilities'
  | 'distribution'
  | 'values'
  | 'ranking'
  | 'partitions'
  | 'features'
  | 'ratings'
  | 'boxes'
  | 'masks'
  | 'points'
  | 'sequences'
  | 'images'
  | 'signals'
  | 'vectors'
  | 'exposure'

/**
 * The model capability a metric needs, in the vocabulary of `estimators` (plan §5.3): `decide` (hard predictions),
 * `score` (an ordering of cases) or `predictive` (a predictive distribution or probabilities).
 */
export type Capability = 'decide' | 'score' | 'predictive'

/** Metadata that lets a metric be used generically: listed, labelled, fed the right input and read the right way. */
export type MetricInfo = {
  /** The export name, e.g. `auroc`. Unique across the module. */
  readonly key: string
  /** A display name, e.g. "Area under the ROC curve". */
  readonly name: string
  /** What the metric reads (see `InputKind`). */
  readonly inputs: InputKind
  /** Whether larger or smaller values are better. */
  readonly direction: 'higher' | 'lower'
  /** The range of attainable values; ±Infinity where unbounded. */
  readonly range: readonly [number, number]
  /** The slug of the site note that defines it. */
  readonly note: string
  /** The model capability it needs, where it evaluates a model's predictions. */
  readonly capability?: Capability
}

/** Any function returning a number (a `never[]` parameter list accepts every signature). */
export type MetricFunction = (...args: never[]) => number

/** A metric: a function returning a number, carrying its metadata as `info`. */
export type Metric<F extends MetricFunction = MetricFunction> = F & { readonly info: MetricInfo }

/** Attach metadata to a metric function (the function itself is returned, with `info` added). */
export function defineMetric<F extends MetricFunction>(info: MetricInfo, f: F): Metric<F> {
  return Object.assign(f, { info })
}

/** True when `x` is a metric defined with `defineMetric`. */
export function isMetric(x: unknown): x is Metric {
  return typeof x === 'function' && 'info' in x && typeof (x as { info?: unknown }).info === 'object'
}

// ── Numeric inputs ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Numeric data a metric accepts: a `number[]`, a typed array, or a tensor (any rank is flattened row-major). */
export type Data = ArrayLike<number> | Tensor

/** A matrix given as a rank-2 tensor or as rows of numbers. */
export type Rows = Tensor | ArrayLike<ArrayLike<number>>

/** A class label: a number, a string or a boolean. */
export type Label = number | string | boolean

/** Labels: an array of numbers, strings or booleans, or a tensor of numbers. */
export type Labels = ArrayLike<Label> | Tensor

/** Every element of numeric data as a new Float64Array (row-major for tensors). */
export function values(x: Data): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/** A rank-1 float64 tensor holding `data`. */
export function vector(data: ArrayLike<number>): Tensor {
  return fromData(data instanceof Float64Array ? data : Float64Array.from(data), [data.length])
}

/** A rank-2 float64 tensor from row-major data. */
export function matrix(data: Float64Array, rows: number, cols: number): Tensor {
  return fromData(data, [rows, cols])
}

/** A matrix as row-major data with its dimensions. A rank-1 input is one column (n × 1). */
export type Dense = { rows: number; cols: number; data: Float64Array }

/** Read a matrix (rank-2 tensor, rank-1 tensor as a column, or rows of numbers) into row-major data. */
export function dense(x: Rows | Data, what: string): Dense {
  if (isTensor(x)) {
    if (x.shape.length === 2) return { rows: x.shape[0], cols: x.shape[1], data: values(x) }
    if (x.shape.length === 1) return { rows: x.shape[0], cols: 1, data: values(x) }
    throw new Error(`metrics: ${what} needs a matrix, got shape [${x.shape.join(', ')}]`)
  }
  const n = x.length
  if (n === 0) return { rows: 0, cols: 0, data: new Float64Array(0) }
  const first = x[0] as unknown
  if (typeof first === 'number') return { rows: n, cols: 1, data: Float64Array.from(x as ArrayLike<number>) }
  const rows = x as ArrayLike<ArrayLike<number>>
  const cols = rows[0].length
  const data = new Float64Array(n * cols)
  for (let i = 0; i < n; i++) {
    const r = rows[i]
    if (r.length !== cols) throw new Error(`metrics: ${what}: row ${i} has ${r.length} values, expected ${cols}`)
    for (let j = 0; j < cols; j++) data[i * cols + j] = r[j]
  }
  return { rows: n, cols, data }
}

/** True for a rank-2 tensor or an array whose first element is itself an array: a matrix rather than a vector. */
export function isMatrixLike(x: unknown): boolean {
  if (isTensor(x)) return x.shape.length === 2
  if (typeof x !== 'object' || x === null || !('length' in x)) return false
  const a = x as ArrayLike<unknown>
  return a.length > 0 && typeof a[0] === 'object' && a[0] !== null && 'length' in (a[0] as object)
}

/** True for numeric data or labels (an array, a typed array or a tensor), false for a plain options object. */
export function isArrayInput(x: unknown): x is Data {
  return isTensor(x) || (typeof x === 'object' && x !== null && 'length' in x)
}

/** Throw unless two inputs have the same number of cases. */
export function sameLength(a: { length: number }, b: { length: number }, what: string): void {
  if (a.length !== b.length) throw new Error(`metrics: ${what}: inputs have ${a.length} and ${b.length} cases`)
}

/** Throw on an empty input, where a mean would be 0/0. */
export function nonEmpty(n: number, what: string): void {
  if (n === 0) throw new Error(`metrics: ${what} needs at least one case`)
}

// ── Labels ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Labels as a plain array (tensor elements become numbers). */
export function labelList(x: Labels): Label[] {
  return isTensor(x) ? Array.from(values(x)) : Array.from(x)
}

/** Order of labels: booleans (false < true), then numbers ascending, then strings in code-point order. */
export function compareLabels(a: Label, b: Label): number {
  const rank = (v: Label) => (typeof v === 'boolean' ? 0 : typeof v === 'number' ? 1 : 2)
  const d = rank(a) - rank(b)
  if (d !== 0) return d
  if (typeof a === 'string') return a < (b as string) ? -1 : a > (b as string) ? 1 : 0
  return Number(a) - Number(b)
}

/** The sorted distinct labels of one or more label lists (as scikit-learn's `unique_labels`). */
export function classesOf(...lists: Label[][]): Label[] {
  const seen = new Set<Label>()
  for (const l of lists) for (const v of l) seen.add(v)
  return [...seen].sort(compareLabels)
}

/** Each label's index in `classes`; −1 for a label not in `classes`. */
export function encode(list: Label[], classes: readonly Label[]): Int32Array {
  const index = new Map<Label, number>()
  classes.forEach((c, k) => index.set(c, k))
  return Int32Array.from(list, (v) => index.get(v) ?? -1)
}

/**
 * The positive class of a binary problem: the given one, else `1` or `true` when present, else the last class in
 * label order (as scikit-learn's `pos_label=1` default, generalised to string labels).
 */
export function positiveOf(classes: readonly Label[], positive?: Label): Label {
  if (positive !== undefined) return positive
  if (classes.includes(1)) return 1
  if (classes.includes(true)) return true
  if (classes.length === 0) return 1
  return classes[classes.length - 1]
}

/** Binary truth as 0/1 from labels and a positive class (default as `positiveOf`). */
export function binaryTruth(yTrue: Labels, positive?: Label): { y: Uint8Array; positive: Label } {
  const list = labelList(yTrue)
  const pos = positiveOf(classesOf(list), positive)
  return { y: Uint8Array.from(list, (v) => (v === pos ? 1 : 0)), positive: pos }
}

// ── Small numeric helpers ────────────────────────────────────────────────────────────────────────────────────────────

/** a / b, or `zero` (default NaN) when b is 0: an undefined ratio is reported, not hidden. */
export function divide(a: number, b: number, zero = NaN): number {
  return b !== 0 ? a / b : zero
}

/** The arithmetic mean of an array (NaN for an empty array). */
export function meanOf(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return x.length ? s / x.length : NaN
}

/** Weighted mean Σ wᵢxᵢ / Σ wᵢ, or the plain mean without weights. */
export function weightedMeanOf(x: ArrayLike<number>, w?: ArrayLike<number>): number {
  if (!w) return meanOf(x)
  let s = 0
  let t = 0
  for (let i = 0; i < x.length; i++) {
    s += w[i] * x[i]
    t += w[i]
  }
  return s / t
}

/** Case weights: the given ones as a Float64Array, checked against n, or undefined. */
export function caseWeights(w: Data | undefined, n: number, what: string): Float64Array | undefined {
  if (w === undefined) return undefined
  const out = values(w)
  if (out.length !== n) throw new Error(`metrics: ${what}: ${out.length} weights for ${n} cases`)
  return out
}

/**
 * Indices that sort `scores` in decreasing order, ties kept in their original order (a stable sort), so that ranking
 * metrics break ties by input position.
 */
export function orderDescending(scores: ArrayLike<number>): Int32Array {
  const idx = Array.from({ length: scores.length }, (_, i) => i)
  idx.sort((a, b) => scores[b] - scores[a] || a - b)
  return Int32Array.from(idx)
}
