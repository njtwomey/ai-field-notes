/** Private helpers for `aifn/classify`: dense float64 views, label checks and the probability-model builder. */

import { categoricalPredictive, type CategoricalPredictive } from 'aifn/estimators'
import { copy, fromData, isContiguous, type Tensor } from 'aifn/tensor'

/** The elements of `t` in row-major order as a Float64Array (shared when already dense float64; do not mutate). */
export function values(t: Tensor): Float64Array {
  let size = 1
  for (const d of t.shape) size *= d
  if (t.dtype === 'float64' && t.offset === 0 && isContiguous(t) && t.data.length === size) {
    return t.data as Float64Array
  }
  return copy(t, 'float64').data as Float64Array
}

/** Rows, columns and values of a matrix [n, d], or throw naming the caller. */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  if (x.shape.length !== 2) throw new Error(`${where}: expected a matrix [n, d], got shape [${x.shape.join(', ')}]`)
  return { n: x.shape[0], d: x.shape[1], v: values(x) }
}

/**
 * Integer class labels 0 … K−1 from a vector [n] (or [n, 1]); K is one more than the largest label (at least 2 unless
 * `minClasses` says otherwise).
 */
export function classLabels(y: Tensor, n: number, where: string, minClasses = 2): { y: Int32Array; k: number } {
  if (y.shape.length > 2 || (y.shape.length === 2 && y.shape[1] !== 1)) {
    throw new Error(`${where}: expected labels [n], got shape [${y.shape.join(', ')}]`)
  }
  const v = values(y)
  if (v.length !== n) throw new Error(`${where}: ${n} rows of x but ${v.length} labels`)
  const out = new Int32Array(n)
  let k = 0
  for (let i = 0; i < n; i++) {
    if (!(Number.isInteger(v[i]) && v[i] >= 0)) throw new Error(`${where}: labels must be integers 0 … K−1`)
    out[i] = v[i]
    k = Math.max(k, v[i] + 1)
  }
  return { y: out, k: Math.max(k, minClasses) }
}

/** Real-valued targets [n] as a Float64Array. */
export function targets(y: Tensor, n: number, where: string): Float64Array {
  const v = values(y)
  if (v.length !== n) throw new Error(`${where}: ${n} rows of x but ${v.length} targets`)
  return v
}

/** Sample weights (default all 1), checked to be non-negative and finite. */
export function sampleWeights(w: Tensor | undefined, n: number, where: string): Float64Array {
  if (!w) return new Float64Array(n).fill(1)
  const v = values(w)
  if (v.length !== n) throw new Error(`${where}: ${n} rows but ${v.length} sample weights`)
  for (const u of v) if (!(u >= 0 && Number.isFinite(u))) throw new Error(`${where}: sample weights must be ≥ 0`)
  return v
}

/** Checks that a prediction input has the fitted number of features. */
export function inputs(x: Tensor, d: number, where: string): { n: number; v: Float64Array } {
  const m = matrix(x, where)
  if (m.d !== d) throw new Error(`${where}: fitted on ${d} features, given ${m.d}`)
  return { n: m.n, v: m.v }
}

/** The index of the largest value in row `i` of an [n, k] array (the first of equals). */
export function argmaxRow(a: ArrayLike<number>, i: number, k: number): number {
  let best = 0
  for (let c = 1; c < k; c++) if (a[i * k + c] > a[i * k + best]) best = c
  return best
}

/** Row-wise argmax of an [n, k] array as int32 labels [n]. */
export function argmaxRows(a: ArrayLike<number>, n: number, k: number): Tensor {
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) out[i] = argmaxRow(a, i, k)
  return fromData(out, [n])
}

/** Row-wise softmax of scores [n, k] (stable: the row maximum is subtracted first). */
export function softmaxRows(s: ArrayLike<number>, n: number, k: number): Float64Array {
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    let m = -Infinity
    for (let c = 0; c < k; c++) m = Math.max(m, s[i * k + c])
    let z = 0
    for (let c = 0; c < k; c++) z += out[i * k + c] = m === -Infinity ? 1 : Math.exp(s[i * k + c] - m)
    for (let c = 0; c < k; c++) out[i * k + c] /= z
  }
  return out
}

/** The logistic function, computed without overflow. */
export const sigmoid = (t: number): number => (t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t)))

/** Squared Euclidean distance between row i of `a` (width d) and row j of `b`. */
export function squaredDistance(a: Float64Array, i: number, b: Float64Array, j: number, d: number): number {
  let s = 0
  for (let c = 0; c < d; c++) {
    const t = a[i * d + c] - b[j * d + c]
    s += t * t
  }
  return s
}

/**
 * The capabilities of a classifier whose class probabilities are computed from a head [m, K] (log-probabilities,
 * scores or probabilities): `forward` and `score` return the head, `predictive` a categorical law, `decide` the most
 * probable class (ties to the lowest label).
 */
export function probabilityModel(
  head: (x: Tensor) => Float64Array,
  toProbs: (h: Float64Array, m: number) => Float64Array,
  k: number,
) {
  const rows = (x: Tensor) => x.shape[0]
  const forward = (x: Tensor): Tensor => fromData(head(x), [rows(x), k])
  const probabilities = (x: Tensor): Float64Array => toProbs(head(x), rows(x))
  return {
    forward,
    score: forward,
    predictive: (x: Tensor): CategoricalPredictive => categoricalPredictive(fromData(probabilities(x), [rows(x), k])),
    decide: (x: Tensor): Tensor => argmaxRows(probabilities(x), rows(x), k),
    /** Class probabilities [m, K] as a tensor. */
    probabilities: (x: Tensor): Tensor => fromData(probabilities(x), [rows(x), k]),
  }
}

/** Row-wise normalisation of log-probabilities [n, k] into probabilities. */
export function normaliseLog(logp: Float64Array, n: number, k: number): Float64Array {
  return softmaxRows(logp, n, k)
}

/** A float64 matrix tensor over `v`. */
export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])

/** An int32 vector tensor. */
export const ints = (v: ArrayLike<number>): Tensor => fromData(Int32Array.from(v), [v.length])

/** A float64 vector tensor. */
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
