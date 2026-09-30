/**
 * Datasets and estimators: the shapes every `fit` takes and returns.
 */

import type { Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import type { TraceOptions } from 'aifn/trace'
import { sizeOf, values } from './util'

/** One named column of a table: a numeric tensor ([n] or [n, k]) or a list of category labels. */
export type Column = Tensor | readonly (string | number)[]

/** Named columns of equal length, e.g. `{ age: tensor([...]), city: ['Cork', 'Paris', ...] }`. */
export type Table = { readonly [name: string]: Column }

/** Features: a matrix [n, d], or a table of named columns. */
export type Features = Tensor | Table

/**
 * A dataset: inputs `x` (n rows), and optionally targets `y` ([n]) and group labels (for grouped splitters). Row i of
 * every field belongs to example i.
 */
export interface Dataset<X = Tensor, Y = Tensor> {
  x: X
  y?: Y
  groups?: Column
}

/** A dataset with targets. */
export interface Supervised<X = Tensor, Y = Tensor> extends Dataset<X, Y> {
  y: Y
}

/** Options every `fit` accepts. */
export interface FitOptions {
  /** The randomness of the fit (initialisation, random features, minibatches). */
  stream?: Stream
  /** How iterative fits record their training trace (default: every step, no recorders). */
  trace?: Pick<TraceOptions<unknown>, 'every' | 'record' | 'checkpointEvery'>
}

/**
 * An estimator: an unfitted model description with hyperparameters, whose `fit` returns a fitted model, a plain object
 * with its fitted state as public fields and its capabilities as methods. Fitting never mutates the estimator.
 */
export interface Estimator<D, M> {
  /** A readable name, e.g. "logistic-regression". */
  readonly name: string
  /** The hyperparameters it was made with. */
  readonly params?: unknown
  fit(data: D, options?: FitOptions): M
}

/** The fitted model type of an estimator. */
export type ModelOf<E> =
  E extends Estimator<never, infer M> ? M : E extends { fit(...args: never[]): infer M } ? M : never

/** The data type an estimator fits on. */
export type DataOf<E> = E extends { fit(data: infer D, ...rest: never[]): unknown } ? D : never

// ── Rows ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

function isTensorLike(x: unknown): x is Tensor {
  return typeof x === 'object' && x !== null && 'shape' in x && 'data' in x && 'strides' in x
}

/** Number of rows of a tensor (its first axis), label list or table (all columns must agree). */
export function rowCount(x: Features | Column): number {
  if (isTensorLike(x)) {
    if (x.shape.length === 0) throw new Error('rowCount: a scalar has no rows')
    return x.shape[0]
  }
  if (Array.isArray(x)) return x.length
  let n = -1
  for (const [name, column] of Object.entries(x as Table)) {
    const m = rowCount(column)
    if (n >= 0 && m !== n) throw new Error(`rowCount: column "${name}" has ${m} rows, others ${n}`)
    n = m
  }
  if (n < 0) throw new Error('rowCount: a table with no columns')
  return n
}

/** The rows `index` (in that order, repeats allowed) of a tensor, label list or table. */
export function takeRows<X extends Features | Column>(x: X, index: ArrayLike<number>): X {
  if (isTensorLike(x)) {
    const rowSize = sizeOf(x.shape.slice(1))
    const src = values(x)
    const out = x.dtype === 'int32' ? new Int32Array(index.length * rowSize) : new Float64Array(index.length * rowSize)
    for (let r = 0; r < index.length; r++) {
      const i = index[r]
      if (!(i >= 0 && i < x.shape[0])) throw new Error(`takeRows: row ${i} out of range [0, ${x.shape[0]})`)
      for (let j = 0; j < rowSize; j++) out[r * rowSize + j] = src[i * rowSize + j]
    }
    return fromData(out, [index.length, ...x.shape.slice(1)]) as X
  }
  if (Array.isArray(x)) return Array.from(index, (i) => x[i]) as unknown as X
  const out: Record<string, Column> = {}
  for (const [name, column] of Object.entries(x as Table)) out[name] = takeRows(column, index)
  return out as X
}

/** The rows `index` of every field of a dataset. */
export function takeData<D extends Dataset<Features | Column, Features | Column>>(
  data: D,
  index: ArrayLike<number>,
): D {
  const out: Record<string, unknown> = { ...(data as object) }
  for (const key of ['x', 'y', 'groups'] as const) {
    const v = (data as Dataset<Features | Column, Features | Column>)[key]
    if (v !== undefined) out[key] = takeRows(v, index)
  }
  return out as D
}
