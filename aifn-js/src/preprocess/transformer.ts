/**
 * The shape of a preprocessing step: an estimator whose fitted model transforms inputs (and inverts, where an inverse
 * exists), plus shared helpers for numeric matrices.
 */

import type { Dataset, Estimator, FitOptions, Transforms } from 'aifn/estimators'
import { copy, fromData, isContiguous, type Tensor } from 'aifn/tensor'

/** A fitted transform: its fitted state as public fields, `transform`, and `inverse` where one exists. */
export interface FittedTransform<X = Tensor, Z = Tensor> extends Transforms<X, Z> {
  /** Which transform this is, e.g. "standard-scaler". */
  readonly kind: string
}

/** A fitted transform with an inverse: `inverse(transform(x))` recovers x (up to rounding). */
export interface Invertible<X = Tensor, Z = Tensor> {
  inverse(z: Z): X
}

/** An unfitted transform: `fit({ x, y? }, options)` returns the fitted transform. */
export type Transformer<X, F> = Estimator<Dataset<X, Tensor>, F>

/** Fit a transformer and transform its training inputs in one call. */
export function fitTransform<X, F extends FittedTransform<X, unknown>>(
  transformer: Transformer<X, F>,
  data: Dataset<X, Tensor>,
  options?: FitOptions,
): { model: F; z: ReturnType<F['transform']> } {
  const model = transformer.fit(data, options)
  return { model, z: model.transform(data.x) as ReturnType<F['transform']> }
}

/** The elements of `t` in row-major order (shared with `t` when already dense float64; do not mutate). */
export function values(t: Tensor): Float64Array {
  if (
    t.dtype === 'float64' &&
    t.offset === 0 &&
    isContiguous(t) &&
    t.data.length === t.shape.reduce((a, b) => a * b, 1)
  ) {
    return t.data as Float64Array
  }
  return copy(t, 'float64').data as Float64Array
}

/** A numeric matrix [n, d] as its size and row-major values; throws naming the caller otherwise. */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  if (x.shape.length !== 2) throw new Error(`${where}: expected a matrix [n, d], got shape [${x.shape.join(', ')}]`)
  return { n: x.shape[0], d: x.shape[1], v: values(x) }
}

/** Check that a transform input has the fitted number of columns. */
export function checkColumns(d: number, fitted: number, where: string): void {
  if (d !== fitted) throw new Error(`${where}: fitted on ${fitted} columns, given ${d}`)
}

/** Apply f(value, column) to every element of an [n, d] matrix. */
export function mapColumns(x: Tensor, d: number, where: string, f: (v: number, j: number) => number): Tensor {
  const m = matrix(x, where)
  checkColumns(m.d, d, where)
  return fromData(
    Float64Array.from(m.v, (v, k) => f(v, k % d)),
    [m.n, d],
  )
}

/** Column j of an [n, d] matrix's values. */
export function column(v: Float64Array, n: number, d: number, j: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = v[i * d + j]
  return out
}
