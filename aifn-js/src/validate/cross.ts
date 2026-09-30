/** Cross-validation (plan §5.5): fit on each split's training rows, evaluate on its test rows, keep everything. */

import {
  hasTraining,
  outputs,
  rowCount,
  takeData,
  type AnyMetric,
  type Column,
  type Distribution,
  type FitOptions,
  type Need,
  type Requirement,
  type Supervised,
} from 'aifn/estimators'
import type { Stream } from 'aifn/random'
import { fromData, isTensor, type Tensor } from 'aifn/tensor'
import { now, type Trace } from 'aifn/trace'
import { assignment, type Split, type Splitter } from './splitters'

/** A dataset for cross-validation: inputs, targets and optional group labels. */
export type CrossValidationData<X> = Supervised<X, Tensor> & { groups?: Column }

/** Something with `fit` on supervised data. */
export type Fittable<X, M> = { readonly name: string; fit(data: Supervised<X, Tensor>, options?: FitOptions): M }

/** One fold of a cross-validation. */
export interface Fold<M> {
  index: number
  /** Sorted row indices of the training and test sets. */
  train: Tensor
  test: Tensor
  /** The model fitted on the training rows, with all its fitted state. */
  model: M
  /** The model's outputs on the test rows, by capability (those the metrics need). */
  predictions: Partial<Record<Need, Tensor | Distribution>>
  /** Each metric on the test rows, by name. */
  metrics: Record<string, number>
  /** Each metric on the training rows, when `trainMetrics` is set. */
  trainMetrics?: Record<string, number>
  /** The model's training trace, when it kept one. */
  training?: Trace<unknown>
  /** Wall time of the fit, in milliseconds. */
  fitMs: number
}

/** The result of `crossValidate`. */
export interface CrossValidation<M> {
  splitter: string
  splits: Split[]
  /** Fold assignment [folds, n]: 1 test, 0 train, −1 unused (see `assignment`). */
  assignment: Tensor
  folds: Fold<M>[]
  /** Each metric over folds, [folds]. */
  scores: Record<string, Tensor>
  mean: Record<string, number>
  /** Sample standard deviation over folds (÷ (folds − 1)). */
  std: Record<string, number>
  /** Each metric's direction, by name. */
  directions: Record<string, 'higher' | 'lower'>
  /**
   * Out-of-fold predictions [n] per tensor-valued capability ('decide', 'expect'; scores of shape [n]): each row's
   * prediction from the model that did not train on it; NaN for rows never tested. With repeated splits the last wins.
   */
  outOfFold: Partial<Record<Need, Tensor>>
}

/** Options of `crossValidate`. */
export interface CrossValidateOptions {
  /** Randomness: the splitter gets `stream.child('split')`, fold f's fit `stream.child('fold', f)`. */
  stream?: Stream
  /** Also evaluate each fold's metrics on its training rows (default false). */
  trainMetrics?: boolean
}

function evaluateOn(
  model: unknown,
  x: unknown,
  y: Tensor,
  metrics: readonly AnyMetric[],
): { predictions: Partial<Record<Need, Tensor | Distribution>>; values: Record<string, number> } {
  const predictions = outputs(
    model,
    metrics.map((m) => m.needs),
    x,
  )
  const values: Record<string, number> = {}
  for (const m of metrics) values[m.name] = (m.compute as (y: Tensor, o: unknown) => number)(y, predictions[m.needs])
  return { predictions, values }
}

/**
 * Cross-validate an estimator (or a pipeline): for each split of `splitter`, fit on the training rows and evaluate
 * `metrics` on the test rows. The fitted model must have every capability the metrics need (a compile error
 * otherwise). Everything is kept: the fold assignment matrix, each fold's fitted model, predictions, metrics and
 * training trace, per-metric scores over folds with their mean and standard deviation, and out-of-fold predictions.
 */
export function crossValidate<X, M extends Requirement<X, Ms[number]['needs']>, const Ms extends readonly AnyMetric[]>(
  estimator: Fittable<X, M>,
  data: CrossValidationData<X>,
  splitter: Splitter,
  metrics: Ms,
  options: CrossValidateOptions = {},
): CrossValidation<M> {
  const { stream, trainMetrics = false } = options
  const n = rowCount(data.x as never)
  const splits = splitter.split({ n, y: data.y, groups: data.groups }, stream?.child('split'))
  const folds: Fold<M>[] = splits.map((split, f) => {
    const train = takeData(data as never, split.train.data) as CrossValidationData<X>
    const test = takeData(data as never, split.test.data) as CrossValidationData<X>
    const start = now()
    const model = estimator.fit(train, { stream: stream?.child('fold', f) })
    const fitMs = now() - start
    const { predictions, values } = evaluateOn(model, test.x, test.y, metrics)
    const fold: Fold<M> = { index: f, train: split.train, test: split.test, model, predictions, metrics: values, fitMs }
    if (trainMetrics) fold.trainMetrics = evaluateOn(model, train.x, train.y, metrics).values
    if (hasTraining(model)) fold.training = model.training
    return fold
  })
  const scores: Record<string, Tensor> = {}
  const mean: Record<string, number> = {}
  const std: Record<string, number> = {}
  const directions: Record<string, 'higher' | 'lower'> = {}
  for (const m of metrics) {
    const v = Float64Array.from(folds, (fold) => fold.metrics[m.name])
    const mu = v.reduce((a, b) => a + b, 0) / v.length
    scores[m.name] = fromData(v, [v.length])
    mean[m.name] = mu
    std[m.name] = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / (v.length - 1)) : NaN
    directions[m.name] = m.direction
  }
  const outOfFold: Partial<Record<Need, Tensor>> = {}
  for (const need of new Set(metrics.map((m) => m.needs))) {
    if (need === 'predictive') continue
    const out = new Float64Array(n).fill(NaN)
    let ok = true
    for (const fold of folds) {
      const p = fold.predictions[need]
      if (!p || !isTensor(p) || p.shape.length !== 1 || p.shape[0] !== fold.test.shape[0]) {
        ok = false
        break
      }
      fold.test.data.forEach((i, r) => (out[i] = p.data[p.offset + r * (p.strides[0] ?? 1)]))
    }
    if (ok) outOfFold[need] = fromData(out, [n])
  }
  return {
    splitter: splitter.name,
    splits,
    assignment: assignment(splits, n),
    folds,
    scores,
    mean,
    std,
    directions,
    outOfFold,
  }
}
