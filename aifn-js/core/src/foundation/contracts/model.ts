/**
 * Fitted models and their capabilities (design S §2.7), after scikit-learn's mixins but checked by the compiler: a
 * figure can ask a model only for what it declares. A model is a plain object with its fitted state as public fields
 * and its capabilities as methods.
 */

import type { Trace } from './algorithm'
import type { Distribution } from './distribution'
import type { Kinded } from './kinds'
import type { Size, Tensor } from './numbers'
import type { Stream } from './random'
import type { Info } from './registry'

/** The partial forward pass: logits, latent means, raw scores (the "head" a readout completes). */
export interface Fitted<X, H = Tensor> {
  forward(x: X): H
}

/** A decision per input: an argmax class, a cluster label, an action, or a point prediction. */
export interface Decides<X, Y = Tensor> {
  decide(x: X): Y
}

/** A predictive distribution over the batch of inputs. */
export interface Predicts<X, D extends Distribution = Distribution> {
  predictive(x: X): D
}

/** E[f(y) | x] per input (the mean without `f`), shape of the predictive's batch. */
export interface Expects<X> {
  expect(x: X, f?: (y: number) => number): Tensor
}

/** Per-class or per-item scores, e.g. logits [N, K] or a binary logit [N]. */
export interface Scores<X> {
  score(x: X): Tensor
}

/** A representation of the inputs: embeddings, projections, transformed features. */
export interface Transforms<X, Z = Tensor> {
  transform(x: X): Z
}

/** Draws from the predictive: [n, ...batch] with `n`, else [...batch]. */
export interface Samples<X, Y = Tensor> {
  sample(s: Stream, x: X, n?: Size): Y
}

/** A fitted model that kept the trace of its training loop. */
export interface Trained<S = unknown> {
  training: Trace<S>
}

/** The names of the capabilities: one vocabulary for models, metrics (`info.capability`) and evaluation. */
export type Capability = 'forward' | 'decide' | 'predictive' | 'expect' | 'score' | 'transform' | 'sample'

/** What a supervised or unsupervised task asks of a model. */
export type Task = 'classification' | 'regression' | 'clustering' | 'density' | 'embedding' | 'ranking' | 'forecasting'

/** Registry metadata of a model: its task, the capabilities it declares, and its hyperparameter space. */
export interface ModelInfo extends Info {
  readonly task: Task
  readonly capabilities: readonly Capability[]
  /** True for models that cannot score new inputs (t-SNE, spectral clustering). */
  readonly transductive?: boolean
}

/**
 * A fitted model (target): the `kind` brand, a reference to its registry entry, and its capabilities as methods (any
 * of `Fitted`, `Decides`, `Predicts`, `Expects`, `Scores`, `Transforms`, `Samples`, `Trained`).
 */
export interface Model extends Kinded<'model'> {
  readonly info?: ModelInfo
  readonly transductive?: true
}
