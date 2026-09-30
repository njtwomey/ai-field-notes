/**
 * Datasets, recipes and truths (design S §2.8). One `Dataset` shape for estimators, validation and the generators;
 * a recipe is data that a generic interpreter replays through the dataset registry; a truth is a model of the
 * generating process. Today two dataset shapes exist (`aifn/learning/estimators`' generic one and `aifn-applied/data`' concrete
 * one), and truths are bespoke objects; phase 1 moves both onto these.
 */

import type { Distribution } from './distribution'
import type { Kinded } from './kinds'
import type { Decides, Expects, Model, Predicts, Task } from './model'
import type { Tensor, TensorWire } from './numbers'
import type { Info } from './registry'
import type { Space } from './space'

/** One named column of a table: a numeric tensor ([n] or [n, k]) or a list of category labels. */
export type Column = Tensor | readonly (string | number)[]

/** Named columns of equal length, e.g. `{ age: tensor([...]), city: ['Cork', 'Paris', ...] }`. */
export type Table = { readonly [name: string]: Column }

/** Features: a matrix [n, d], or a table of named columns. */
export type Features = Tensor | Table

/** One step of how a dataset was made: the generator or modifier (`op`) and its parameters. */
export interface RecipeStep {
  op: string
  params: Record<string, unknown>
}

/**
 * A recipe (target): a registered base generator, a seed, the base's knobs, and modifiers applied in order, each
 * looked up in the dataset registry. Plain data, so it round-trips through a URL.
 */
export interface Recipe {
  readonly base: string
  readonly seed: number | string
  readonly knobs: Readonly<Record<string, unknown>>
  readonly modifiers: readonly { readonly op: string; readonly params: Readonly<Record<string, unknown>> }[]
}

/**
 * The truth of a synthetic problem (target): a model of the generating process, with the Bayes rule (`decide`), the
 * Bayes posterior or the conditional law of y (`predictive`) and the regression function (`expect`), plus the lowest
 * risk any predictor can reach.
 */
export interface Truth<X = Tensor> extends Model, Decides<X>, Predicts<X, Distribution>, Expects<X> {
  readonly task: 'classification' | 'regression'
  /** The Bayes error (classification) or Bayes risk under squared loss (regression). */
  readonly bayesRisk: number
}

/** What a dataset is and where it came from (target). */
export interface DatasetMeta {
  /** A short name, e.g. `moons`. */
  readonly name: string
  /** One or two sentences for a caption: how the data were made and what the labels mean. */
  readonly description: string
  readonly task: Task | 'manifold' | 'sequence' | 'images' | 'recommendation'
  readonly featureNames?: readonly string[]
  readonly labelNames?: readonly string[]
  readonly targetName?: string
  /** A citation for real data or for the generator's recipe. */
  readonly source?: string
  readonly url?: string
  /** The known generating process, where it has a closed form. */
  readonly truth?: Truth
  /** How the dataset was made: the provenance. */
  readonly recipe?: Recipe
}

/**
 * A dataset (target): features `x` (n rows), optional targets `y`, group labels, a continuous coordinate `t` (for
 * colouring), the noise-free target `f`, and metadata. Row i of every field belongs to example i.
 */
export interface Dataset<X extends Features = Features, Y = Tensor> extends Kinded<'dataset'> {
  readonly x: X
  readonly y?: Y
  readonly groups?: Column
  readonly t?: Tensor
  readonly f?: Tensor
  readonly meta?: DatasetMeta
}

/** Registry metadata of a dataset generator: its task, knobs and whether it has a truth. */
export interface DatasetInfo extends Info {
  readonly task: DatasetMeta['task']
  readonly knobs: Space
  readonly truth: boolean
}

/** A dataset on the wire; the truth, which holds functions, stays behind (its recipe rebuilds it). */
export interface DatasetWire {
  readonly kind: 'dataset'
  readonly x: TensorWire
  readonly y?: TensorWire
  readonly meta?: Omit<DatasetMeta, 'truth'>
}
