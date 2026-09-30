/**
 * Registry metadata (design S §3.1): one `Info` pattern for every kind of named entry (primitives, metrics, losses,
 * distribution families, kernels, windows, datasets, models, algorithms, …). Keys are addresses: `metrics/auroc`
 * names one entry across the lab, the site, URLs, fixtures and workers. Metric and loss metadata specialise it.
 */

import type { Capability } from './model'

/** The kinds of registry entries. The lab adds `view`. */
export type EntryKind =
  | 'primitive'
  | 'metric'
  | 'loss'
  | 'distribution'
  | 'bijector'
  | 'kernel'
  | 'window'
  | 'wavelet'
  | 'filter-design'
  | 'dataset'
  | 'modifier'
  | 'objective'
  | 'model'
  | 'algorithm'
  | 'engine'
  | 'kl-rule'

/** How settled an entry is: stable entries change only through deprecation. */
export type Stability = 'stable' | 'experimental' | 'deprecated'

/** What every registry entry declares. */
export interface Info {
  /** Unique within its kind; equals the export name (`auroc`, `Normal`, `moons`, `adam`). */
  readonly key: string
  readonly kind: EntryKind
  /** The module that defines it, e.g. `metrics`. */
  readonly module: string
  /** Display name, plain text. */
  readonly name: string
  /** The display name in TeX, where it has maths. */
  readonly tex?: string
  /** One sentence, for the catalog and search. */
  readonly summary?: string
  /** Site note slugs; the first is the defining note. */
  readonly notes?: readonly string[]
  /** A `content/glossary.yaml` key. */
  readonly glossary?: string
  /** `content/references.yaml` keys. */
  readonly cite?: readonly string[]
  readonly tags?: readonly string[]
  readonly stability: Stability
  /** For deprecated entries: `module/key` of the replacement. */
  readonly replacedBy?: string
  /** True when it takes a stream (the lab then adds a seed control). */
  readonly random?: boolean
}

/** A value carrying its registry metadata. */
export type Entry<T, I extends Info = Info> = T & { readonly info: I }

// ── Metrics and losses ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a metric reads from a model or a prediction, so a caller can feed it the right output: `labels` (decisions),
 * `sets` (multi-label 0/1 rows), `scores` (an ordering), `probabilities`, `distribution` (a predictive), `values`
 * (point predictions of a real target), `ranking`, `partitions`, `features`, `ratings`, `boxes`, `masks`, `points`,
 * `sequences`, `images`, `signals`, `vectors` and `exposure`.
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

/** The capabilities a metric can need from a model: hard decisions, an ordering of cases, or a predictive. */
export type MetricCapability = Extract<Capability, 'decide' | 'score' | 'predictive'>

/** Metric metadata: the registry `Info` plus what the metric reads, its direction and its range. */
export interface MetricInfo extends Info {
  readonly kind: 'metric'
  readonly inputs: InputKind
  readonly direction: 'higher' | 'lower'
  readonly range: readonly [number, number]
  readonly capability?: MetricCapability
}

/** The family a loss belongs to, as the site's notes group them. */
export type LossFamily = 'classification' | 'regression' | 'ranking' | 'retrieval' | 'representation' | 'divergence'

/**
 * What a loss reads from a model: `logits`, `probabilities`, `margins` (scores with labels in {−1, +1}), `values`,
 * `distribution` (predictive parameters), `scores` (of the items of a list), `embeddings` or `distributions` (two to
 * compare). Shared names mean the same as in `InputKind`.
 */
export type LossInput =
  'logits' | 'probabilities' | 'margins' | 'values' | 'distribution' | 'scores' | 'embeddings' | 'distributions'

/** Loss metadata: the registry `Info` plus its family, inputs, and the metric its minimiser optimises. */
export interface LossInfo extends Info {
  readonly kind: 'loss'
  readonly family: LossFamily
  readonly inputs: LossInput
  readonly target?: string
  /** The key of the metric this loss's minimiser optimises (`metrics/<key>`). */
  readonly pairedMetric?: string
}
