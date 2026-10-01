/**
 * Registry metadata (design S §3.1): one `Info` pattern for every kind of named entry (primitives, metrics, losses,
 * distribution families, kernels, windows, datasets, models, algorithms, …). Keys are addresses: `metrics/auroc`
 * names one entry across the lab, the site, URLs, fixtures and workers. Metric and loss metadata specialise it.
 */

import type { DomainKind } from './environment'
import type { Capability } from './model'
import type { Space } from './space'

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
  | 'environment'
  | 'agent'
  | 'log-density'
  | 'algorithm'
  | 'engine'
  | 'kl-rule'
  | 'likelihood'
  | 'link'

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

// ── Probability ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Where a family puts its mass, as a name the catalog and pickers can read without building a distribution: `real`,
 * `positive` (0, ∞), `non-negative` [0, ∞), `unit-interval` (0, 1), `interval` (set by parameters), `circle`,
 * `integers` (bounded by parameters), `non-negative-integers`, `positive-integers`, `binary` {0, 1}, `categories`
 * {0, …, K − 1}, `simplex`, `real-vector`, `count-vector` and `positive-definite`. Mixtures and compositions use `varies`.
 */
export type SupportName =
  | 'real'
  | 'positive'
  | 'non-negative'
  | 'unit-interval'
  | 'interval'
  | 'circle'
  | 'integers'
  | 'non-negative-integers'
  | 'positive-integers'
  | 'binary'
  | 'categories'
  | 'simplex'
  | 'real-vector'
  | 'count-vector'
  | 'positive-definite'
  | 'varies'

/** A distribution family (a constructor such as `Normal`): its parameters in constructor order, support and structure. */
export interface DistributionInfo extends Info {
  readonly kind: 'distribution'
  /** The constructor's parameters, in argument order, with ranges and defaults for pickers and tests. */
  readonly params: Space
  readonly support: SupportName
  readonly discrete: boolean
  /** 0 for scalar events, 1 for vectors, 2 for matrices. */
  readonly eventRank: number
  /** True when instances expose `expFamily` (natural parameters, sufficient statistics, log-partition). */
  readonly expFamily: boolean
  /** True for a constructor that builds on other distributions (Mixture, Independent, Transformed). */
  readonly composite?: boolean
}

/** A bijector (or a bijector factory): its domain and codomain as support names, and its parameters. */
export interface BijectorInfo extends Info {
  readonly kind: 'bijector'
  readonly domain: SupportName
  readonly codomain: SupportName
  /** Factory arguments; empty for a fixed map. */
  readonly params: Space
  /** True when the entry is a factory returning a bijector rather than a bijector. */
  readonly factory: boolean
}

/** A closed-form KL(p ‖ q) rule between two registered families. */
export interface KlRuleInfo extends Info {
  readonly kind: 'kl-rule'
  /** The family key of p. */
  readonly p: string
  /** The family key of q. */
  readonly q: string
}

/** A GLM link function. */
export interface LinkInfo extends Info {
  readonly kind: 'link'
  /** The mean space the link maps from, as a support name. */
  readonly meanSpace: SupportName
}

/** An exponential-dispersion family (a factory returning a `Family`). */
export interface LikelihoodInfo extends Info {
  readonly kind: 'likelihood'
  /** The response's support. */
  readonly support: SupportName
  readonly canonicalLink: string
  /** The links the family is used with (keys of link entries). */
  readonly links: readonly string[]
  /** True when the dispersion is estimated rather than fixed at 1. */
  readonly dispersion: boolean
  /** Factory arguments (the negative binomial's θ); empty otherwise. */
  readonly params: Space
}

// ── Kernels ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A covariance kernel factory: its hyperparameters and whether it depends on x − y only. */
export interface KernelInfo extends Info {
  readonly kind: 'kernel'
  readonly hyper: Space
  readonly stationary: boolean
  /** True for a combinator of other kernels (sum, product). */
  readonly composite?: boolean
}

// ── Signals ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A window function: its parameters and its spectral figures of merit at the default parameters (Harris, 1978),
 * measured on the periodic window: the main-lobe width between the first nulls, in DFT bins, and the peak side-lobe
 * level relative to the main lobe, in dB.
 */
export interface WindowInfo extends Info {
  readonly kind: 'window'
  readonly params: Space
  /** Omitted when it depends on the length (a Gaussian window's width is set in samples). */
  readonly mainLobeWidth?: number
  readonly sideLobeDb?: number
}

/** A wavelet: its family, vanishing moments and filter length (orthogonal wavelets), or its parameters (continuous). */
export interface WaveletInfo extends Info {
  readonly kind: 'wavelet'
  readonly family: 'haar' | 'daubechies' | 'morlet'
  readonly continuous: boolean
  readonly orthogonal: boolean
  /** Vanishing moments of ψ (orthogonal wavelets). */
  readonly vanishingMoments?: number
  /** The number of taps of each filter (orthogonal wavelets). */
  readonly taps?: number
  readonly params: Space
}

/** A filter design method: FIR or IIR, and the specification fields it honours. */
export interface FilterDesignInfo extends Info {
  readonly kind: 'filter-design'
  readonly family: 'iir' | 'fir'
  /** The band types it can design. */
  readonly bands: readonly ('lowpass' | 'highpass' | 'bandpass' | 'bandstop')[]
  /** The specification fields it reads: `order`, `numtaps`, `cutoff`, `passRippleDb`, `stopAttenDb`, `window`. */
  readonly honours: readonly string[]
  readonly params: Space
}

// ── Algorithms ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What an algorithm's factory takes: `objective` (a function to minimise), `least-squares` (residuals), `root` (a
 * scalar equation), `system` (a nonlinear system), `integral`, `ode`, `sde`, `log-density` (a target to sample),
 * `factor-graph`, `chain` (chain potentials), `gaussian-model`, `graph`, `flow-network`, `linear-program`,
 * `quadratic-program`, `integer-program`, `assignment`, `dynamic-program`, `riccati`, `transport`, `signal`,
 * `lti-system`, `network` (a model to train), `sequence` (a data stream to filter).
 */
export type AlgorithmProblem =
  | 'objective'
  | 'least-squares'
  | 'root'
  | 'system'
  | 'integral'
  | 'ode'
  | 'sde'
  | 'log-density'
  | 'factor-graph'
  | 'chain'
  | 'gaussian-model'
  | 'graph'
  | 'flow-network'
  | 'linear-program'
  | 'quadratic-program'
  | 'integer-program'
  | 'assignment'
  | 'dynamic-program'
  | 'riccati'
  | 'transport'
  | 'signal'
  | 'lti-system'
  | 'network'
  | 'sequence'

/** The `Status` flags a state may set. */
export type StatusFlag = 'converged' | 'diverged' | 'stalled' | 'terminated'

/**
 * Which state fields play the roles a generic trace view plots by default (design S §2.3): the `iterate` (x, the
 * position, the weights), the `objective` value being optimised or tracked (a loss, a log-likelihood, an ELBO, a
 * residual norm), its `grad`, and the `stepSize`. Each names a top-level field of the state; a role an algorithm has no
 * field for is omitted. `flags` lists the `Status` flags its states set.
 */
export interface StateRoles {
  readonly iterate?: string
  readonly objective?: string
  readonly grad?: string
  readonly stepSize?: string
  readonly flags: readonly StatusFlag[]
}

/** An algorithm factory: the problem it takes and the roles of its state's fields. */
export interface AlgorithmInfo extends Info {
  readonly kind: 'algorithm'
  readonly problem: AlgorithmProblem
  readonly state: StateRoles
}

// ── Primitives (the catalog's view of the primitive table) ──────────────────────────────────────────────────────────

/** A primitive as the catalog lists it: its arity, rule sources and documentation. */
export interface PrimitiveInfo extends Info {
  readonly kind: 'primitive'
  readonly primitive: 'elementwise' | 'general'
  readonly arity: number | 'variadic'
  readonly rules: { readonly vjp: string; readonly jvp: string; readonly batch: string; readonly shape: string }
  readonly formula?: string
}

// ── Environments ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Environment metadata: a named environment for sequential decisions (a bandit, a Markov decision process, a control
 * problem) with the space of its scalar parameters. Environments on the protocol (`Environment`) also declare the kinds
 * of their observation and action domains and their optional capabilities, so the lab offers only valid pairs.
 */
export interface EnvironmentInfo extends Info {
  readonly kind: 'environment'
  readonly family: 'bandit' | 'contextual-bandit' | 'mdp' | 'control'
  readonly params: Space
  readonly observation?: DomainKind
  readonly action?: DomainKind
  readonly capabilities?: readonly ('model' | 'oracle' | 'render')[]
}

/** What an agent needs of an environment: domain kinds, and an explicit model for planners. */
export interface AgentRequires {
  readonly observation?: DomainKind
  readonly action?: DomainKind
  readonly model?: 'tabular'
}

/** Agent metadata: a learning or acting agent (`Agent`), its hyperparameters and what it requires. */
export interface AgentInfo extends Info {
  readonly kind: 'agent'
  readonly params: Space
  readonly requires: AgentRequires
}
