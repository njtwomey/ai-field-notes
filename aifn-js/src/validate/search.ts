/**
 * Hyperparameter search and nested cross-validation (plan §5.5). A search scores every candidate by cross-validation,
 * keeps the full results table, and refits the best candidate on all the data; as an estimator it can itself be
 * cross-validated, which is nested cross-validation.
 */

import type { AnyMetric, FitOptions } from 'aifn/estimators'
import type { Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { crossValidate, type CrossValidation, type CrossValidationData, type Fittable } from './cross'
import type { Splitter } from './splitters'

/** A grid: the values to try for each parameter. */
export type GridSpace = { readonly [name: string]: readonly unknown[] }

/** The parameters a grid produces: one value of each list. */
export type GridParams<S extends GridSpace> = { -readonly [K in keyof S]: S[K][number] }

/** A random-search space: per parameter, a list to choose from uniformly or a sampler of the stream. */
export type RandomSpace = { readonly [name: string]: readonly unknown[] | ((s: Stream) => unknown) }

/** The parameters a random space produces. */
export type RandomParams<S extends RandomSpace> = {
  -readonly [K in keyof S]: S[K] extends (s: Stream) => infer T ? T : S[K] extends readonly (infer T)[] ? T : never
}

/** A value uniform on [low, high). */
export function uniformRange(low: number, high: number): (s: Stream) => number {
  return (s) => low + (high - low) * s.uniform()
}

/** A value log-uniform on [low, high) (low > 0), for scale parameters such as penalties and learning rates. */
export function logUniformRange(low: number, high: number): (s: Stream) => number {
  if (!(low > 0 && high > low)) throw new Error('logUniformRange: needs 0 < low < high')
  return (s) => Math.exp(Math.log(low) + (Math.log(high) - Math.log(low)) * s.uniform())
}

/** An integer uniform on {low, …, high}. */
export function integerRange(low: number, high: number): (s: Stream) => number {
  return (s) => low + s.int(high - low + 1)
}

/** One row of a search's results table. */
export interface SearchRow<P> {
  index: number
  params: P
  /** The selection metric on each fold, [folds]. */
  scores: Tensor
  mean: number
  std: number
  /** 1 for the best mean (by the metric's direction); ties share the lower rank. */
  rank: number
  /** Total fit time over the folds, in milliseconds. */
  fitMs: number
  /** Every metric's mean over folds, by name. */
  means: Record<string, number>
}

/** The full result of a search. */
export interface SearchResult<P, M> {
  metric: string
  direction: 'higher' | 'lower'
  /** One row per candidate, in the order tried. */
  rows: SearchRow<P>[]
  /** The selection metric per candidate and fold, [candidates, folds]. */
  scores: Tensor
  best: SearchRow<P>
  /** Each candidate's full cross-validation (fold models, predictions, traces). */
  validations: CrossValidation<M>[]
  /** The best candidate refitted on all the data (absent when `refit` is false). */
  model?: M
}

/** A search: how to make an estimator from parameters, the candidates, and the metric that selects. */
export interface Search<P, M> {
  readonly name: string
  readonly make: (params: P) => Fittable<unknown, M>
  /** The candidates to try; random searches draw them from the stream. */
  candidates(s?: Stream): P[]
  /** The metric that selects the best candidate (the first metric). */
  readonly metric: AnyMetric
  /** Further metrics recorded for every candidate. */
  readonly metrics: readonly AnyMetric[]
  /** Refit the best candidate on all the data (default true). */
  readonly refit: boolean
  /** Score every candidate by cross-validation with `splitter`, and refit the best. */
  run(data: CrossValidationData<unknown>, splitter: Splitter, s?: Stream): SearchResult<P, M>
  /** The search as an estimator whose fitted model is the refitted best model plus the search results. */
  estimator(splitter: Splitter): Fittable<unknown, SearchModel<P, M>>
}

/** A fitted search: the best model's capabilities, the results table and the chosen parameters. */
export type SearchModel<P, M> = M & { readonly search: SearchResult<P, M>; readonly params: P }

function ranks(means: number[], direction: 'higher' | 'lower'): number[] {
  const sign = direction === 'higher' ? -1 : 1
  return means.map((m) => 1 + means.filter((o) => sign * o < sign * m || (Number.isNaN(m) && !Number.isNaN(o))).length)
}

function makeSearch<P, M>(
  name: string,
  make: (params: P) => Fittable<unknown, M>,
  candidates: (s?: Stream) => P[],
  metrics: readonly AnyMetric[],
  refit: boolean,
): Search<P, M> {
  if (metrics.length === 0) throw new Error(`${name}: needs at least one metric`)
  const metric = metrics[0]
  const search: Search<P, M> = {
    name,
    make,
    candidates,
    metric,
    metrics,
    refit,
    run(data, splitter, s) {
      const list = candidates(s?.child('candidates'))
      // Every candidate sees the same splits and fold streams (common random numbers).
      const validations = list.map((params) =>
        crossValidate(make(params) as Fittable<unknown, never>, data, splitter, metrics as readonly AnyMetric[], {
          stream: s?.child('cv'),
        }),
      ) as unknown as CrossValidation<M>[]
      const means = validations.map((v) => v.mean[metric.name])
      const rank = ranks(means, metric.direction)
      const rows: SearchRow<P>[] = list.map((params, k) => ({
        index: k,
        params,
        scores: validations[k].scores[metric.name],
        mean: means[k],
        std: validations[k].std[metric.name],
        rank: rank[k],
        fitMs: validations[k].folds.reduce((a, f) => a + f.fitMs, 0),
        means: validations[k].mean,
      }))
      const best = rows.reduce((a, b) => (b.rank < a.rank ? b : a))
      const folds = validations[0]?.folds.length ?? 0
      const scores = new Float64Array(rows.length * folds)
      rows.forEach((r, k) => r.scores.data.forEach((v, f) => (scores[k * folds + f] = v)))
      const result: SearchResult<P, M> = {
        metric: metric.name,
        direction: metric.direction,
        rows,
        scores: fromData(scores, [rows.length, folds]),
        best,
        validations,
      }
      if (refit) result.model = make(best.params).fit(data, { stream: s?.child('refit') })
      return result
    },
    estimator(splitter) {
      return {
        name: `${name} with ${splitter.name}`,
        fit(data, options: FitOptions = {}) {
          const result = search.run(data, splitter, options.stream)
          if (!result.model) throw new Error(`${name}: an estimator needs refit`)
          return { ...(result.model as object), search: result, params: result.best.params } as SearchModel<P, M>
        },
      }
    },
  }
  return search
}

/** Every combination of a grid, the last parameter varying fastest. */
function product<S extends GridSpace>(space: S): GridParams<S>[] {
  const keys = Object.keys(space)
  let out: Record<string, unknown>[] = [{}]
  for (const k of keys) out = out.flatMap((p) => space[k].map((v) => ({ ...p, [k]: v })))
  return out as GridParams<S>[]
}

/**
 * Exhaustive grid search: every combination of `space`'s values (the last parameter varying fastest), each scored by
 * cross-validation on the first of `metrics`; the best mean wins. The search keeps the full results table.
 *
 * @example
 * const search = gridSearch((p) => pipeline(standardScaler(), logisticRegression({ l2: p.l2 })), { l2: [0.01, 0.1, 1] }, { metrics: [logLoss] })
 * search.run(data, kFold({ k: 5 })).best.params // { l2: … }
 */
export function gridSearch<const S extends GridSpace, M>(
  make: (params: GridParams<S>) => Fittable<never, M>,
  space: S,
  { metrics, refit = true }: { metrics: readonly AnyMetric[]; refit?: boolean },
): Search<GridParams<S>, M> {
  return makeSearch(
    'grid-search',
    make as (p: GridParams<S>) => Fittable<unknown, M>,
    () => product(space),
    metrics,
    refit,
  )
}

/**
 * Random search (Bergstra and Bengio, 2012, "Random search for hyper-parameter optimization", JMLR 13): `iterations`
 * candidates drawn from `space` (lists uniformly, samplers from the stream `s.child('candidate', k)`), scored as in
 * `gridSearch`. Needs a stream.
 */
export function randomSearch<const S extends RandomSpace, M>(
  make: (params: RandomParams<S>) => Fittable<never, M>,
  space: S,
  { metrics, iterations = 10, refit = true }: { metrics: readonly AnyMetric[]; iterations?: number; refit?: boolean },
): Search<RandomParams<S>, M> {
  const candidates = (s?: Stream) => {
    if (!s) throw new Error('randomSearch: needs a stream')
    return Array.from({ length: iterations }, (_, k) => {
      const c = s.child('candidate', k)
      const p: Record<string, unknown> = {}
      for (const [name, spec] of Object.entries(space)) {
        p[name] = typeof spec === 'function' ? spec(c.child(name)) : spec[c.child(name).int(spec.length)]
      }
      return p as RandomParams<S>
    })
  }
  return makeSearch('random-search', make as (p: RandomParams<S>) => Fittable<unknown, M>, candidates, metrics, refit)
}

/** The result of nested cross-validation. */
export interface NestedCrossValidation<P, M> {
  /** The outer cross-validation of the whole search; each fold's model carries its inner search. */
  outer: CrossValidation<SearchModel<P, M>>
  /** Per outer fold: the chosen parameters, the inner best mean and the outer test score of the selection metric. */
  perFold: { params: P; innerScore: number; outerScore: number }[]
  metric: string
  direction: 'higher' | 'lower'
  /** The nested estimate: the mean outer test score. An honest estimate of the tuned procedure's performance. */
  nestedScore: number
  nestedStd: number
  /** The search run on all the data with the inner splitter. */
  unnested: SearchResult<P, M>
  /** The best inner mean of `unnested`: the usual, optimistic, estimate. */
  unnestedScore: number
  /** How much better the unnested estimate looks: unnested − nested for a higher-is-better metric, else the reverse. */
  optimism: number
}

/**
 * Nested cross-validation (Varma and Simon, 2006, "Bias in error estimation when using cross-validation for model
 * selection", BMC Bioinformatics 7; Cawley and Talbot, 2010, JMLR 11): the `inner` splitter selects hyperparameters
 * with `search` on each `outer` training set, and the outer test sets score the selected, refitted models. Both
 * levels are kept, and the search is also run once on all the data so that the optimism of its best inner score, the
 * unnested estimate, can be shown.
 */
export function nested<P, M>(
  outer: Splitter,
  inner: Splitter,
  search: Search<P, M>,
  data: CrossValidationData<unknown>,
  { stream }: { stream?: Stream } = {},
): NestedCrossValidation<P, M> {
  const cv = crossValidate(
    search.estimator(inner) as Fittable<unknown, never>,
    data,
    outer,
    search.metrics as readonly AnyMetric[],
    {
      stream: stream?.child('outer'),
    },
  ) as unknown as CrossValidation<SearchModel<P, M>>
  const name = search.metric.name
  const perFold = cv.folds.map((f) => ({
    params: f.model.params,
    innerScore: f.model.search.best.mean,
    outerScore: f.metrics[name],
  }))
  const unnested = search.run(data, inner, stream?.child('unnested'))
  const nestedScore = cv.mean[name]
  const sign = search.metric.direction === 'higher' ? 1 : -1
  return {
    outer: cv,
    perFold,
    metric: name,
    direction: search.metric.direction,
    nestedScore,
    nestedStd: cv.std[name],
    unnested,
    unnestedScore: unnested.best.mean,
    optimism: sign * (unnested.best.mean - nestedScore),
  }
}
