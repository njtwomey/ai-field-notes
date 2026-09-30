/**
 * Metrics meet estimators (plan §5.3): `evaluate(model, data, metrics)` asks each metric for the capability it needs,
 * and the types reject a metric the model cannot serve (a log loss on a model that only decides).
 *
 * A `Metric` here is a small object: a name, the capability it needs, a direction and a function of targets and one
 * model output. `adaptMetric` wraps a metric function of `aifn/metrics` (a function carrying `info`) in this shape,
 * feeding it decisions, scores, class probabilities or expectations according to its declared input; this module does
 * not import `aifn/metrics`, which sits above it. Five common metrics are also defined here for the reference
 * estimators and tests.
 */

import { fromData, type Tensor } from 'aifn/tensor'
import type { Decides, Expects, Predicts, Scores } from './capabilities'
import { asTensor, classProbabilities, meanOf, type Distribution } from './distribution'
import { values } from './util'

/** The model capability a metric reads. */
export type Need = 'decide' | 'score' | 'predictive' | 'expect'

/** What a metric receives for each need: a tensor of decisions, scores or expectations, or the predictive. */
export type OutputFor<N extends Need> = N extends 'predictive' ? Distribution : Tensor

/** A metric: a function of targets and one model output, with its declared input and direction. */
export interface Metric<N extends Need = Need> {
  readonly name: string
  /** The capability the metric needs from a model. */
  readonly needs: N
  /** Whether higher or lower values are better. */
  readonly direction: 'higher' | 'lower'
  /** The metric of targets y [N] and the model's output on the same N inputs. */
  compute(y: Tensor, output: OutputFor<N>): number
}

/** Any metric (the needs of a metric list are read from its elements). */
export type AnyMetric = Metric<'decide'> | Metric<'score'> | Metric<'predictive'> | Metric<'expect'>

/** The capabilities a model must have to serve metrics with needs `N` on inputs `X`. */
export type Requirement<X, N> = ('decide' extends N ? Decides<X, Tensor> : unknown) &
  ('score' extends N ? Scores<X> : unknown) &
  ('predictive' extends N ? Predicts<X, Distribution> : unknown) &
  ('expect' extends N ? Expects<X> : unknown)

/** Define a metric from its name, need, direction and function. */
export function defineMetric<N extends Need>(metric: Metric<N>): Metric<N> {
  return metric
}

/**
 * A metric function of `aifn/metrics`: `(yTrue, prediction, options?) => number` with its metadata in `info` (only the
 * fields used here are listed).
 */
export type MetricFunctionLike = ((yTrue: never, prediction: never) => number) & {
  readonly info: {
    readonly key: string
    readonly inputs: string
    readonly direction: 'higher' | 'lower'
    readonly capability?: 'decide' | 'score' | 'predictive'
  }
}

/**
 * Adapt a metric function of `aifn/metrics` to a `Metric` needing `need` (for the type check). The model output is fed
 * according to the function's `info.inputs`: class probabilities from the predictive for `'probabilities'`
 * ([N] P(y = 1) for a Bernoulli, [N, K] for a categorical), the predictive itself for `'distribution'`, and the
 * tensor of decisions, scores or expectations otherwise. Throws when `need` contradicts `info.capability`.
 *
 * @example evaluate(model, test, [adaptMetric(metrics.auroc, 'score'), adaptMetric(metrics.logLoss, 'predictive')])
 */
export function adaptMetric<N extends Need>(f: MetricFunctionLike, need: N): Metric<N> {
  const { key, inputs, direction, capability } = f.info
  const compatible = capability === undefined || capability === need || (capability === 'decide' && need === 'expect')
  if (!compatible) throw new Error(`adaptMetric: ${key} needs ${capability}, not ${need}`)
  const call = f as unknown as (y: Tensor, p: unknown) => number
  return {
    name: key,
    needs: need,
    direction,
    compute(y, output) {
      if (need === 'predictive' && inputs === 'probabilities') {
        const { probs, n, k } = classProbabilities(output as Distribution)
        const p =
          (output as Distribution).name === 'Bernoulli'
            ? fromData(
                Float64Array.from({ length: n }, (_, i) => probs[2 * i + 1]),
                [n],
              )
            : fromData(probs, [n, k])
        return call(y, p)
      }
      return call(y, output)
    },
  }
}

/** The output of `model` that a metric with need `n` reads, on inputs x. */
export function outputFor(model: unknown, n: Need, x: unknown): Tensor | Distribution {
  const m = model as Record<string, ((x: unknown) => Tensor | Distribution) | undefined>
  const f = m[n]
  if (typeof f !== 'function') throw new Error(`evaluate: the model has no ${n}`)
  return f.call(model, x)
}

/** A cache of model outputs by need, so that several metrics share one call. */
export function outputs(
  model: unknown,
  needs: Iterable<Need>,
  x: unknown,
): Partial<Record<Need, Tensor | Distribution>> {
  const out: Partial<Record<Need, Tensor | Distribution>> = {}
  for (const n of needs) if (!(n in out)) out[n] = outputFor(model, n, x)
  return out
}

/**
 * Evaluate a fitted model on data: each metric's value, keyed by name. The model must have every capability the
 * metrics need (checked by the compiler); each capability is called once.
 *
 * @example evaluate(model, test, [accuracy, logLoss]) // { accuracy: 0.93, 'log-loss': 0.21 }
 */
export function evaluate<X, const Ms extends readonly AnyMetric[]>(
  model: Requirement<X, Ms[number]['needs']>,
  data: { x: X; y: Tensor },
  metrics: Ms,
): Record<Ms[number]['name'], number> {
  const cache = outputs(
    model,
    metrics.map((m) => m.needs),
    data.x,
  )
  const result: Record<string, number> = {}
  for (const m of metrics) result[m.name] = (m.compute as (y: Tensor, o: unknown) => number)(data.y, cache[m.needs])
  return result as Record<Ms[number]['name'], number>
}

// ── Stopgap metrics ──────────────────────────────────────────────────────────────────────────────────────────────────

function pair(y: Tensor, output: Tensor, where: string): [Float64Array, Float64Array] {
  const a = values(y)
  const b = values(output)
  if (a.length !== b.length) throw new Error(`${where}: ${a.length} targets but ${b.length} predictions`)
  return [a, b]
}

/** Fraction of decisions equal to the target. Needs `decide`. */
export const accuracy: Metric<'decide'> = defineMetric({
  name: 'accuracy',
  needs: 'decide',
  direction: 'higher',
  compute(y, decision) {
    const [a, b] = pair(y, decision, 'accuracy')
    let hits = 0
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) hits++
    return hits / a.length
  },
})

/** Mean negative log predictive probability (density) of the targets: −(1/N) Σ log p(yᵢ | xᵢ). Needs `predictive`. */
export const logLoss: Metric<'predictive'> = defineMetric({
  name: 'log-loss',
  needs: 'predictive',
  direction: 'lower',
  compute(y, d) {
    const lp = values(asTensor(d.logProb(y)))
    let s = 0
    for (const v of lp) s -= v
    return s / lp.length
  },
})

/** Mean squared error of the predictive mean (`expect`). */
export const meanSquaredError: Metric<'expect'> = defineMetric({
  name: 'mse',
  needs: 'expect',
  direction: 'lower',
  compute(y, mean) {
    const [a, b] = pair(y, mean, 'meanSquaredError')
    let s = 0
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
    return s / a.length
  },
})

/** Mean absolute error of the predictive mean (`expect`). */
export const meanAbsoluteError: Metric<'expect'> = defineMetric({
  name: 'mae',
  needs: 'expect',
  direction: 'lower',
  compute(y, mean) {
    const [a, b] = pair(y, mean, 'meanAbsoluteError')
    let s = 0
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i])
    return s / a.length
  },
})

/** Coefficient of determination R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)² of the predictive mean (`expect`). */
export const rSquared: Metric<'expect'> = defineMetric({
  name: 'r2',
  needs: 'expect',
  direction: 'higher',
  compute(y, mean) {
    const [a, b] = pair(y, mean, 'rSquared')
    let ybar = 0
    for (const v of a) ybar += v / a.length
    let ss = 0
    let tot = 0
    for (let i = 0; i < a.length; i++) {
      ss += (a[i] - b[i]) ** 2
      tot += (a[i] - ybar) ** 2
    }
    return 1 - ss / tot
  },
})

/** The mean of a predictive, for metrics written against distributions. */
export function predictiveMean(d: Distribution): Tensor {
  return meanOf(d)
}
