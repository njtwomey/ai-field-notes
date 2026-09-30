/**
 * Metrics of ordinal classification, part of `aifn/learning/metrics`: they take ordered labels and scores and name no
 * model. Ordinal mean absolute error, its macro average over classes, accuracy within a tolerance, and Cohen's kappa
 * with quadratic weights.
 */

import { confusionMatrix } from './confusion'
import { classesOf, defineMetric, encodeLabels, labelList, nonEmpty, sameLength, type Label, type Labels } from './core'
import { kappaFromTable } from './classification'

// ── Ordinal ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Class indices 0 … K − 1 of the truth and prediction in the ordinal order `labels` (default the sorted labels). */
function ordinalIndices(yTrue: Labels, yPred: Labels, labels?: readonly Label[]) {
  const t = labelList(yTrue)
  const p = labelList(yPred)
  sameLength(t, p, 'ordinal metric')
  nonEmpty(t.length, 'ordinal metric')
  const classes = labels ? [...labels] : classesOf(t, p)
  const ti = encodeLabels(t, classes)
  const pi = encodeLabels(p, classes)
  if (ti.includes(-1) || pi.includes(-1)) throw new Error('metrics: ordinal metric: a label is missing from `labels`')
  return { ti, pi, classes }
}

/**
 * Ordinal mean absolute error on the class index, (1/n) Σ |ĵᵢ − jᵢ|, with classes numbered in the order `labels`
 * (default the sorted labels, so the steps between neighbouring classes cost 1) (ordinal-classification-metrics).
 */
export const ordinalMeanAbsoluteError = defineMetric(
  {
    key: 'ordinalMeanAbsoluteError',
    name: 'Ordinal mean absolute error',
    inputs: 'labels',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[] } = {}): number => {
    const { ti, pi } = ordinalIndices(yTrue, yPred, options.labels)
    let s = 0
    for (let i = 0; i < ti.length; i++) s += Math.abs(ti[i] - pi[i])
    return s / ti.length
  },
)

/**
 * Macro-averaged ordinal MAE (Baccianella et al. 2009): the MAE within each true class, averaged over the classes
 * present in `yTrue`.
 */
export const macroMeanAbsoluteError = defineMetric(
  {
    key: 'macroMeanAbsoluteError',
    name: 'Macro-averaged MAE',
    inputs: 'labels',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[] } = {}): number => {
    const { ti, pi, classes } = ordinalIndices(yTrue, yPred, options.labels)
    const sums = new Float64Array(classes.length)
    const counts = new Float64Array(classes.length)
    for (let i = 0; i < ti.length; i++) {
      sums[ti[i]] += Math.abs(ti[i] - pi[i])
      counts[ti[i]]++
    }
    let s = 0
    let K = 0
    for (let k = 0; k < classes.length; k++)
      if (counts[k] > 0) {
        s += sums[k] / counts[k]
        K++
      }
    return s / K
  },
)

/** Accuracy within `tolerance` classes (default 1): the fraction with |ĵ − j| ≤ tolerance on the class index. */
export const withinToleranceAccuracy = defineMetric(
  {
    key: 'withinToleranceAccuracy',
    name: 'Accuracy within one class',
    inputs: 'labels',
    direction: 'higher',
    range: [0, 1],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[]; tolerance?: number } = {}): number => {
    const { ti, pi } = ordinalIndices(yTrue, yPred, options.labels)
    const tol = options.tolerance ?? 1
    let hit = 0
    for (let i = 0; i < ti.length; i++) if (Math.abs(ti[i] - pi[i]) <= tol) hit++
    return hit / ti.length
  },
)

/** Quadratic weighted κ: Cohen's κ with weights (j − k)² on the ordinal class index (ordinal-classification-metrics). */
export const quadraticWeightedKappa = defineMetric(
  {
    key: 'quadraticWeightedKappa',
    name: 'Quadratic weighted κ',
    inputs: 'labels',
    direction: 'higher',
    range: [-1, 1],
    notes: ['ordinal-classification-metrics'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[] } = {}): number =>
    kappaFromTable(confusionMatrix(yTrue, yPred, { labels: options.labels }).matrix, 'quadratic'),
)
