import { normalCdf } from 'aifn-compute/numerics/special'
/**
 * The binormal model used by the classification-metric figures: negative scores ~ N(0, 1), positive scores ~ N(d, 1),
 * a fraction π of the population positive, and "predict positive" when the score exceeds a threshold t. Everything is
 * in closed form, so the figures are exact rather than simulated.
 */

export type Counts = { tp: number; fn: number; fp: number; tn: number }

/** True- and false-positive rates at threshold t for separation d. */
export function rates(t: number, d: number): { tpr: number; fpr: number } {
  return { tpr: 1 - normalCdf(t - d), fpr: 1 - normalCdf(t) }
}

/** Expected confusion-matrix counts for n cases at prevalence π. */
export function expectedCounts(t: number, d: number, pi: number, n: number): Counts {
  const { tpr, fpr } = rates(t, d)
  const pos = pi * n
  const neg = n - pos
  return { tp: pos * tpr, fn: pos * (1 - tpr), fp: neg * fpr, tn: neg * (1 - fpr) }
}

const ratio = (a: number, b: number) => (b > 0 ? a / b : Number.NaN)

/** Every threshold metric in the notes, from confusion counts. NaN where a denominator is zero. */
export function metricsFrom({ tp, fn, fp, tn }: Counts) {
  const n = tp + fn + fp + tn
  const recall = ratio(tp, tp + fn)
  const specificity = ratio(tn, tn + fp)
  const precision = ratio(tp, tp + fp)
  const npv = ratio(tn, tn + fn)
  const accuracy = ratio(tp + tn, n)
  const expectedAgreement = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
  return {
    accuracy,
    balancedAccuracy: (recall + specificity) / 2,
    precision,
    recall,
    specificity,
    fpr: 1 - specificity,
    npv,
    f1: ratio(2 * tp, 2 * tp + fp + fn),
    jaccard: ratio(tp, tp + fp + fn),
    mcc: ratio(tp * tn - fp * fn, Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
    kappa: ratio(accuracy - expectedAgreement, 1 - expectedAgreement),
    informedness: recall + specificity - 1,
    markedness: precision + npv - 1,
    selectionRate: ratio(tp + fp, n),
  }
}

export type MetricKey = keyof ReturnType<typeof metricsFrom>

export const METRIC_LABELS: Record<MetricKey, string> = {
  accuracy: 'accuracy',
  balancedAccuracy: 'balanced accuracy',
  precision: 'precision (PPV)',
  recall: 'recall (TPR)',
  specificity: 'specificity (TNR)',
  fpr: 'false-positive rate',
  npv: 'NPV',
  f1: 'F₁',
  jaccard: 'Jaccard',
  mcc: 'MCC',
  kappa: "Cohen's κ",
  informedness: 'informedness',
  markedness: 'markedness',
  selectionRate: 'predicted positive rate',
}

/** AUROC of the binormal model: Φ(d/√2), the probability a positive outscores a negative. */
export const binormalAuc = (d: number) => normalCdf(d / Math.SQRT2)

/** The ROC and precision-recall curves on a threshold grid, from high threshold (strict) to low (lenient). */
export function curves(d: number, pi: number, count = 400) {
  const lo = Math.min(-4, d - 4)
  const hi = Math.max(4, d + 4)
  const thresholds = Array.from({ length: count }, (_, i) => hi - ((hi - lo) * i) / (count - 1))
  const tpr: number[] = []
  const fpr: number[] = []
  const precision: number[] = []
  for (const t of thresholds) {
    const r = rates(t, d)
    tpr.push(r.tpr)
    fpr.push(r.fpr)
    const denom = pi * r.tpr + (1 - pi) * r.fpr
    precision.push(denom > 0 ? (pi * r.tpr) / denom : 1)
  }
  return { thresholds, tpr, fpr, precision }
}

/** Average precision: Σ (Rₙ − Rₙ₋₁) Pₙ over the threshold grid (the step-wise integral, not a trapezoid). */
export function averagePrecision(d: number, pi: number): number {
  const c = curves(d, pi, 4000)
  let ap = 0
  for (let i = 1; i < c.tpr.length; i++) ap += (c.tpr[i] - c.tpr[i - 1]) * c.precision[i]
  return ap
}

/** Equal error rate of the binormal model: where FPR = FNR, at t = d/2, EER = Φ(−d/2). */
export const binormalEer = (d: number) => normalCdf(-d / 2)

/** The curve index nearest a point (fpr, tpr) in ROC space, for dragging an operating point along the curve. */
export function nearestIndex(xs: number[], ys: number[], [x, y]: [number, number]): number {
  let best = 0
  let bestDistance = Infinity
  for (let i = 0; i < xs.length; i++) {
    const dist = (xs[i] - x) ** 2 + (ys[i] - y) ** 2
    if (dist < bestDistance) {
      best = i
      bestDistance = dist
    }
  }
  return best
}
