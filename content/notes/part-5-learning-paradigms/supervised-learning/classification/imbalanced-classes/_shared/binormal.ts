import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'
/**
 * The binormal model shared by the imbalanced-classes figures. A score s is N(0, 1) for negatives and N(d, 1) for
 * positives, and a fraction π of cases is positive. The posterior is then exactly logistic in s,
 * logit P(y = 1 | s) = logit π + d s − d²/2, so every quantity below is in closed form.
 */

export const logit = (p: number) => Math.log(p / (1 - p))
export const expit = (z: number) => 1 / (1 + Math.exp(-z))

/** P(y = 1 | s) at prevalence π and separation d. */
export const posterior = (s: number, pi: number, d: number) => expit(logit(pi) + d * s - (d * d) / 2)

/** The score at which the posterior equals p. */
export const scoreForPosterior = (p: number, pi: number, d: number) => (logit(p) - logit(pi) + (d * d) / 2) / d

/** Class-conditional densities of the score. */
export const negativeDensity = (s: number) => normalPdf(s)
export const positiveDensity = (s: number, d: number) => normalPdf(s - d)

/** True- and false-positive rates when predicting positive for s ≥ t. */
export function rates(t: number, d: number): { tpr: number; fpr: number } {
  return { tpr: 1 - normalCdf(t - d), fpr: 1 - normalCdf(t) }
}

const ratio = (a: number, b: number) => (b > 0 ? a / b : Number.NaN)

/** Population metrics at score threshold t, prevalence π and separation d. */
export function metricsAt(t: number, pi: number, d: number) {
  const { tpr, fpr } = rates(t, d)
  const tp = pi * tpr
  const fn = pi * (1 - tpr)
  const fp = (1 - pi) * fpr
  const tn = (1 - pi) * (1 - fpr)
  const precision = ratio(tp, tp + fp)
  return {
    tpr,
    fpr,
    precision,
    recall: tpr,
    f1: ratio(2 * tp, 2 * tp + fp + fn),
    accuracy: tp + tn,
    balancedAccuracy: (tpr + 1 - fpr) / 2,
    mcc: ratio(tp * tn - fp * fn, Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
    fnRate: fn,
    fpRate: fp,
  }
}

/** AUROC of the binormal model, Φ(d/√2). It does not depend on π. */
export const auroc = (d: number) => normalCdf(d / Math.SQRT2)

/** Average precision, Σ (Rₙ − Rₙ₋₁) Pₙ, on a fine threshold grid from strict to lenient. */
export function averagePrecision(pi: number, d: number, count = 1500): number {
  const hi = d + 6
  const lo = -6
  let ap = 0
  let prevRecall = 0
  for (let i = 0; i < count; i++) {
    const t = hi - ((hi - lo) * i) / (count - 1)
    const { tpr, fpr } = rates(t, d)
    const denom = pi * tpr + (1 - pi) * fpr
    const precision = denom > 0 ? (pi * tpr) / denom : 1
    ap += (tpr - prevRecall) * precision
    prevRecall = tpr
  }
  return ap
}

/** Draw n labelled scores from the binormal model with a seeded generator. */
export function sample(
  n: number,
  pi: number,
  d: number,
  g: { uniform: () => number; normal: () => number },
): { s: number[]; y: number[] } {
  const s: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const positive = g.uniform() < pi
    y.push(positive ? 1 : 0)
    s.push((positive ? d : 0) + g.normal())
  }
  return { s, y }
}
