import {
  classifierFit,
  classifierLogRatio,
  drawSamples,
  featureRows,
  kdeLogRatio,
  kernelRatio,
  kliepTrace,
  pickCentres,
  ratioErrors,
  ulsif,
  ulsifScores,
  type Shift,
} from '../_shared/ratio'

/** The estimators in the race, in palette-slot order. */
export const METHODS = ['divided KDEs', 'logistic regression', 'KLIEP', 'uLSIF'] as const

/** Sample sizes per frame: geometric from 10 to 1000 points in each sample. */
export const SIZES = Array.from({ length: 19 }, (_, i) => Math.round(10 * 100 ** (i / 18)))

const CENTRES = 50
const CLASSIFIER_CENTRES = 20
const CLASSIFIER_PENALTY = 0.01
const LAMBDAS = [0.001, 0.01, 0.1, 1]

/**
 * Each estimator's log r̂ on the grid xs, and its two errors, for nested samples of every size in SIZES. The kernel
 * methods share the width σ; uLSIF picks λ by cross-validation at each size.
 */
export function raceFrames(seed: number, s: Shift, sigma: number, xs: readonly number[]) {
  const all = drawSamples(seed, SIZES.at(-1)!, SIZES.at(-1)!, s)
  return SIZES.map((n) => {
    const xp = all.xp.slice(0, n)
    const xq = all.xq.slice(0, n)
    const centres = pickCentres(xp, CENTRES)
    const kde = kdeLogRatio(xp, xq, xs)

    const pooled = pickCentres(
      [...xp, ...xq].sort((a, b) => a - b),
      CLASSIFIER_CENTRES,
    )
    const rows = (x: readonly number[]) => featureRows(x, 'kernel', pooled, sigma)
    const w = classifierFit(rows(xp), rows(xq), CLASSIFIER_PENALTY)
    const classifier = classifierLogRatio(rows(xs), w, n, n)

    const kl = kliepTrace(xp, xq, centres, sigma, 80, true)
    const kliep = kernelRatio(kl.alphas.at(-1)!, centres, sigma, xs).map(Math.log)

    const scores = ulsifScores(xp, xq, centres, [sigma], LAMBDAS, 3)[0]
    const lambda = LAMBDAS[scores.indexOf(Math.min(...scores))]
    const ls = kernelRatio(ulsif(xp, xq, centres, sigma, lambda), centres, sigma, xs).map(Math.log)

    const logR = [kde, classifier, kliep, ls]
    return { n, xp, xq, logR, errors: logR.map((l) => ratioErrors(xs, l, s)), lambda }
  })
}
