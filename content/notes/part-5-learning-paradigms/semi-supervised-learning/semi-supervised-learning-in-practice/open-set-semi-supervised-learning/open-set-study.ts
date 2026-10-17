/**
 * The study behind the OpenSetContamination figure: two known classes on the line, a few labels per class, and an
 * unlabelled set in which a share ρ of the points comes from a third class that no label covers. Semi-supervised EM
 * with one Gaussian per known class is run on all unlabelled points, or only on those that a Gaussian novelty filter
 * fitted to the labelled points keeps. Errors are exact, on the two known classes.
 */
import { child, stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'
import { gaussianNoveltyScores } from 'aifn-methods/learning/semi-supervised'
import { semiSupervisedMixtureSteps, type SemiSupervisedMixtureState } from 'aifn-methods/unsupervised/clustering'
import { drawLineSample, drawMixture, ruleError, type LineMixtureProblem } from '../_shared/line-mixtures'

/** The known problem: classes at ∓1.2 with standard deviation 0.6, equal priors. */
export const KNOWN: LineMixtureProblem = { classes: [[[1, -1.2, 0.6]], [[1, 1.2, 0.6]]] }
export const UNSEEN_SD = 0.5
export const UNLABELLED = 300
export const REPEATS = 8
export const RHOS = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
const MAX_STEPS = 300

/** One sample: m labels per known class, (1 − ρ)·U unlabelled known points and ρ·U unseen points. */
export function drawOpenSetSample(seed: number, repeat: number, perClass: number, rho: number, unseenMean: number) {
  const s = child(stream(seed), 'repeat', repeat)
  const unseen = Math.round(rho * UNLABELLED)
  const base = drawLineSample(KNOWN, s, perClass, UNLABELLED - unseen)
  const extra = drawMixture([[1, unseenMean, UNSEEN_SD]], child(s, 'unseen'), unseen)
  return {
    x: [...base.x, ...extra],
    labels: [...base.labels, ...extra.map(() => -1)],
    isUnseen: [...base.x.map(() => false), ...extra.map(() => true)],
  }
}

type Sample = ReturnType<typeof drawOpenSetSample>

/** Which points the novelty filter keeps: every labelled point and the unlabelled points with p-value at least α. */
export function noveltyKeep(sample: Sample, alpha: number): boolean[] {
  const p = gaussianNoveltyScores(
    sample.x.map((v) => [v]),
    sample.labels,
    2,
  ).pValue
  return sample.x.map((_, i) => sample.labels[i] >= 0 || p[i] >= alpha)
}

/** Semi-supervised EM on the kept points (all points without a mask); λ = 0 gives the labels-only fit. */
export function fitEm(sample: Sample, keep: readonly boolean[] | null, lambda = 1): SemiSupervisedMixtureState {
  const idx = sample.x.map((_, i) => i).filter((i) => (keep ? keep[i] : true))
  const alg = semiSupervisedMixtureSteps(
    { x: idx.map((i) => [sample.x[i]]), labels: idx.map((i) => sample.labels[i]), classes: 2 },
    {
      covariance: 'diagonal',
      start: 'per-class',
      regularisation: 0,
      varianceFloor: 0.01,
      objective: 'per-row',
      unlabelledWeight: lambda,
      tolerance: 1e-7,
    },
  )
  return run(alg, undefined, MAX_STEPS)
}

/** Mean exact error over repeats for each contamination share: labels only, all unlabelled, filtered unlabelled. */
export function contaminationCurve(seed: number, perClass: number, alpha: number, unseenMean: number) {
  return RHOS.map((rho) => {
    const sums = { supervised: 0, plain: 0, filtered: 0, unseenRemoved: 0, knownRemoved: 0 }
    for (let r = 0; r < REPEATS; r++) {
      const sample = drawOpenSetSample(seed, r, perClass, rho, unseenMean)
      const keep = noveltyKeep(sample, alpha)
      sums.supervised += ruleError(KNOWN, fitEm(sample, null, 0))
      sums.plain += ruleError(KNOWN, fitEm(sample, null))
      sums.filtered += ruleError(KNOWN, fitEm(sample, keep))
      const unseen = sample.isUnseen.filter(Boolean).length
      const knownUnlabelled = sample.x.filter((_, i) => !sample.isUnseen[i] && sample.labels[i] < 0).length
      sums.unseenRemoved += unseen > 0 ? keep.filter((k, i) => !k && sample.isUnseen[i]).length / unseen : 0
      sums.knownRemoved += keep.filter((k, i) => !k && !sample.isUnseen[i]).length / Math.max(1, knownUnlabelled)
    }
    return {
      rho,
      supervised: sums.supervised / REPEATS,
      plain: sums.plain / REPEATS,
      filtered: sums.filtered / REPEATS,
      unseenRemoved: sums.unseenRemoved / REPEATS,
      knownRemoved: sums.knownRemoved / REPEATS,
    }
  })
}
