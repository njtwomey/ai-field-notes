import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Supervised } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, repeated, shuffleSplit, type Splitter } from 'aifn-compute/learning/validate'
import { correctedResampledTTest, fiveByTwoCvTest, mcnemarTest, oneSampleTTest } from 'aifn-compute/probability/tests'
import { decisionTree } from 'aifn-methods/learning/trees-and-ensembles'

/**
 * Two learners that are equally good by construction. The inputs are two standard normals and the class is 1 when the
 * radius, plus noise, exceeds its median (a circular boundary, so the problem looks the same after any rotation).
 * Learner A is a depth-4 decision tree on the inputs; learner B is the same tree on the inputs rotated by 45°, plus
 * Gaussian noise of standard deviation `handicap` on each rotated input. With `handicap` 0 the two learners have the
 * same expected accuracy at every training size, so the null hypothesis of every test holds exactly; on any one
 * sample, one of them fits better.
 */

/** The median of the radius of a standard bivariate normal, √(2 ln 2), so the classes are about balanced. */
const MEDIAN_RADIUS = 1.1774
const RADIUS_NOISE = 0.3
const DEPTH = 4
export const learner = decisionTree({ maxDepth: DEPTH })

export type Pair = {
  /** The data as learner A sees them. */
  a: Supervised<Tensor, Tensor>
  /** The same rows as learner B sees them. */
  b: Supervised<Tensor, Tensor>
}

/** n rows of the circle problem, as learners A and B see them. */
export function drawPair(s: Stream, n: number, handicap: number): Pair {
  const x = toFlat(normals(child(s, 'x'), [n, 2]))
  const e = toFlat(normals(child(s, 'e'), [n]))
  const z = toFlat(normals(child(s, 'z'), [n, 2]))
  const y = Float64Array.from({ length: n }, (_, i) =>
    Math.hypot(x[2 * i], x[2 * i + 1]) + RADIUS_NOISE * e[i] > MEDIAN_RADIUS ? 1 : 0,
  )
  const r = Math.SQRT1_2
  const rotated = Float64Array.from({ length: 2 * n }, (_, t) => {
    const i = Math.floor(t / 2)
    const u = t % 2 === 0 ? r * (x[2 * i] - x[2 * i + 1]) : r * (x[2 * i] + x[2 * i + 1])
    return u + handicap * z[t]
  })
  const labels = fromData(y)
  return {
    a: dataset(fromData(Float64Array.from(x), [n, 2]), labels),
    b: dataset(fromData(rotated, [n, 2]), labels),
  }
}

/** Accuracy of A minus accuracy of B on each split; both learners see the same splits. */
export function differences(pair: Pair, splitter: Splitter, s: Stream): number[] {
  const a = toFlat(crossValidate(learner, pair.a, splitter, [accuracy], { stream: s }).scores.accuracy)
  const b = toFlat(crossValidate(learner, pair.b, splitter, [accuracy], { stream: s }).scores.accuracy)
  return Array.from(a, (v, i) => v - b[i])
}

/** McNemar's test of A against B on one random hold-out third of the rows, both trained on the other two thirds. */
export function holdoutMcnemar(pair: Pair, s: Stream) {
  const split = shuffleSplit({ splits: 1, testSize: 1 / 3 })
  const a = crossValidate(learner, pair.a, split, [accuracy], { stream: s })
  const b = crossValidate(learner, pair.b, split, [accuracy], { stream: s })
  const test = Array.from(toFlat(a.splits[0].test))
  const pa = toFlat(a.outOfFold.decide!)
  const pb = toFlat(b.outOfFold.decide!)
  const y = toFlat(pair.a.y)
  return mcnemarTest(
    test.map((i) => pa[i]),
    test.map((i) => pb[i]),
    test.map((i) => y[i]),
  )
}

/** The tests compared, in display order. */
export const TESTS = [
  { key: 'mcnemar', label: 'McNemar' },
  { key: 'cvT', label: 't, 10-fold' },
  { key: 'resampledT', label: 't, r×10-fold' },
  { key: 'corrected', label: 'corrected t' },
  { key: 'fiveByTwoT', label: '5×2cv t' },
  { key: 'combinedF', label: '5×2cv F' },
] as const
export type TestKey = (typeof TESTS)[number]['key']

/** One simulated data set: the p-value of every test, and the naive and corrected p-values after 1, …, r repeats. */
export type Replicate = {
  p: Record<TestKey, number>
  naiveByRepeats: number[]
  correctedByRepeats: number[]
  /** A's minus B's accuracy on fresh data, each fitted to all n rows. */
  trueDifference: number
}

const FOLDS = 10
const FRESH = 2000

/** Every test on one data set of n rows, with r repeats of 10-fold cross-validation. */
export function replicate(s: Stream, n: number, handicap: number, repeats: number): Replicate {
  const pair = drawPair(child(s, 'data'), n, handicap)
  const trainSize = (n * (FOLDS - 1)) / FOLDS
  const testSize = n / FOLDS
  const cv = differences(pair, repeated(kFold({ k: FOLDS, shuffle: true }), repeats), child(s, 'cv'))
  const naiveByRepeats: number[] = []
  const correctedByRepeats: number[] = []
  for (let r = 1; r <= repeats; r++) {
    const d = cv.slice(0, r * FOLDS)
    naiveByRepeats.push(oneSampleTTest(d).pValue)
    correctedByRepeats.push(correctedResampledTTest(d, { trainSize, testSize }).pValue)
  }
  const f = differences(pair, repeated(kFold({ k: 2, shuffle: true }), 5), child(s, '5x2'))
  const five = fiveByTwoCvTest([0, 1, 2, 3, 4].map((i) => [f[2 * i], f[2 * i + 1]]))
  const fresh = drawPair(child(s, 'fresh'), FRESH, handicap)
  const accA = accuracy(fresh.a.y, learner.fit(pair.a).decide(fresh.a.x))
  const accB = accuracy(fresh.b.y, learner.fit(pair.b).decide(fresh.b.x))
  return {
    p: {
      mcnemar: holdoutMcnemar(pair, child(s, 'holdout')).pValue,
      cvT: naiveByRepeats[0],
      resampledT: naiveByRepeats[repeats - 1],
      corrected: correctedByRepeats[repeats - 1],
      fiveByTwoT: five.pValue,
      combinedF: five.combinedF.pValue,
    },
    naiveByRepeats,
    correctedByRepeats,
    trueDifference: accA - accB,
  }
}
