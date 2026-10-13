import { child, normals, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Supervised } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, type Splitter } from 'aifn-compute/learning/validate'
import { type TypeITest } from 'aifn-methods/theory/algorithm-comparison'
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

/**
 * The tests of the engine's Type I error study (`TYPE_I_TESTS` of `aifn-methods/theory/algorithm-comparison`), in
 * display order, with their labels and their two-line axis labels (narrow panels have little room per bar).
 */
export const TESTS: readonly { key: TypeITest; label: string; axis: string }[] = [
  { key: 'mcnemar', label: 'McNemar', axis: 'McNemar\nhold-out' },
  { key: 'cvT', label: 't, 10-fold', axis: 't\n10-fold' },
  { key: 'resampledT', label: 't, r×10-fold', axis: 't\nr×10-fold' },
  { key: 'corrected', label: 'corrected t', axis: 'corrected\nt' },
  { key: 'fiveByTwoT', label: '5×2cv t', axis: '5×2cv\nt' },
  { key: 'combinedF', label: '5×2cv F', axis: '5×2cv\nF' },
]
