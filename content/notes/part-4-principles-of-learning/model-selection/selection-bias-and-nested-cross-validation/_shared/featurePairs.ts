import { child, normals, shuffle, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { columns, pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { gridSearch } from 'aifn-compute/learning/validate'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'

/**
 * The selection problem of this sub-chapter. There are `pairs` candidate models; candidate j is a 5-nearest-neighbour
 * classifier that sees only the feature pair (x_{2j}, x_{2j+1}). Every feature is standard normal. Labels are exactly
 * balanced (half of each class, in random order). Only pair 0 carries signal: its two features are shifted by
 * ±separation/(2√2) according to the class, so the two class means of pair 0 lie `separation` apart. With separation 0
 * no candidate can beat accuracy 0.5 on new data.
 */
export interface FeaturePairsOptions {
  /** Rows. */
  n: number
  /** Candidate models, one per feature pair. */
  pairs: number
  /** Distance between the class means of pair 0, in standard deviations; 0 makes the labels pure noise. */
  separation: number
}

/** The feature names of a table with `pairs` pairs: x0, x1, …, x(2·pairs − 1). */
const featureName = (k: number) => `x${k}`

/** A labelled sample of `n` rows from the feature-pairs problem, as a table dataset. */
export function featurePairs(s: Stream, { n, pairs, separation }: FeaturePairsOptions) {
  const p = 2 * pairs
  const labels = shuffle(
    child(s, 'labels'),
    Array.from({ length: n }, (_, i) => (i < Math.floor(n / 2) ? 1 : 0)),
  )
  const noise = toFlat(normals(child(s, 'features'), [n, p]))
  const shift = separation / (2 * Math.SQRT2)
  const table: Record<string, Tensor> = {}
  for (let k = 0; k < p; k++) {
    const signed = k < 2 ? shift : 0
    table[featureName(k)] = fromData(
      Float64Array.from({ length: n }, (_, i) => noise[i * p + k] + (labels[i] === 1 ? signed : -signed)),
    )
  }
  return dataset(table, fromData(Float64Array.from(labels)))
}

/** Candidate j: 5-nearest neighbours on the feature pair j only. */
export function pairCandidate({ j }: { j: number }) {
  return pipeline(
    columns({ [featureName(2 * j)]: 'passthrough', [featureName(2 * j + 1)]: 'passthrough' }, { remainder: 'drop' }),
    kNearestNeighbours({ k: 5 }),
  )
}

/** A grid search over the first `pairs` candidates, selecting by accuracy. */
export function pairSearch(pairs: number) {
  const candidates = space({ j: oneOf(Array.from({ length: pairs }, (_, j) => j)) })
  // The pipeline's input is a table; the search's types expect a matrix, hence the cast.
  return gridSearch(pairCandidate as never, candidates, { metrics: [accuracy] })
}

/** Accuracy of candidate j fitted on `train`, measured on a large fresh `test` sample: its true accuracy, nearly. */
export function trueAccuracy(j: number, train: ReturnType<typeof featurePairs>, test: ReturnType<typeof featurePairs>) {
  const model = (pairCandidate({ j }) as never as { fit: (d: unknown) => { decide: (x: unknown) => Tensor } }).fit(
    train,
  )
  return accuracy(test.y, model.decide(test.x))
}
