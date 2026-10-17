/**
 * The study behind the PretrainFineTuneDistil figure: four ways to use a few labels and many unlabelled points, run
 * on one point cloud for several label counts. (1) Supervised: a random-feature logistic regression on the labelled
 * points. (2) Semi-supervised from scratch: self-training of the same model. (3) Pretrain and fine-tune: a
 * representation learned from all inputs without labels (Laplacian eigenmaps, the spectral embedding that the spectral
 * contrastive loss recovers on an augmentation graph), then a logistic head trained on the labelled points. (4)
 * Distil: the fine-tuned teacher labels every unlabelled point and a fresh random-feature logistic student is trained on
 * those labels, so it predicts new inputs without the transductive embedding.
 */
import { circles, moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { laplacianEigenmaps } from 'aifn-methods/unsupervised/embedding/manifold'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromRows, tensor, toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dataset, takeRows } from 'aifn-compute/learning/estimators'
import { accuracyOf, probabilities, selfTrainingSteps } from 'aifn-methods/learning/semi-supervised'
import { randomFeatureLogistic } from '../_shared/models'

export const POINTS = 200
export const TEST_POINTS = 400
export const LABEL_COUNTS = [1, 2, 4, 8, 16]
export const REPEATS = 4
export const EMBEDDING_DIMS = 2
export const SELF_TRAINING_ROUNDS = 3

export type Problem = 'moons' | 'circles' | 'across'

/** A labelled point cloud: moons, circles, or two blobs whose class cuts across both of them. */
export function pointCloud(problem: Problem, s: Stream, n: number): { x: Tensor; y: number[] } {
  if (problem === 'moons') {
    const d = moons(s, { n, noise: 0.1, spacing: 'random' })
    return { x: d.x, y: toFlat(d.y) }
  }
  if (problem === 'circles') {
    const d = circles(s, { n, noise: 0.06, factor: 0.5 })
    return { x: d.x, y: toFlat(d.y) }
  }
  // Two round blobs at x₁ = ±2.5; the class is the sign of x₂, so every cluster holds both classes.
  const z = toArray(normals(child(s, 'points'), [n, 2])) as number[][]
  const rows = z.map(([a, b], i) => [(i % 2 === 0 ? -2.5 : 2.5) + 0.8 * a, 0.8 * b])
  return { x: fromRows(rows), y: rows.map(([, b]) => (b > 0 ? 1 : 0)) }
}

const STUDENT = randomFeatureLogistic({ components: 40, lengthscale: 0.6, l2: 0.01, seed: 1 })
const SUPERVISED = randomFeatureLogistic({ components: 40, lengthscale: 0.6, l2: 0.01, seed: 1 })
const HEAD = logisticRegression({ l2: 0.01, tolerance: 1e-8 })

/** Mean accuracy per label count of the four pipelines. */
export interface StudyRow {
  supervised: number
  selfTraining: number
  fineTuned: number
  distilled: number
  /** The distilled student on fresh test points (the teacher's embedding cannot embed them). */
  distilledTest: number
}

/** The study's result: one row per label count, and the pretrained embedding of every point with its class. */
export interface Study {
  rows: StudyRow[]
  embedding: number[][]
  y: number[]
}

/** Run the study (under a second). Accuracies are on the unlabelled points unless named otherwise. */
export function runStudy(problem: Problem, seed: number): Study {
  const root = stream(seed)
  const data = pointCloud(problem, child(root, 'data'), POINTS)
  const test = pointCloud(problem, child(root, 'test'), TEST_POINTS)
  // Pretraining sees all inputs and no labels.
  const raw = toArray(
    laplacianEigenmaps({ neighbours: 10, dims: EMBEDDING_DIMS }).fit(dataset(data.x)).embedding,
  ) as number[][]
  // Standardise each coordinate, so that the head's L2 penalty means the same on every axis.
  const mean = raw[0].map((_, j) => raw.reduce((a, r) => a + r[j], 0) / raw.length)
  const sd = raw[0].map((_, j) => Math.sqrt(raw.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / raw.length))
  const z = fromRows(raw.map((r) => r.map((v, j) => (v - mean[j]) / sd[j])))
  const rows = LABEL_COUNTS.map((m) => {
    const sums = { supervised: 0, selfTraining: 0, fineTuned: 0, distilled: 0, distilledTest: 0 }
    for (let r = 0; r < REPEATS; r++) {
      const split = semiSupervisedSplit(child(root, 'split', m, r), data.y, m)
      const L = split.labelled
      const U = split.unlabelled
      const yL = tensor(L.map((i) => data.y[i]))
      const sup = SUPERVISED.fit(dataset(takeRows(data.x, L), yL))
      sums.supervised += accuracyOf(probabilities(sup, data.x, 2), 2, data.y, U)
      const st = trace(
        selfTrainingSteps(
          { x: data.x, labels: split.labels, classes: 2, truth: data.y },
          { base: SUPERVISED, threshold: 0.9 },
        ),
        undefined,
        SELF_TRAINING_ROUNDS,
      )
      sums.selfTraining += st.final.unlabelledAccuracy
      // Fine-tune: a logistic head on the frozen embedding of the labelled points.
      const head = HEAD.fit(dataset(takeRows(z, L), yL))
      const teacher = probabilities(head, z, 2)
      sums.fineTuned += accuracyOf(teacher, 2, data.y, U)
      // Distil: the teacher's hard labels on every point train a student on the raw inputs.
      const taught = Array.from({ length: POINTS }, (_, i) =>
        split.labels[i] >= 0 ? split.labels[i] : teacher[2 * i + 1] > teacher[2 * i] ? 1 : 0,
      )
      const student = STUDENT.fit(dataset(data.x, tensor(taught)))
      sums.distilled += accuracyOf(probabilities(student, data.x, 2), 2, data.y, U)
      sums.distilledTest += accuracyOf(probabilities(student, test.x, 2), 2, test.y)
    }
    return {
      supervised: sums.supervised / REPEATS,
      selfTraining: sums.selfTraining / REPEATS,
      fineTuned: sums.fineTuned / REPEATS,
      distilled: sums.distilled / REPEATS,
      distilledTest: sums.distilledTest / REPEATS,
    }
  })
  const standardised = toArray(z) as number[][]
  return { rows, embedding: standardised, y: data.y }
}
