/**
 * Shared figure helpers for the foundations-of-semi-supervised-learning notes, copied from the self-training and
 * co-training chapter's `_shared/models.ts`: a nonlinear probabilistic classifier composed from engine parts (random
 * Fourier features, then L2-penalised logistic regression), and a prediction grid.
 */
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { randomFourierFeatures } from 'aifn-methods/learning/preprocessing'
import { stream, type Stream } from 'aifn-compute/foundation/random'
import { fromRows, linspace, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Estimator, type Predicts } from 'aifn-compute/learning/estimators'
import type { TrainingSet } from 'aifn-methods/learning/semi-supervised'

/** Options of `randomFeatureLogistic`. */
export interface RandomFeatureLogisticOptions {
  /** Number D of random Fourier features. */
  components: number
  /** Lengthscale of the squared-exponential kernel the features approximate. */
  lengthscale: number
  /** L2 penalty on the logistic weights. */
  l2: number
  /** Seed of the feature draw, fixed so that every fit of the estimator uses the same features. */
  seed: number
  /** Input dimension (default 2). */
  dim?: number
  /** Noise applied to the training set in feature space before each fit (e.g. feature dropout). */
  featureNoise?: (data: TrainingSet, s: Stream) => TrainingSet
}

/**
 * A logistic regression on D fixed random Fourier features of the input: a smooth nonlinear classifier, in effect a
 * one-hidden-layer network whose first layer is frozen at random weights. Fitting takes a few milliseconds for a few
 * hundred points with D = 40.
 */
export function randomFeatureLogistic(options: RandomFeatureLogisticOptions): Estimator<TrainingSet, Predicts<Tensor>> {
  const dim = options.dim ?? 2
  const features = randomFourierFeatures({ components: options.components, lengthscale: options.lengthscale }).fit(
    { x: tensor(new Array(dim).fill(0), [1, dim]) },
    { stream: stream(options.seed) },
  )
  return {
    name: 'random-feature-logistic',
    fit(data, fitOptions) {
      let z: TrainingSet = dataset(features.transform(data.x), data.y)
      if (options.featureNoise) z = options.featureNoise(z, fitOptions?.stream ?? stream(0))
      const model = logisticRegression({ l2: options.l2, tolerance: 1e-8 }).fit(z)
      return { predictive: (x: Tensor) => model.predictive(features.transform(x)) }
    },
  }
}

/** A rectangular grid of prediction points: the axis values and the points row by row (y outer, x inner). */
export function predictionGrid(
  xRange: readonly [number, number],
  yRange: readonly [number, number],
  nx: number,
  ny: number,
): { gx: number[]; gy: number[]; points: Tensor } {
  const gx = toFlat(linspace(xRange[0], xRange[1], nx))
  const gy = toFlat(linspace(yRange[0], yRange[1], ny))
  return { gx, gy, points: fromRows(gy.flatMap((y) => gx.map((x) => [x, y]))) }
}

/** P(class 1) − ½ on a grid from row-major two-class probabilities, as Raster rows. */
export function centredField(probs: Float64Array, nx: number, ny: number): number[][] {
  return Array.from({ length: ny }, (_, r) => Array.from({ length: nx }, (_, c) => probs[2 * (r * nx + c) + 1] - 0.5))
}
