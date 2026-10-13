/**
 * Shared problems and learners for the cross-validation figures. Everything here composes engine calls: the data come
 * from engine generators, the learners are engine estimators or thin wrappers that hand a P-spline fit to
 * `crossValidate`, and the metrics are engine metrics.
 */
import { child, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { pspline } from 'aifn-compute/numerics/interpolate'
import { linearRegression } from 'aifn-methods/learning/linear'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { drawTrainingSet, targetFunction } from 'aifn-methods/theory'

/** The regression function of every regression figure: $f(x) = \sin(\pi x)$ on $[-1, 1]$. */
export const TARGET = 'sine' as const
export const truth = targetFunction(TARGET)

/** $n$ points with $x$ uniform on $[-1, 1]$ and $y = \sin(\pi x) + \sigma\varepsilon$, as a dataset with $x$ as $n \times 1$. */
export function sineData(s: Stream, n: number, noise: number) {
  const { x, y } = drawTrainingSet(s, TARGET, n, noise)
  return { x, y, data: dataset(fromData(x, [n, 1]), fromData(y)) }
}

/**
 * A large fresh sample from the same generator: the test error of a fitted model on it stands in for the model's true
 * prediction error.
 */
export function freshSample(s: Stream, noise: number, n = 2000) {
  const { x, y } = drawTrainingSet(child(s, 'fresh'), TARGET, n, noise)
  return { x: fromData(x, [n, 1]), y: fromData(y), raw: { x, y } }
}

/**
 * A polynomial of degree $d$ fitted by ridge regression with a small penalty `l2` (default 0.1), which keeps fits to a
 * handful of points stable.
 */
export const polynomial = (degree: number, l2 = 0.1) =>
  pipeline(polynomialFeatures({ degree }), linearRegression({ l2 }))

/** The test mean squared error of a model on a sample. */
export function testError(model: { decide: (x: Tensor) => Tensor }, sample: { x: Tensor; y: Tensor }) {
  return meanSquaredError(sample.y, model.decide(sample.x))
}

const column = (t: Tensor) => {
  const v = Float64Array.from(toFlat(t))
  return fromData(v, [v.length])
}

/**
 * A P-spline smoother with a fixed basis on $[-1, 1]$, as an estimator `crossValidate` can refit. The basis and the
 * penalty do not depend on which points are fitted, so the smoother is the same linear map for every fold, which the
 * leave-one-out shortcut needs.
 */
export function psplineLearner(lambda: number, segments = 15) {
  const options = { lambda, segments, range: [-1, 1] as const }
  return {
    name: 'p-spline',
    fit: (d: { x: Tensor; y: Tensor }) => {
      const fit = pspline(column(d.x), column(d.y), options)
      return { fit, decide: (x: Tensor) => fit.evaluate(column(x)) }
    },
  }
}
