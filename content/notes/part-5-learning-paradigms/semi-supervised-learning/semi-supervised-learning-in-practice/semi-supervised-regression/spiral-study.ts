/**
 * The data and fits behind the SpiralRegression figure: points along a spiral in the plane whose target is their
 * position along the curve, a few labelled points spread along it, and kernel ridge regression against Laplacian
 * regularised least squares on a nearest-neighbour graph of all points.
 */
import { pointAffinity } from 'aifn-compute/graph/propagation'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { fromRows, matmul, toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { laplacianRegularisedLeastSquares } from 'aifn-methods/learning/semi-supervised'
import { laplacian } from 'aifn-compute/graph/matrices'

export const POINTS = 150
export const T_RANGE: [number, number] = [Math.PI, 4 * Math.PI]
const NOISE = 0.04
const GRAPH_SIGMA = 0.15

/** Arc length of the spiral r = t/4 from t = 0 to t. */
const arc = (t: number) => 0.125 * (t * Math.sqrt(t * t + 1) + Math.asinh(t))

/** The point of the spiral at parameter t. */
export const spiralPoint = (t: number): [number, number] => [0.25 * t * Math.cos(t), 0.25 * t * Math.sin(t)]

/** The target at parameter t: the arc-length position along the spiral, scaled to [−1, 1]. */
export function target(t: number): number {
  const a = arc(T_RANGE[0])
  const b = arc(T_RANGE[1])
  return (2 * (arc(t) - a)) / (b - a) - 1
}

/** Points on the spiral (sorted by t) with small noise, and their targets. */
export function spiralData(seed: number) {
  const s = stream(seed)
  const t = toFlat(uniform(child(s, 't'), T_RANGE[0], T_RANGE[1], { shape: [POINTS] })).sort((a, b) => a - b)
  const z = toArray(normals(child(s, 'noise'), [POINTS, 2])) as number[][]
  const xy = t.map((v, i) => {
    const [a, b] = spiralPoint(v)
    return [a + NOISE * z[i][0], b + NOISE * z[i][1]]
  })
  return { t, xy, x: fromRows(xy), y: t.map(target) }
}

/** `count` labelled points spread evenly along the curve (by rank in t). */
export function spreadLabels(count: number): number[] {
  return Array.from({ length: count }, (_, q) =>
    Math.round((0.03 + (0.94 * q) / Math.max(1, count - 1)) * (POINTS - 1)),
  )
}

/** The graph Laplacian of the k-nearest-neighbour graph with heat weights. */
export function spiralLaplacian(x: Tensor, k: number): Tensor {
  return laplacian(pointAffinity(x, { gamma: 1 / (2 * GRAPH_SIGMA ** 2), neighbours: k }))
}

/** LapRLS (γ_I > 0) or kernel ridge regression (γ_I = 0) fitted values at the training points and at extra points. */
export function fitSpiral(
  data: ReturnType<typeof spiralData>,
  laplacian: Tensor,
  labelled: readonly number[],
  options: { lengthscale: number; gammaA: number; gammaI: number; extra?: Tensor },
) {
  const kernel = rbf({ lengthscale: options.lengthscale })
  const K = gram(kernel, data.x)
  const m = laplacianRegularisedLeastSquares(K, laplacian, data.y, labelled, {
    gammaA: options.gammaA,
    gammaI: options.gammaI,
  })
  const fitted = toFlat(m.fitted)
  const extra = options.extra ? toFlat(matmul(gram(kernel, options.extra, data.x), m.alpha)) : []
  const unlabelled = data.y.map((_, i) => i).filter((i) => !labelled.includes(i))
  const rmse = Math.sqrt(unlabelled.reduce((a, i) => a + (fitted[i] - data.y[i]) ** 2, 0) / unlabelled.length)
  return { fitted, extra, rmse }
}
