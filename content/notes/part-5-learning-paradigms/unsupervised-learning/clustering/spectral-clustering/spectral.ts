import { spectralClustering, kmeans as aifnKmeans } from 'aifn-methods/unsupervised/clustering'
import { dataset } from 'aifn/learning/estimators'
import { fromData, toFlat, toRows } from 'aifn/foundation/tensor'
import type { Vec2 as Point } from 'aifn/numerics/linalg'

/** Lloyd's algorithm from k-means++ starts, backed by aifn-methods. */
export function kmeans(points: Point[], k: number, seeds = 5): number[] {
  const flat = Float64Array.from(points.flat())
  const x = fromData(flat, [points.length, 2])
  const model = aifnKmeans({ k, restarts: seeds }).fit(dataset(x))
  return Array.from(toFlat(model.decide(x)))
}

/**
 * Ng–Jordan–Weiss spectral clustering into two clusters with a Gaussian similarity of width sigma,
 * backed by aifn-methods.
 */
export function spectralTwo(points: Point[], sigma: number) {
  const flat = Float64Array.from(points.flat())
  const x = fromData(flat, [points.length, 2])
  const model = spectralClustering({
    k: 2,
    affinity: { kind: 'rbf', lengthscale: sigma },
    normaliseRows: true,
  }).fit(dataset(x))
  const embedding = toRows(model.embedding) as Point[]
  const labels = Array.from(toFlat(model.labels))
  const eigenvalues = Array.from(toFlat(model.eigenvalues))
  return {
    embedding,
    labels,
    laplacianEigenvalues: eigenvalues.slice(0, 4).map((e) => 1 - e),
  }
}
