import { dbscan as aifnDbscan, CORE } from 'aifn-methods/unsupervised/clustering'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import { dataset } from 'aifn/learning/estimators'
import type { Point } from '../../_shared/datasets'

export type DbscanResult = { labels: number[]; core: boolean[]; clusters: number }

/** DBSCAN with Euclidean distance, backed by aifn-methods. */
export function dbscan(points: Point[], eps: number, minPts: number): DbscanResult {
  const flat = Float64Array.from(points.flat())
  const x = fromData(flat, [points.length, 2])
  const model = aifnDbscan({ eps, minSamples: minPts }).fit(dataset(x))
  const labels = Array.from(toFlat(model.labels))
  const roles = Array.from(toFlat(model.roles))
  const core = roles.map((r) => r === CORE)
  return { labels, core, clusters: model.clusters }
}
