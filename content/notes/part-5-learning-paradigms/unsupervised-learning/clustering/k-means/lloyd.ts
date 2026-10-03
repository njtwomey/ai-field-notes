import { assignNearest, lloydUpdate, kmeansPlusPlus } from 'aifn/numerics/neighbours'
import { fromData, toFlat, toRows } from 'aifn/foundation/tensor'
import { stream } from 'aifn/foundation/random'
import type { Point, CentreInit } from '@/lib/math/cluster'

export type State = { centroids: Point[]; labels: number[]; inertia: number; iteration: number; done: boolean }

export function assign(points: Point[], centroids: Point[]) {
  const x = fromData(Float64Array.from(points.flat()), [points.length, 2])
  const c = fromData(Float64Array.from(centroids.flat()), [centroids.length, 2])
  const res = assignNearest(x, c)
  return { labels: Array.from(toFlat(res.labels)), inertia: res.inertia }
}

export function step(points: Point[], s: State): State {
  const x = fromData(Float64Array.from(points.flat()), [points.length, 2])
  const c = fromData(Float64Array.from(s.centroids.flat()), [s.centroids.length, 2])
  const labelsTensor = fromData(Int32Array.from(s.labels), [s.labels.length])
  const next = lloydUpdate(x, labelsTensor, c)
  const nextCentroids = toRows(next.centroids) as Point[]
  const nextAssign = assignNearest(x, next.centroids)
  return {
    centroids: nextCentroids,
    labels: Array.from(toFlat(nextAssign.labels)),
    inertia: nextAssign.inertia,
    iteration: s.iteration + 1,
    done: next.shift <= 1e-12,
  }
}

export function converge(points: Point[], k: number, init: CentreInit, seed: number, maxIter = 100): State {
  const x = fromData(Float64Array.from(points.flat()), [points.length, 2])
  const s = stream(`kmeans/${seed}`)
  const c0 = init === 'kmeans++' ? kmeansPlusPlus(s, x, k).centroids : fromData(Float64Array.from(points.slice(0, k).flat()), [k, 2])
  let curr: State = { centroids: toRows(c0) as Point[], ...assign(points, toRows(c0) as Point[]), iteration: 0, done: false }
  for (let i = 0; i < maxIter && !curr.done; i++) curr = step(points, curr)
  return curr
}
