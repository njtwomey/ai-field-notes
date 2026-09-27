/** k-nearest-neighbour distance and local outlier factor (Breunig et al. 2000) for small 2D point sets. */
import { rng } from '@/lib/math'

export type Pt = [number, number]

const dist = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1])

/** Indices and distances of the k nearest points to q, skipping index `self` (a training point's own entry). */
function nearest(points: Pt[], q: Pt, k: number, self = -1): { idx: number[]; d: number[] } {
  const order = points.map((p, i) => ({ i, d: i === self ? Infinity : dist(p, q) }))
  order.sort((a, b) => a.d - b.d)
  const top = order.slice(0, k)
  return { idx: top.map((o) => o.i), d: top.map((o) => o.d) }
}

export type DensityModel = { points: Pt[]; k: number; kdist: number[]; lrd: number[]; knn: number[]; lof: number[] }

/** Local reachability density from a neighbour list: the inverse mean of max(k-distance(o), d(p, o)). */
const reachDensity = (kdist: number[], idx: number[], d: number[]) =>
  1 / Math.max(1e-12, idx.reduce((s, o, j) => s + Math.max(kdist[o], d[j]), 0) / idx.length)

export function fitDensity(points: Pt[], k: number): DensityModel {
  const nbrs = points.map((p, i) => nearest(points, p, k, i))
  const kdist = nbrs.map((n) => n.d[k - 1])
  const lrd = nbrs.map((n) => reachDensity(kdist, n.idx, n.d))
  const lof = nbrs.map((n, i) => n.idx.reduce((s, o) => s + lrd[o], 0) / n.idx.length / lrd[i])
  return { points, k, kdist, lrd, knn: kdist, lof }
}

/** Scores of a new point q against the fitted set: its k-NN distance and its LOF. */
export function scorePoint(m: DensityModel, q: Pt): { knn: number; lof: number } {
  const { idx, d } = nearest(m.points, q, m.k)
  const lrdQ = reachDensity(m.kdist, idx, d)
  return { knn: d[m.k - 1], lof: idx.reduce((s, o) => s + m.lrd[o], 0) / idx.length / lrdQ }
}

/**
 * A tight cluster, a diffuse cluster and three planted outliers: two just outside the tight cluster (local outliers)
 * and one far from both (a global outlier). The diffuse cluster's edge points are farther from their neighbours than
 * the local outliers are, which is what defeats a global distance threshold.
 */
export function twoDensityData(seed = 7): { points: Pt[]; planted: number[] } {
  const g = rng(seed)
  const points: Pt[] = []
  for (let i = 0; i < 80; i++) points.push([0.35 * g.normal(), 0.35 * g.normal()])
  for (let i = 0; i < 60; i++) points.push([5 + 1.4 * g.normal(), 3 + 1.4 * g.normal()])
  const planted: Pt[] = [
    [1.5, -0.9],
    [-1.3, 1.2],
    [8.5, -2],
  ]
  return { points: [...points, ...planted], planted: planted.map((_, j) => points.length + j) }
}
