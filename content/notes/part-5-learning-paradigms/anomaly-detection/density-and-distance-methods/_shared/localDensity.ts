import {
  knnScore,
  localOutlierFactor,
  localOutlierScore,
  type LocalOutlierFactor,
} from 'aifn-methods/unsupervised/anomaly'
import { fromData } from 'aifn/foundation/tensor'
import { stream, normal } from 'aifn/foundation/random'

export type Pt = [number, number]

export type DensityModel = {
  points: Pt[]
  k: number
  kdist: number[]
  lrd: number[]
  knn: number[]
  lof: number[]
  _model: LocalOutlierFactor
}

export function fitDensity(points: Pt[], k: number): DensityModel {
  const flat = Float64Array.from(points.flat())
  const x = fromData(flat, [points.length, 2])
  const model = localOutlierFactor(x, { k })
  const kdist = Array.from(model.kDistance)
  const lrd = Array.from(model.density)
  const lof = Array.from(model.factor)
  return {
    points,
    k,
    kdist,
    lrd,
    knn: kdist,
    lof,
    _model: model,
  }
}

/** Scores of a new point q against the fitted set, backed by aifn-methods. */
export function scorePoint(m: DensityModel, q: Pt): { knn: number; lof: number } {
  const qTensor = fromData(Float64Array.from(q), [1, 2])
  const knn = knnScore(m._model.train, qTensor, { k: m.k })[0]
  const lof = localOutlierScore(m._model, qTensor)[0]
  return { knn, lof }
}

/**
 * A tight cluster, a diffuse cluster and three planted outliers: two just outside the tight cluster (local outliers)
 * and one far from both (a global outlier).
 */
export function twoDensityData(seed = 7): { points: Pt[]; planted: number[] } {
  const s = stream(`local-density/${seed}`)
  const points: Pt[] = []
  for (let i = 0; i < 80; i++) points.push([0.35 * normal(s), 0.35 * normal(s)])
  for (let i = 0; i < 60; i++) points.push([5 + 1.4 * normal(s), 3 + 1.4 * normal(s)])
  const planted: Pt[] = [
    [1.5, -0.9],
    [-1.3, 1.2],
    [8.5, -2],
  ]
  return { points: [...points, ...planted], planted: planted.map((_, j) => points.length + j) }
}
