import { kmeansPlusPlus } from 'aifn/numerics/neighbours'
import { stream } from 'aifn/foundation/random'
import { fromData, toRows } from 'aifn/foundation/tensor'
import { rng } from './index'

export type Point = [number, number]
export type CentreInit = 'random' | 'kmeans++'

export const CENTRE_INIT_OPTIONS = [
  { value: 'random', label: 'random' },
  { value: 'kmeans++', label: 'k-means++' },
] as const satisfies readonly { value: CentreInit; label: string }[]

export const d2 = (a: Point, b: Point) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

/**
 * k starting centres, chosen from the data, backed by aifn/numerics/neighbours for kmeans++.
 */
export function initialCentres(points: Point[], k: number, init: CentreInit, seed: number): Point[] {
  if (init === 'kmeans++' && points.length >= k) {
    const flat = Float64Array.from(points.flat())
    const x = fromData(flat, [points.length, 2])
    const s = stream(seed)
    const centres = kmeansPlusPlus(s, x, k).centroids
    return toRows(centres) as Point[]
  }
  const r = rng(seed)
  const picked = new Set<number>()
  while (picked.size < Math.min(k, points.length)) picked.add(Math.floor(r.uniform() * points.length))
  return [...picked].map((i) => points[i])
}
