/** Starting centres for k-means and Gaussian mixtures: k distinct data points at random, or k-means++. */
import { stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toRows } from 'aifn-compute/foundation/tensor'
import type { Vec2 } from 'aifn-compute/numerics/linalg'
import { kmeansPlusPlus } from 'aifn-compute/numerics/neighbours'

export type CentreInit = 'random' | 'kmeans++'

export const CENTRE_INIT_OPTIONS = [
  { value: 'random', label: 'random' },
  { value: 'kmeans++', label: 'k-means++' },
] as const satisfies readonly { value: CentreInit; label: string }[]

/** k starting centres chosen from the data: distinct points drawn uniformly, or by k-means++ (aifn). */
export function initialCentres(points: Vec2[], k: number, init: CentreInit, seed: number): Vec2[] {
  const s = stream(seed)
  if (init === 'kmeans++' && points.length >= k) {
    const x = fromData(Float64Array.from(points.flat()), [points.length, 2])
    return toRows(kmeansPlusPlus(s, x, k).centroids) as Vec2[]
  }
  const picked = new Set<number>()
  while (picked.size < Math.min(k, points.length)) picked.add(Math.floor(uniform(s) * points.length))
  return [...picked].map((i) => points[i])
}
