/** Helpers shared by the clustering widgets (k-means, Gaussian mixtures). */
import { rng } from './index'

export type Point = [number, number]
export type CentreInit = 'random' | 'kmeans++'

export const CENTRE_INIT_OPTIONS = [
  { value: 'random', label: 'random' },
  { value: 'kmeans++', label: 'k-means++' },
] as const satisfies readonly { value: CentreInit; label: string }[]

export const d2 = (a: Point, b: Point) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

/**
 * k starting centres, chosen from the data. `random` picks k distinct points uniformly. `kmeans++` picks each further
 * centre with probability proportional to its squared distance from the nearest centre already chosen.
 */
export function initialCentres(points: Point[], k: number, init: CentreInit, seed: number): Point[] {
  const r = rng(seed)
  if (init === 'random') {
    const picked = new Set<number>()
    while (picked.size < Math.min(k, points.length)) picked.add(Math.floor(r.uniform() * points.length))
    return [...picked].map((i) => points[i])
  }
  const centres = [points[Math.floor(r.uniform() * points.length)]]
  while (centres.length < k) {
    const dist = points.map((p) => Math.min(...centres.map((c) => d2(p, c))))
    let t = r.uniform() * dist.reduce((a, b) => a + b, 0)
    const next = dist.findIndex((d) => (t -= d) <= 0)
    centres.push(points[next === -1 ? points.length - 1 : next])
  }
  return centres
}
