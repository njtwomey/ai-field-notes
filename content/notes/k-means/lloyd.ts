import { d2, initialCentres, type CentreInit, type Point } from '@/lib/math/cluster'

/** Lloyd's algorithm, shared by the step-by-step figure and the elbow curve. */
export type State = { centroids: Point[]; labels: number[]; inertia: number; iteration: number; done: boolean }

export function assign(points: Point[], centroids: Point[]) {
  const labels = points.map((p) => centroids.reduce((best, c, j) => (d2(p, c) < d2(p, centroids[best]) ? j : best), 0))
  const inertia = points.reduce((s, p, i) => s + d2(p, centroids[labels[i]]), 0)
  return { labels, inertia }
}

export function step(points: Point[], s: State): State {
  const k = s.centroids.length
  const centroids = s.centroids.map((c, j) => {
    const members = points.filter((_, i) => s.labels[i] === j)
    if (!members.length) return c
    return [
      members.reduce((a, p) => a + p[0], 0) / members.length,
      members.reduce((a, p) => a + p[1], 0) / members.length,
    ] as Point
  })
  const moved = centroids.some((c, j) => d2(c, s.centroids[j]) > 1e-12)
  const { labels, inertia } = assign(points, centroids)
  return { centroids, labels, inertia, iteration: s.iteration + 1, done: !moved || k === 0 }
}

/** Lloyd's algorithm from one initialisation, run until no centroid moves. */
export function converge(points: Point[], k: number, init: CentreInit, seed: number, maxIter = 100): State {
  const centroids = initialCentres(points, k, init, seed)
  let s: State = { centroids, ...assign(points, centroids), iteration: 0, done: false }
  for (let i = 0; i < maxIter && !s.done; i++) s = step(points, s)
  return s
}
