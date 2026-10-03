/** Small 2-D helpers shared by the approximate nearest-neighbour widgets. */
import { rng } from '@/lib/math'

export type P = [number, number]

export const dist2 = (a: P, b: P) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

/** Index of the point in `pts` nearest to `q`. */
export function nearest(pts: P[], q: P): number {
  let best = 0
  let bestD = Infinity
  pts.forEach((p, i) => {
    const d = dist2(p, q)
    if (d < bestD) {
      bestD = d
      best = i
    }
  })
  return best
}

/** Indices of `pts` sorted by distance to `q`. */
export function rankBy(pts: P[], q: P): number[] {
  return pts
    .map((p, i) => [dist2(p, q), i] as const)
    .sort((a, b) => a[0] - b[0])
    .map(([, i]) => i)
}

/** `n` points from a mixture of `clusters` round Gaussian blobs with centres uniform in [-3, 3]². */
export function blobs(n: number, clusters: number, spread: number, seed: number): P[] {
  const g = rng(seed)
  const centres = Array.from({ length: clusters }, (): P => [6 * g.uniform() - 3, 6 * g.uniform() - 3])
  return Array.from({ length: n }, (): P => {
    const c = centres[Math.floor(g.uniform() * clusters)]
    return [c[0] + spread * g.normal(), c[1] + spread * g.normal()]
  })
}

/** Lloyd's algorithm in 2-D from k-means++-style farthest-first seeding; returns centres and assignments. */
export function kmeans2d(pts: P[], k: number, seed: number, iters = 25): { centres: P[]; assign: number[] } {
  const g = rng(seed)
  const centres: P[] = [pts[Math.floor(g.uniform() * pts.length)]]
  while (centres.length < k) {
    const d = pts.map((p) => Math.min(...centres.map((c) => dist2(p, c))))
    let t = g.uniform() * d.reduce((a, b) => a + b, 0)
    const next = d.findIndex((v) => (t -= v) <= 0)
    centres.push(pts[next === -1 ? pts.length - 1 : next])
  }
  let assign = pts.map((p) => nearest(centres, p))
  for (let it = 0; it < iters; it++) {
    const sum = centres.map(() => [0, 0, 0])
    pts.forEach((p, i) => {
      const s = sum[assign[i]]
      s[0] += p[0]
      s[1] += p[1]
      s[2] += 1
    })
    sum.forEach((s, j) => {
      if (s[2] > 0) centres[j] = [s[0] / s[2], s[1] / s[2]]
    })
    const next = pts.map((p) => nearest(centres, p))
    if (next.every((a, i) => a === assign[i])) break
    assign = next
  }
  return { centres, assign }
}

/** Lloyd's algorithm on scalars (a 1-D codebook of size k), seeded at evenly spaced quantiles. Returns sorted centres. */
export function kmeans1d(values: number[], k: number, iters = 40): number[] {
  const sorted = [...values].sort((a, b) => a - b)
  let centres = Array.from({ length: k }, (_, j) => sorted[Math.floor(((j + 0.5) / k) * sorted.length)])
  for (let it = 0; it < iters; it++) {
    const sum = centres.map(() => 0)
    const count = centres.map(() => 0)
    for (const v of sorted) {
      const j = nearest1d(centres, v)
      sum[j] += v
      count[j] += 1
    }
    centres = centres.map((c, j) => (count[j] ? sum[j] / count[j] : c)).sort((a, b) => a - b)
  }
  return centres
}

export function nearest1d(centres: number[], v: number): number {
  let best = 0
  for (let j = 1; j < centres.length; j++) if (Math.abs(centres[j] - v) < Math.abs(centres[best] - v)) best = j
  return best
}
