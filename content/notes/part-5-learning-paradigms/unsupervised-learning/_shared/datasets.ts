/** Seeded toy data sets for the unsupervised-learning widgets. */
import { rng } from '@/lib/math'

export type Point = [number, number]

/** Isotropic Gaussian blobs: `sizes[j]` points around `centres[j]` with standard deviation `sd[j]`. */
export function blobs(
  centres: Point[],
  sizes: number[],
  sd: number[],
  seed: number,
): { points: Point[]; labels: number[] } {
  const r = rng(seed)
  const points: Point[] = []
  const labels: number[] = []
  centres.forEach((c, j) => {
    for (let i = 0; i < sizes[j]; i++) {
      points.push([c[0] + sd[j] * r.normal(), c[1] + sd[j] * r.normal()])
      labels.push(j)
    }
  })
  return { points, labels }
}

/** Two interleaved half circles, the usual test that convex clustering fails. */
export function moons(n: number, noise: number, seed: number): { points: Point[]; labels: number[] } {
  const r = rng(seed)
  const points: Point[] = []
  const labels: number[] = []
  for (let i = 0; i < n; i++) {
    const upper = i < n / 2
    const t = Math.PI * r.uniform()
    const [x, y] = upper ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)]
    points.push([x + noise * r.normal(), y + noise * r.normal()])
    labels.push(upper ? 0 : 1)
  }
  return { points, labels }
}

/** Two concentric rings of radius 1 and `inner`. */
export function rings(n: number, inner: number, noise: number, seed: number): { points: Point[]; labels: number[] } {
  const r = rng(seed)
  const points: Point[] = []
  const labels: number[] = []
  for (let i = 0; i < n; i++) {
    const outer = i < n / 2
    const t = 2 * Math.PI * r.uniform()
    const radius = outer ? 1 : inner
    points.push([radius * Math.cos(t) + noise * r.normal(), radius * Math.sin(t) + noise * r.normal()])
    labels.push(outer ? 0 : 1)
  }
  return { points, labels }
}
