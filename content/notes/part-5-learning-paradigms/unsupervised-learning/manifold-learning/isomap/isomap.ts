import type { Point } from '../../_shared/datasets'
import { eigSymmetric } from '../../_shared/linalg'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

/**
 * Points along a spiral with radius 0.5 + 2t at angle π/2 + 3πt, for t uniform in [0, 1]. Successive turns are about
 * 1.3 apart. `arc` is each point's arc length from the start, the coordinate a perfect unrolling would recover.
 */
export function spiral(n: number, seed: number): { points: Point[]; t: number[]; arc: number[] } {
  const r = stream(seed)
  const t = Array.from({ length: n }, () => uniform(r)).sort((a, b) => a - b)
  const points = t.map((s) => {
    const angle = Math.PI / 2 + 3 * Math.PI * s
    const radius = 0.5 + 2 * s
    return [radius * Math.cos(angle) + 0.03 * normal(r), radius * Math.sin(angle) + 0.03 * normal(r)] as Point
  })
  // Speed |dp/dt| = sqrt((3π r)² + 2²); integrate it with the midpoint rule on a fine grid.
  const speed = (u: number) => Math.hypot(3 * Math.PI * (0.5 + 2 * u), 2)
  const arc = t.map((s) => {
    const steps = 200
    let total = 0
    for (let i = 0; i < steps; i++) total += (speed(((i + 0.5) * s) / steps) * s) / steps
    return total
  })
  return { points, t, arc }
}

/** Symmetrised k-nearest-neighbour graph as an edge list with Euclidean lengths. */
export function knnEdges(points: Point[], k: number): [number, number, number][] {
  const n = points.length
  const dist = (i: number, j: number) => Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
  const keep = new Set<string>()
  const edges: [number, number, number][] = []
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => dist(i, a) - dist(i, b))
      .slice(0, k)
    for (const j of order) {
      const key = i < j ? `${i}-${j}` : `${j}-${i}`
      if (!keep.has(key)) {
        keep.add(key)
        edges.push([i, j, dist(i, j)])
      }
    }
  }
  return edges
}

/** All-pairs shortest paths by Floyd–Warshall; unreachable pairs stay Infinity. */
export function geodesics(n: number, edges: [number, number, number][]): number[][] {
  const d = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 0 : Infinity)))
  for (const [i, j, w] of edges) d[i][j] = d[j][i] = Math.min(d[i][j], w)
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++) {
      const dik = d[i][k]
      if (dik === Infinity) continue
      for (let j = 0; j < n; j++) if (dik + d[k][j] < d[i][j]) d[i][j] = dik + d[k][j]
    }
  return d
}

/** Classical MDS: double-centre the squared distances and take the top eigenvectors scaled by √λ. */
export function classicalMds(d: number[][], dims: number): { coords: number[][]; values: number[] } {
  const n = d.length
  const sq = d.map((row) => row.map((v) => v * v))
  const rowMean = sq.map((row) => row.reduce((a, b) => a + b, 0) / n)
  const all = rowMean.reduce((a, b) => a + b, 0) / n
  const b = sq.map((row, i) => row.map((v, j) => -0.5 * (v - rowMean[i] - rowMean[j] + all)))
  const { values, vectors } = eigSymmetric(b)
  const coords = Array.from({ length: n }, (_, i) =>
    Array.from({ length: dims }, (_, k) => vectors[k][i] * Math.sqrt(Math.max(values[k], 0))),
  )
  return { coords, values }
}
