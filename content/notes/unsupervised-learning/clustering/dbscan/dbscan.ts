import type { Point } from '../../_shared/datasets'

export type DbscanResult = { labels: number[]; core: boolean[]; clusters: number }

/** DBSCAN with Euclidean distance. A point's neighbourhood includes itself. Label −1 is noise. */
export function dbscan(points: Point[], eps: number, minPts: number): DbscanResult {
  const n = points.length
  const e2 = eps * eps
  const neighbours = points.map((p) =>
    points.flatMap((q, j) => ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= e2 ? [j] : [])),
  )
  const core = neighbours.map((nb) => nb.length >= minPts)
  const labels = new Array<number>(n).fill(-1)
  let clusters = 0
  for (let i = 0; i < n; i++) {
    if (!core[i] || labels[i] !== -1) continue
    // Grow a cluster from an unvisited core point: every core point reached expands the frontier, border points do not.
    const queue = [i]
    labels[i] = clusters
    while (queue.length) {
      const p = queue.pop()!
      if (!core[p]) continue
      for (const q of neighbours[p]) {
        if (labels[q] === -1) {
          labels[q] = clusters
          queue.push(q)
        }
      }
    }
    clusters++
  }
  return { labels, core, clusters }
}
