import { d2, initialCentres, type Point } from '@/lib/math/cluster'
import { eigSymmetric } from '../../_shared/linalg'

/** Lloyd's algorithm from k-means++ starts, best of a few seeds. */
export function kmeans(points: Point[], k: number, seeds = 5): number[] {
  let best: { labels: number[]; inertia: number } = { labels: [], inertia: Infinity }
  for (let seed = 0; seed < seeds; seed++) {
    let centres = initialCentres(points, k, 'kmeans++', seed)
    let labels: number[] = []
    for (let it = 0; it < 100; it++) {
      labels = points.map((p) => centres.reduce((b, c, j) => (d2(p, c) < d2(p, centres[b]) ? j : b), 0))
      const next = centres.map((c, j) => {
        const members = points.filter((_, i) => labels[i] === j)
        if (!members.length) return c
        return [
          members.reduce((s, p) => s + p[0], 0) / members.length,
          members.reduce((s, p) => s + p[1], 0) / members.length,
        ] as Point
      })
      const moved = next.some((c, j) => d2(c, centres[j]) > 1e-12)
      centres = next
      if (!moved) break
    }
    const inertia = points.reduce((s, p, i) => s + d2(p, centres[labels[i]]), 0)
    if (inertia < best.inertia) best = { labels, inertia }
  }
  return best.labels
}

/**
 * Ng–Jordan–Weiss spectral clustering into two clusters with a Gaussian similarity of width sigma. The top two
 * eigenvectors of M = D^(-1/2) W D^(-1/2) are the bottom two of the normalised Laplacian I − M. Their rows,
 * normalised to unit length, are clustered by k-means. Jacobi rotations give exact eigenvectors, fast enough for
 * about 120 points.
 */
export function spectralTwo(points: Point[], sigma: number) {
  const w = points.map((p, i) => points.map((q, j) => (i === j ? 0 : Math.exp(-d2(p, q) / (2 * sigma * sigma)))))
  const deg = w.map((row) => row.reduce((a, b) => a + b, 0))
  const inv = deg.map((d) => 1 / Math.sqrt(Math.max(d, 1e-300)))
  const m = w.map((row, i) => row.map((v, j) => v * inv[i] * inv[j]))
  const { values, vectors } = eigSymmetric(m)
  const [u, v] = vectors
  const embedding: Point[] = points.map((_, i) => {
    const r = Math.hypot(u[i], v[i]) || 1
    return [u[i] / r, v[i] / r]
  })
  return { embedding, labels: kmeans(embedding, 2), laplacianEigenvalues: values.slice(0, 4).map((e) => 1 - e) }
}
