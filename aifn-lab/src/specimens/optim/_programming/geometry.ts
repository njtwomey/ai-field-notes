/** Plane geometry for the programming specimens: polygons of 2-D constraint sets, and lines clipped to a view box. */

import type { Segment } from '@lab/viz'

export type Pt = [number, number]

/** The vertices of {x ∈ ℝ² : A x ≤ b}, counter-clockwise (empty when the set is empty or has no vertex). */
export function polygonOf(A: readonly (readonly number[])[], b: readonly number[]): Pt[] {
  const pts: Pt[] = []
  const feasible = (p: Pt) => A.every((a, i) => a[0] * p[0] + a[1] * p[1] <= b[i] + 1e-9)
  for (let i = 0; i < A.length; i++)
    for (let j = i + 1; j < A.length; j++) {
      const det = A[i][0] * A[j][1] - A[i][1] * A[j][0]
      if (Math.abs(det) < 1e-12) continue
      const p: Pt = [(b[i] * A[j][1] - A[i][1] * b[j]) / det, (A[i][0] * b[j] - b[i] * A[j][0]) / det]
      if (feasible(p) && !pts.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-9)) pts.push(p)
    }
  if (pts.length < 3) return pts
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length
  return pts.sort((p, q) => Math.atan2(p[1] - cy, p[0] - cx) - Math.atan2(q[1] - cy, q[0] - cx))
}

/** A polygon as closed x and y arrays for a line series. */
export function closedPath(pts: readonly Pt[]): { x: number[]; y: number[] } {
  const ring = pts.length ? [...pts, pts[0]] : []
  return { x: ring.map((p) => p[0]), y: ring.map((p) => p[1]) }
}

/** The centroid of a polygon's vertices. */
export function centroid(pts: readonly Pt[]): Pt {
  if (!pts.length) return [0, 0]
  return [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length]
}

/** The line a·x = b clipped to the square box [lo, hi]², or null if it misses the box. */
export function clipLine(a: readonly number[], b: number, [lo, hi]: [number, number]): Segment | null {
  const pts: Pt[] = []
  const inside = (v: number) => v >= lo - 1e-9 && v <= hi + 1e-9
  if (Math.abs(a[1]) > 1e-12)
    for (const x of [lo, hi]) {
      const y = (b - a[0] * x) / a[1]
      if (inside(y)) pts.push([x, y])
    }
  if (Math.abs(a[0]) > 1e-12)
    for (const y of [lo, hi]) {
      const x = (b - a[1] * y) / a[0]
      if (inside(x)) pts.push([x, y])
    }
  if (pts.length < 2) return null
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1])
  return { from: pts[0], to: pts[pts.length - 1] }
}

/** Integer points of a box, for lattice backgrounds. */
export function lattice(lo: number, hi: number): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (let i = Math.ceil(lo); i <= Math.floor(hi); i++)
    for (let j = Math.ceil(lo); j <= Math.floor(hi); j++) {
      x.push(i)
      y.push(j)
    }
  return { x, y }
}
