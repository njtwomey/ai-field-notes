import type { SeriesSpec } from 'aifn-render'
import type { Point } from './solver'

/** The plotting window shared by the figures: every point of the example with room for the margins. */
export const BOX = { x: [0, 5] as [number, number], y: [0, 4] as [number, number] }
export const Y_RANGE: [number | undefined, number | undefined] = [BOX.y[0], BOX.y[1]]

/** The part of the line wᵀx + b = c inside the plotting window, or nothing if the line misses it. */
export function clipLine(w: Point, b: number, c: number): { x: number[]; y: number[] } {
  const [x0, x1] = BOX.x
  const [y0, y1] = BOX.y
  const pts: Point[] = []
  const eps = 1e-9
  if (Math.abs(w[1]) > 1e-12) {
    for (const x of BOX.x) {
      const y = (c - b - w[0] * x) / w[1]
      if (y >= y0 - eps && y <= y1 + eps) pts.push([x, y])
    }
  }
  if (Math.abs(w[0]) > 1e-12) {
    for (const y of BOX.y) {
      const x = (c - b - w[1] * y) / w[0]
      if (x >= x0 - eps && x <= x1 + eps) pts.push([x, y])
    }
  }
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1])
  return pts.length >= 2 ? { x: [pts[0][0], pts.at(-1)![0]], y: [pts[0][1], pts.at(-1)![1]] } : { x: [], y: [] }
}

/** Points coloured by class, the boundary wᵀx + b = 0 and the margins wᵀx + b = ±1 (omitted while w = 0). */
export function svmSeries(x: Point[], y: number[], w: Point, b: number): SeriesSpec[] {
  const out: SeriesSpec[] = [
    {
      name: 'points',
      type: 'scatter',
      x: x.map((p) => p[0]),
      y: x.map((p) => p[1]),
      group: y.map((v) => (v > 0 ? 1 : 0)),
      groupNames: ['class −1', 'class +1'],
    },
  ]
  if (Math.hypot(w[0], w[1]) < 1e-9) return out
  out.push(
    { name: 'f = 0', type: 'line', ...clipLine(w, b, 0), emphasis: true },
    { name: 'f = +1', type: 'line', ...clipLine(w, b, 1), slot: 1, dashed: true },
    { name: 'f = −1', type: 'line', ...clipLine(w, b, -1), slot: 0, dashed: true },
  )
  return out
}

/**
 * Each point's slack as a segment to its own margin line, along the normal w: moving x by y ξ w/‖w‖² raises y f(x) by
 * ξ, to exactly 1. Segments of points inside the margin (0 < ξ ≤ 1) and misclassified points (ξ > 1) are separate
 * series; NaN breaks a line between segments.
 */
export function slackSeries(x: Point[], y: number[], w: Point, b: number): SeriesSpec[] {
  const n2 = w[0] * w[0] + w[1] * w[1]
  const inside = { x: [] as number[], y: [] as number[] }
  const across = { x: [] as number[], y: [] as number[] }
  if (n2 < 1e-12) return []
  x.forEach((p, t) => {
    const xi = Math.max(0, 1 - y[t] * (w[0] * p[0] + w[1] * p[1] + b))
    if (xi < 1e-9) return
    const to: Point = [p[0] + (y[t] * xi * w[0]) / n2, p[1] + (y[t] * xi * w[1]) / n2]
    const s = xi > 1 + 1e-9 ? across : inside
    s.x.push(p[0], to[0], NaN)
    s.y.push(p[1], to[1], NaN)
  })
  const out: SeriesSpec[] = []
  if (inside.x.length) out.push({ name: 'slack (inside)', type: 'line', ...inside, slot: 2 })
  if (across.x.length) out.push({ name: 'slack (misclassified)', type: 'line', ...across, slot: 3 })
  return out
}

/** A number as a short fraction when it is one (1/4, −9/4, 5/4), otherwise to three decimals. */
export function exact(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  for (const d of [1, 2, 4, 8, 16, 32, 3, 5, 6, 10]) {
    const n = Math.round(v * d)
    if (Math.abs(v * d - n) < 1e-7 * d) {
      const s = n < 0 ? '−' : ''
      return d === 1 ? `${s}${Math.abs(n)}` : `${s}${Math.abs(n)}/${d}`
    }
  }
  return (v < 0 ? '−' : '') + Math.abs(v).toFixed(3)
}
