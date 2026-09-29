/**
 * Level sets of a function sampled on a grid, by marching squares. Segments are chained into polylines so that dashed
 * lines keep their pattern; polylines are joined into one series with NaN breaks.
 */

export type Grid = { xs: number[]; ys: number[]; z: number[][] }
export type Polylines = { x: number[]; y: number[] }

/** Sample f on an nx × ny grid over the ranges. z[j][i] is the value at (xs[i], ys[j]). Non-finite values are skipped. */
export function sampleGrid(
  f: (x: number, y: number) => number,
  xRange: [number, number],
  yRange: [number, number],
  n = 81,
): Grid {
  const xs = Array.from({ length: n }, (_, i) => xRange[0] + ((xRange[1] - xRange[0]) * i) / (n - 1))
  const ys = Array.from({ length: n }, (_, j) => yRange[0] + ((yRange[1] - yRange[0]) * j) / (n - 1))
  return { xs, ys, z: ys.map((y) => xs.map((x) => f(x, y))) }
}

/** Levels at quantiles of the grid's finite values: an even spread of contours for any function. */
export function quantileLevels(grid: Grid, qs = [0.03, 0.1, 0.2, 0.32, 0.45, 0.6, 0.75, 0.9]): number[] {
  const values = grid.z
    .flat()
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  if (values.length === 0) return []
  return qs.map((q) => values[Math.min(values.length - 1, Math.floor(q * values.length))])
}

export function contour(grid: Grid, level: number): Polylines {
  const { xs, ys, z } = grid
  const points = new Map<string, [number, number]>()
  const segments: [string, string][] = []
  const above = (v: number) => v >= level
  // A crossing on the edge between two grid nodes, keyed by the edge so that neighbouring cells share it.
  const cross = (key: string, x0: number, y0: number, v0: number, x1: number, y1: number, v1: number) => {
    if (above(v0) === above(v1)) return null
    if (!points.has(key)) {
      const s = v1 === v0 ? 0.5 : (level - v0) / (v1 - v0)
      points.set(key, [x0 + s * (x1 - x0), y0 + s * (y1 - y0)])
    }
    return key
  }
  for (let j = 0; j < ys.length - 1; j++) {
    for (let i = 0; i < xs.length - 1; i++) {
      const a = z[j][i]
      const b = z[j][i + 1]
      const c = z[j + 1][i + 1]
      const d = z[j + 1][i]
      if (![a, b, c, d].every(Number.isFinite)) continue
      const bottom = cross(`h${i},${j}`, xs[i], ys[j], a, xs[i + 1], ys[j], b)
      const right = cross(`v${i + 1},${j}`, xs[i + 1], ys[j], b, xs[i + 1], ys[j + 1], c)
      const top = cross(`h${i},${j + 1}`, xs[i], ys[j + 1], d, xs[i + 1], ys[j + 1], c)
      const left = cross(`v${i},${j}`, xs[i], ys[j], a, xs[i], ys[j + 1], d)
      const hits = [bottom, right, top, left].filter((k): k is string => k !== null)
      if (hits.length === 2) segments.push([hits[0], hits[1]])
      else if (hits.length === 4) {
        // Saddle cell: the centre value decides which pair of opposite corners is connected.
        const centreAboveLikeA = above((a + b + c + d) / 4) === above(a)
        if (centreAboveLikeA) segments.push([bottom!, right!], [top!, left!])
        else segments.push([bottom!, left!], [right!, top!])
      }
    }
  }
  return chain(segments, points)
}

/** Join segments that share an end into polylines. */
function chain(segments: [string, string][], points: Map<string, [number, number]>): Polylines {
  const at = new Map<string, number[]>()
  segments.forEach(([p, q], s) => {
    at.set(p, [...(at.get(p) ?? []), s])
    at.set(q, [...(at.get(q) ?? []), s])
  })
  const used = new Array<boolean>(segments.length).fill(false)
  const extend = (path: string[]) => {
    for (;;) {
      const end = path[path.length - 1]
      const next = (at.get(end) ?? []).find((s) => !used[s])
      if (next === undefined) return
      used[next] = true
      const [p, q] = segments[next]
      path.push(p === end ? q : p)
    }
  }
  const x: number[] = []
  const y: number[] = []
  segments.forEach(([p, q], s) => {
    if (used[s]) return
    used[s] = true
    const forward = [p, q]
    extend(forward)
    const backward = [p]
    extend(backward)
    for (const key of [...backward.slice(1).reverse(), ...forward]) {
      const [px, py] = points.get(key)!
      x.push(px)
      y.push(py)
    }
    x.push(NaN)
    y.push(NaN)
  })
  return { x, y }
}

/** Several levels joined into one series. */
export function contours(grid: Grid, levels: number[]): Polylines {
  const x: number[] = []
  const y: number[] = []
  for (const level of levels) {
    const c = contour(grid, level)
    x.push(...c.x)
    y.push(...c.y)
  }
  return { x, y }
}
