/** Plain-array helpers for the neural ODE showcase: trails, arrows and per-run arrow scales. */
import type { Vector } from '@lab/viz'

export type Segment = { from: [number, number]; to: [number, number] }

/**
 * Trails of shown points from frame 0 to `frame`, one list per group: paths are [frames × P × dim] row-major; `a` and
 * `b` pick the drawn coordinates; `group[i]` is point i's group.
 */
export function trails(
  paths: Float64Array,
  P: number,
  dim: number,
  frame: number,
  a: number,
  b: number,
  group: ArrayLike<number>,
  groups: number,
): Segment[][] {
  const out: Segment[][] = Array.from({ length: groups }, () => [])
  for (let f = 0; f < frame; f++)
    for (let i = 0; i < P; i++) {
      const p = (f * P + i) * dim
      const q = ((f + 1) * P + i) * dim
      out[group[i]]?.push({ from: [paths[p + a], paths[p + b]], to: [paths[q + a], paths[q + b]] })
    }
  return out
}

/** Coordinates a and b of every shown point at `frame`, as columns. */
export function at(paths: Float64Array, P: number, dim: number, frame: number, a: number, b: number) {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < P; i++) {
    const p = (frame * P + i) * dim
    x.push(paths[p + a])
    y.push(paths[p + b])
  }
  return { x, y }
}

/** The 95th percentile of arrow lengths over several fields ([g² × 2] each). */
export function typicalLength(fields: readonly Float64Array[]): number {
  const lengths: number[] = []
  for (const f of fields) for (let i = 0; i < f.length / 2; i++) lengths.push(Math.hypot(f[2 * i], f[2 * i + 1]))
  lengths.sort((p, q) => p - q)
  return lengths[Math.floor(0.95 * (lengths.length - 1))] || 1
}

/**
 * Arrows of a field on a grid: `xs` and `ys` the axes (x inner, y outer in the field's row-major order), each arrow
 * scaled by `scale` (data units per unit of field) and capped at 0.9 cells; thin and muted, with small heads, so trails
 * and points read on top.
 */
export function arrows(field: Float64Array, xs: ArrayLike<number>, ys: ArrayLike<number>, scale: number): Vector[] {
  const out: Vector[] = []
  const cell = Math.min(Math.abs(xs[1] - xs[0]), Math.abs(ys[1] - ys[0]))
  for (let i = 0; i < ys.length; i++)
    for (let j = 0; j < xs.length; j++) {
      const k = i * xs.length + j
      let u = field[2 * k] * scale
      let v = field[2 * k + 1] * scale
      const len = Math.hypot(u, v)
      if (len > 0.9 * cell) {
        u *= (0.9 * cell) / len
        v *= (0.9 * cell) / len
      }
      if (len < 1e-9) continue
      out.push({ from: [xs[j], ys[i]], to: [xs[j] + u, ys[i] + v], head: 6, width: 1, muted: true })
    }
  return out
}

/** A row-major g × g array as raster rows. */
export const rows = (v: ArrayLike<number>, g: number, map: (x: number) => number = (x) => x) =>
  Array.from({ length: g }, (_, i) => Array.from({ length: g }, (__, j) => map(v[i * g + j])))

/** Every k-th entry and its index (1-based iterations), for long curves. */
export function thin(values: Float64Array, max = 600): { x: number[]; y: number[] } {
  const stride = Math.max(1, Math.floor(values.length / max))
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < values.length; i += stride) {
    x.push(i)
    y.push(values[i])
  }
  return { x, y }
}
