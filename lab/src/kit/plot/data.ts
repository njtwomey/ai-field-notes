/** Shared data for the Plot kit pages. */
import type { Vec2 } from 'aifn-render/viz'

export const grid = (lo: number, hi: number, n: number) =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

/** Three Gaussian bumps and their weighted sum, a density on the plane. */
export const BUMPS = [
  { at: [-1.5, -1] as Vec2, weight: 1, width: 0.9 },
  { at: [1.6, -0.6] as Vec2, weight: 0.8, width: 0.7 },
  { at: [0.2, 1.7] as Vec2, weight: 0.6, width: 1.1 },
]
export const bump = (b: (typeof BUMPS)[number], x: number, y: number) =>
  b.weight * Math.exp(-((x - b.at[0]) ** 2 + (y - b.at[1]) ** 2) / (2 * b.width ** 2))
export const density2 = (x: number, y: number) => BUMPS.reduce((s, b) => s + bump(b, x, y), 0)
