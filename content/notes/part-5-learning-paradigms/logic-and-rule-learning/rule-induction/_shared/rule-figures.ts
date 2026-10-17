/**
 * Drawing helpers shared by the rule-induction figures: the box a conjunction of thresholds cuts out of a 2-d plot,
 * a prediction grid over the plot, two-moons data as plain rows, and the Titanic table as coded rows.
 */
import { simplifyRule, type Predicate } from 'aifn-compute/learning/explain'
import { moons } from 'aifn-methods/data'
import { titanic } from 'aifn-methods/data/real'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

export type Range = [number, number]

/** The plotting window used by the two-moons figures. */
export const MOONS_X: Range = [-1.6, 2.6]
export const MOONS_Y: Range = [-1.1, 1.6]

/** The four edges of the box a rule of thresholds on features 0 and 1 cuts out of the window. */
export function boxSegments(
  rule: readonly Predicate[],
  xr: Range,
  yr: Range,
): { from: [number, number]; to: [number, number] }[] {
  const b = { x0: xr[0], x1: xr[1], y0: yr[0], y1: yr[1] }
  for (const p of simplifyRule(rule)) {
    if (p.feature === 0) {
      b.x0 = Math.max(b.x0, p.lower)
      b.x1 = Math.min(b.x1, p.upper)
    } else if (p.feature === 1) {
      b.y0 = Math.max(b.y0, p.lower)
      b.y1 = Math.min(b.y1, p.upper)
    }
  }
  if (b.x0 >= b.x1 || b.y0 >= b.y1) return []
  const c: [number, number][] = [
    [b.x0, b.y0],
    [b.x1, b.y0],
    [b.x1, b.y1],
    [b.x0, b.y1],
  ]
  return c.map((from, k) => ({ from, to: c[(k + 1) % 4] }))
}

/** Cell centres of a prediction grid over the window, and the grid's points as rows (row-major, y outer). */
export function predictionGrid(xr: Range, yr: Range, nx: number, ny: number) {
  const gx = Array.from({ length: nx }, (_, i) => xr[0] + ((xr[1] - xr[0]) * i) / (nx - 1))
  const gy = Array.from({ length: ny }, (_, j) => yr[0] + ((yr[1] - yr[0]) * j) / (ny - 1))
  const rows = gy.flatMap((y) => gx.map((x) => [x, y]))
  return { gx, gy, rows }
}

/** Values on the grid's rows as the row-major z of a `Raster`. */
export function asGrid(values: ArrayLike<number>, nx: number, ny: number): number[][] {
  return Array.from({ length: ny }, (_, r) => Array.from({ length: nx }, (_, c) => values[r * nx + c]))
}

/** Two moons (aifn `moons`, seeded) as plain rows and labels. */
export function moonRows(seed: number | string, n: number, noise: number) {
  const d = moons(stream(seed), { n, noise, spacing: 'random' })
  const f = toFlat(d.x)
  const y = Array.from(toFlat(d.y), Number)
  const rows = y.map((_, i) => [f[2 * i], f[2 * i + 1]])
  return { rows, y, x1: rows.map((r) => r[0]), x2: rows.map((r) => r[1]) }
}

/** Display names of the Titanic features and their levels, in the coding of `titanicRows`. */
export const TITANIC_NAMES = ['class', 'sex', 'age'] as const
export const TITANIC_LEVELS: Readonly<Record<number, readonly string[]>> = {
  0: ['1st', '2nd', '3rd', 'crew'],
  1: ['male', 'female'],
  2: ['child', 'adult'],
}

/**
 * The 2201 people aboard the Titanic (aifn `titanic`) as rows of level codes: class (0 1st, 1 2nd, 2 3rd, 3 crew),
 * sex (0 male, 1 female), age (0 child, 1 adult); `y` is 1 for a survivor.
 */
export function titanicRows() {
  const t = titanic().table as unknown as Record<string, ArrayLike<string | number>>
  const classes = TITANIC_LEVELS[0]
  const rows: number[][] = []
  const y: number[] = []
  for (let i = 0; i < t.class.length; i++) {
    rows.push([classes.indexOf(String(t.class[i])), t.sex[i] === 'female' ? 1 : 0, t.age[i] === 'adult' ? 1 : 0])
    y.push(Number(t.survived[i]))
  }
  return { rows, y }
}
