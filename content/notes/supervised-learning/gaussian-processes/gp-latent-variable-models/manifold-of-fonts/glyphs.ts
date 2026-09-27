/**
 * Synthetic "fonts" for the toy glyph manifold. A font draws two glyphs, an "o" and an "I", as closed polylines whose
 * vertices are in correspondence across fonts: vertex i of every font's outer "o" contour sits at the same angle, and
 * the twelve vertices of the "I" always trace the same corners. This is the toy counterpart of the universal
 * parameterisation in Campbell and Kautz (2014): every font becomes one fixed-length vector.
 */
import { rng } from '@/lib/math'

export const O_POINTS = 32
/** Vertices per font: outer and inner "o" contours and the "I" outline. */
export const VERTICES = 2 * O_POINTS + 12
/** Index of the "I" vertex at the top of its right serif (the handle in the figure). */
export const SERIF_TIP = 2 * O_POINTS + 6

export type Style = {
  /** Width of the "o" relative to its height. */
  width: number
  /** Stroke weight: the thickness of vertical strokes. */
  weight: number
  /** Horizontal shear, x += slant · y. */
  slant: number
  /** Stroke contrast: horizontal strokes are (1 − contrast) times as thick as vertical ones. */
  contrast: number
  /** Serif length each side of the "I" stem; 0 for a sans serif. */
  serif: number
}

export type Font = { name: string; kind: 0 | 1 | 2; style: Style; vector: number[] }

export const KIND_NAMES = ['sans', 'serif', 'italic']

/** The glyph outlines of one style, as a flat vector [x₀, y₀, x₁, y₁, …]. */
export function drawStyle({ width, weight, slant, contrast, serif }: Style): number[] {
  const a = 0.3 * width
  const b = 0.3
  const cy = 0.3
  const thin = weight * (1 - contrast)
  const pts: [number, number][] = []
  for (let i = 0; i < O_POINTS; i++) {
    const t = (2 * Math.PI * i) / O_POINTS
    pts.push([a * Math.cos(t), cy + b * Math.sin(t)])
  }
  for (let i = 0; i < O_POINTS; i++) {
    const t = (2 * Math.PI * i) / O_POINTS
    pts.push([(a - weight) * Math.cos(t), cy + (b - thin) * Math.sin(t)])
  }
  const h = weight / 2
  const H = 0.9
  const st = Math.max(0.8 * thin, 0.02)
  const x0 = 0.72
  const s = serif
  const I: [number, number][] = [
    [-h - s, 0],
    [h + s, 0],
    [h + s, st],
    [h, st],
    [h, H - st],
    [h + s, H - st],
    [h + s, H],
    [-h - s, H],
    [-h - s, H - st],
    [-h, H - st],
    [-h, st],
    [-h - s, st],
  ]
  for (const [x, y] of I) pts.push([x + x0, y])
  return pts.flatMap(([x, y]) => [x + slant * y, y])
}

/** Twenty fonts: eight upright sans, eight upright serif, four italics (two of each). */
export function makeFonts(): Font[] {
  const g = rng(7)
  const u = (lo: number, hi: number) => lo + (hi - lo) * g.uniform()
  return Array.from({ length: 20 }, (_, i): Font => {
    const serifed = i >= 10
    const italic = i % 5 === 4
    const style: Style = {
      width: u(0.85, 1.25),
      weight: u(0.06, 0.16),
      contrast: serifed ? u(0.35, 0.6) : u(0, 0.2),
      serif: serifed ? u(0.08, 0.16) : 0,
      slant: italic ? 0.22 : 0,
    }
    const kind = italic ? 2 : serifed ? 1 : 0
    return { name: `${KIND_NAMES[kind]} ${i + 1}`, kind, style, vector: drawStyle(style) }
  })
}

/** Splits a font vector into its three closed contours, each ending where it starts. */
export function contours(v: number[]): [number, number][][] {
  const pt = (i: number): [number, number] => [v[2 * i], v[2 * i + 1]]
  const ring = (start: number, count: number) => {
    const r = Array.from({ length: count }, (_, k) => pt(start + k))
    return [...r, r[0]]
  }
  return [ring(0, O_POINTS), ring(O_POINTS, O_POINTS), ring(2 * O_POINTS, 12)]
}

/** Stroke contrast read off a generated "o": 1 − (thickness at the top) ÷ (thickness at the side). */
export function measuredContrast(v: number[]): number {
  const top = O_POINTS / 4
  const side = v[0] - v[2 * O_POINTS]
  const topThickness = v[2 * top + 1] - v[2 * (O_POINTS + top) + 1]
  return 1 - topThickness / side
}

/** Serif length read off a generated "I": the underside of the top-right serif, from the stem edge to its end. */
export const measuredSerif = (v: number[]) => v[2 * (SERIF_TIP - 1)] - v[2 * (SERIF_TIP - 2)]
