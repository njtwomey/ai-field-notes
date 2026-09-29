/**
 * Data colours. The single source is design/palette.json, shared with Python.
 *
 * Rules (from the palette validation):
 * - Categorical slots are assigned in fixed order by entity, never by rank, never cycled.
 * - Scatter-type charts use at most three categorical slots unless a second encoding (marker shape) is present.
 * - Sequential = one hue light→dark. Diverging = two hues with a neutral midpoint.
 */
import palette from '@design/palette.json'

export type Mode = 'light' | 'dark'

export const MARKER_SHAPES = ['circle', 'rect', 'triangle', 'diamond', 'pin', 'roundRect', 'arrow', 'circle'] as const

export function categorical(mode: Mode): string[] {
  return palette.categorical[mode]
}

export function seriesColor(mode: Mode, slot: number): string {
  const colours = palette.categorical[mode]
  if (slot < 0 || slot >= colours.length) throw new Error(`categorical slot ${slot} out of range (max 8 series)`)
  return colours[slot]
}

export const sequential: string[] = palette.sequential

/** The red, green and blue channels of `#rrggbb` or `rgb(r, g, b)` / `rgba(…)`. */
const rgb = (color: string) =>
  color.startsWith('#')
    ? [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16))
    : (color.match(/[\d.]+/g) ?? ['0', '0', '0']).slice(0, 3).map(Number)

/**
 * `color` blended towards `base` (usually the surface), keeping the fraction `t` of `color`: an opaque muted colour,
 * which tiles without the seams that overlapping translucent cells show.
 */
export function mute(color: string, base: string, t: number): string {
  const [a, b] = [rgb(color), rgb(base)]
  return `rgb(${a.map((c, j) => Math.round(b[j] + (c - b[j]) * t)).join(', ')})`
}

/**
 * The colour at fraction t ∈ [0, 1] along evenly spaced colour stops, interpolated linearly in RGB as ECharts'
 * continuous visualMap does, so a mark coloured here matches a heatmap cell of the same value.
 */
export function interpolateColors(stops: string[], t: number): string {
  const u = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1) * (stops.length - 1)
  const i = Math.min(Math.floor(u), stops.length - 2)
  const [a, b] = [rgb(stops[i]), rgb(stops[i + 1])]
  const mix = a.map((c, j) => Math.round(c + (b[j] - c) * (u - i)))
  return `rgb(${mix.join(', ')})`
}

export function diverging(mode: Mode): string[] {
  return [...palette.diverging.negative, palette.diverging.midpoint[mode], ...palette.diverging.positive]
}

export function chrome(mode: Mode) {
  return palette.chrome[mode]
}
