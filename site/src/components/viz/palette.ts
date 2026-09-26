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

export function diverging(mode: Mode): string[] {
  return [...palette.diverging.negative, palette.diverging.midpoint[mode], ...palette.diverging.positive]
}

export function chrome(mode: Mode) {
  return palette.chrome[mode]
}
