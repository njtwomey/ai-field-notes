import { useCallback } from 'react'
import { useTheme } from '@/components/theme-provider'
import { diverging, interpolateColors, sequential } from './palette'

/**
 * The colour at fraction t ∈ [0, 1] of a sequential or diverging scale, in the current theme: the same colour a Heatmap
 * with that `scale` gives the value at t of its range. For marks that encode an ordered value, e.g. class k of K at
 * t = k/(K − 1), so that bars, lines and points match the heatmap cells.
 */
export function useScaleColor(scale: 'sequential' | 'diverging'): (t: number) => string {
  const { resolved: mode } = useTheme()
  return useCallback(
    (t: number) => interpolateColors(scale === 'diverging' ? diverging(mode) : sequential, t),
    [scale, mode],
  )
}
