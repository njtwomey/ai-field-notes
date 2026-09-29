import { useMemo } from 'react'
import { useScaleColor } from '@/components/viz'

/** The class colour scale of every ordinal widget: diverging, dark blue for the lowest class to dark red for the highest. */
export const CLASS_SCALE = 'diverging' as const

/**
 * The colours of K ordered classes: class k (0-based) sits at k/(K − 1) of the diverging scale, the same colour a
 * class heatmap over [1, K] gives it. Every ordinal widget colours classes with these, so a class has one colour.
 */
export function useClassColors(K: number): string[] {
  const at = useScaleColor(CLASS_SCALE)
  return useMemo(() => Array.from({ length: K }, (_, k) => at(K > 1 ? k / (K - 1) : 0.5)), [at, K])
}
