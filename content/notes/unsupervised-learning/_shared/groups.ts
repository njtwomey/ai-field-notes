import type { XYSeries } from '@/components/viz'
import type { Point } from './datasets'

/** Most clusters given their own colour and marker; the palette has 8 slots and scatter plots should use few. */
export const MAX_COLOURED = 6

/**
 * Scatter series for a clustering. Labels below 0 are noise. Clusters are numbered in order of their first point, so a
 * cluster keeps its colour while a parameter changes. Clusters past `MAX_COLOURED`, singletons and noise share one
 * muted series.
 */
export function clusterSeries(points: Point[], labels: number[], mutedName = 'noise or small'): XYSeries[] {
  const size = new Map<number, number>()
  labels.forEach((l) => l >= 0 && size.set(l, (size.get(l) ?? 0) + 1))
  const slot = new Map<number, number>()
  labels.forEach((l) => {
    if (l >= 0 && (size.get(l) ?? 0) > 1 && !slot.has(l) && slot.size < MAX_COLOURED) slot.set(l, slot.size)
  })
  const coloured = points.flatMap((_, i) => (slot.has(labels[i]) ? [i] : []))
  const rest = points.flatMap((_, i) => (slot.has(labels[i]) ? [] : [i]))
  const series: XYSeries[] = []
  if (coloured.length)
    series.push({
      name: 'clusters',
      type: 'scatter',
      x: coloured.map((i) => points[i][0]),
      y: coloured.map((i) => points[i][1]),
      group: coloured.map((i) => slot.get(labels[i])!),
      groupNames: [...slot.values()].map((s) => `cluster ${s + 1}`),
    })
  if (rest.length)
    series.push({
      name: mutedName,
      type: 'scatter',
      x: rest.map((i) => points[i][0]),
      y: rest.map((i) => points[i][1]),
      muted: true,
    })
  return series
}
