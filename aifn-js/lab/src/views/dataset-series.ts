import { useMemo } from 'react'
import type { Dataset } from 'aifn-applied/data'
import { toFlat } from 'aifn/foundation/tensor'
import { useScaleColor, type PointsProps } from '@lab/viz'

/** A dataset's scatter as `Points` props: grouped by class, coloured by a ramp over `t`, or plain. */
export type DatasetPoints = Pick<PointsProps, 'name' | 'x' | 'y' | 'group' | 'groupNames' | 'colors' | 'slot'>

/**
 * The scatter of a dataset, as props for a `Points` layer (`<Points {...points} />`): by class (groups), by the
 * coordinate t (a colour ramp), or plain; and, for one feature with a known f, the curve f(x).
 */
export function useDatasetSeries(
  data: Dataset,
  dims: [number, number] | undefined,
  colorBy: 'label' | 't' | 'none' | undefined,
) {
  const ramp = useScaleColor('sequential')
  return useMemo(() => {
    const [n, d] = data.x.shape
    const x = toFlat(data.x)
    const oneD = d === 1
    const [i, j] = dims ?? [0, 1]
    const xs = Array.from({ length: n }, (_, r) => x[r * d + i])
    const ys = oneD ? toFlat(data.y!) : Array.from({ length: n }, (_, r) => x[r * d + j])
    const labelled = data.y !== undefined && data.y.dtype === 'int32'
    const mode = colorBy ?? (labelled ? 'label' : data.t ? 't' : 'none')
    let points: DatasetPoints
    if (mode === 'label' && labelled && !oneD)
      points = { name: data.meta.name, x: xs, y: ys, group: toFlat(data.y!), groupNames: data.meta.labelNames }
    else if (mode === 't' && data.t) {
      const t = toFlat(data.t)
      const lo = Math.min(...t)
      const hi = Math.max(...t)
      points = { name: data.meta.name, x: xs, y: ys, colors: t.map((v) => ramp((v - lo) / (hi - lo || 1))) }
    } else points = { name: data.meta.name, x: xs, y: ys, slot: 0 }
    const f = data.f && oneD ? { x: xs, y: toFlat(data.f) } : null
    return {
      points,
      f,
      xLabel: data.meta.featureNames[i],
      yLabel: oneD ? (data.meta.targetName ?? 'y') : data.meta.featureNames[j],
      oneD,
    }
  }, [data, dims, colorBy, ramp])
}
