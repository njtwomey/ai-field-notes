import { useMemo } from 'react'
import type { Dataset } from 'aifn-applied/data'
import { toFlat } from 'aifn/foundation/tensor'
import { useScaleColor, type XYSeries } from '@lab/viz'

/** The scatter series of a dataset: by class (groups), by the coordinate t (a colour ramp), or plain. */
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
    let series: XYSeries
    if (mode === 'label' && labelled && !oneD)
      series = {
        name: data.meta.name,
        type: 'scatter',
        x: xs,
        y: ys,
        group: toFlat(data.y!),
        groupNames: data.meta.labelNames,
      }
    else if (mode === 't' && data.t) {
      const t = toFlat(data.t)
      const lo = Math.min(...t)
      const hi = Math.max(...t)
      series = {
        name: data.meta.name,
        type: 'scatter',
        x: xs,
        y: ys,
        pointColors: t.map((v) => ramp((v - lo) / (hi - lo || 1))),
      }
    } else series = { name: data.meta.name, type: 'scatter', x: xs, y: ys, slot: 0 }
    const f: XYSeries[] = data.f && oneD ? [{ name: 'f(x)', type: 'line', x: xs, y: toFlat(data.f), slot: 1 }] : []
    return {
      series: [series, ...f],
      xLabel: data.meta.featureNames[i],
      yLabel: oneD ? (data.meta.targetName ?? 'y') : data.meta.featureNames[j],
      oneD,
    }
  }, [data, dims, colorBy, ramp])
}
