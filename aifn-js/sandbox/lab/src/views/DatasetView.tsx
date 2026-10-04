import type { Dataset } from 'aifn-methods/data'
import { useMemo } from 'react'
import { PanelSlot } from '@lab/layout'
import { Curve, Plot, Points, Readout, useAxis } from '@lab/viz'
import { useDatasetSeries } from './dataset-series'
import { registerKind, registerView } from './registry'

export type DatasetPanelProps = {
  data: Dataset
  /** The two feature columns to plot. Default [0, 1] (or [0] against y for one feature). */
  dims?: [number, number]
  /** Colour by class label (default when y is integer), by the continuous coordinate `t`, or not at all. */
  colorBy?: 'label' | 't' | 'none'
  /** Equal units on both axes (default true for two features). */
  equal?: boolean
  /** The chart alone, without the metadata readouts. */
  bare?: boolean
}

/** A dataset (`aifn-methods/data`) as a scatter of two features coloured by class or by its coordinate, with its metadata. */
export function DatasetPanel({ data, dims, colorBy, equal, bare }: DatasetPanelProps) {
  const { points, f, xLabel, yLabel, oneD } = useDatasetSeries(data, dims, colorBy)
  const xAxis = useAxis({ label: xLabel })
  const yAxis = useAxis({ label: yLabel, equal: (equal ?? !oneD) ? xAxis : undefined })
  // f(x) through the rows in x order.
  const curve = useMemo(() => {
    if (!f) return null
    const order = Array.from(f.x, (_, i) => i).sort((a, b) => f.x[a] - f.x[b])
    return { x: order.map((i) => f.x[i]), y: order.map((i) => f.y[i]) }
  }, [f])
  const chart = (
    <Plot x={xAxis} y={yAxis}>
      <Points {...points} />
      {curve && <Curve name="f(x)" x={curve.x} y={curve.y} slot={1} />}
    </Plot>
  )
  if (bare) return chart
  return (
    <>
      <PanelSlot slot="readouts">
        <Readout label="shape" value={`${data.x.shape.join(' × ')}`} />
        {data.meta.labelNames && <Readout label="classes" value={data.meta.labelNames.length} />}
        {data.meta.key && <Readout label="key" value={data.meta.key.path} />}
      </PanelSlot>
      {chart}
    </>
  )
}

/** A dataset instance: features `x` as a tensor and `meta` with a name. */
const isDataset = (o: unknown): o is Dataset =>
  typeof o === 'object' &&
  o !== null &&
  typeof (o as Dataset).x === 'object' &&
  Array.isArray((o as Dataset).x?.shape) &&
  typeof (o as Dataset).meta?.name === 'string'

registerKind('dataset', isDataset)
registerView<Dataset>({
  key: 'dataset/scatter',
  kind: 'dataset',
  description: 'Two features as a scatter coloured by class or by the coordinate t, with the shape and class count.',
  title: (d) => d.meta.name,
  render: (d) => <DatasetPanel data={d} />,
})
