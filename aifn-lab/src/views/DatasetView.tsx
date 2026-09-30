import type { Dataset } from 'aifn/datasets'
import { Figure } from '@lab/layout'
import { Readout, XYChart } from '@lab/viz'
import { useDatasetSeries } from './dataset-series'
import type { FrameProps } from './frame'

export type DatasetViewProps = FrameProps & {
  data: Dataset
  /** The two feature columns to plot. Default [0, 1] (or [0] against y for one feature). */
  dims?: [number, number]
  /** Colour by class label (default when y is integer), by the continuous coordinate `t`, or not at all. */
  colorBy?: 'label' | 't' | 'none'
  /** Equal units on both axes (default true for two features). */
  equal?: boolean
  /** Draw inside an existing figure instead of its own. */
  bare?: boolean
}

/** A dataset (`aifn/datasets`) as a scatter of two features coloured by class or by its coordinate, with its metadata. */
export function DatasetView({
  data,
  dims,
  colorBy,
  equal,
  bare,
  title,
  description,
  controls,
  readouts,
  caption,
  id,
  defaultSize,
}: DatasetViewProps) {
  const { series, xLabel, yLabel, oneD } = useDatasetSeries(data, dims, colorBy)
  const chart = <XYChart series={series} xLabel={xLabel} yLabel={yLabel} aspect={(equal ?? !oneD) ? 'equal' : 'fit'} />
  if (bare) return chart
  return (
    <Figure
      id={id}
      title={title ?? data.meta.name}
      description={description ?? data.meta.description}
      controls={controls}
      defaultSize={defaultSize}
      readouts={
        <>
          {readouts}
          <Readout label="shape" value={`${data.x.shape.join(' × ')}`} />
          {data.meta.labelNames && <Readout label="classes" value={data.meta.labelNames.length} />}
          {data.meta.stream && <Readout label="stream" value={data.meta.stream} />}
        </>
      }
      caption={caption ?? data.meta.source}
    >
      {chart}
    </Figure>
  )
}
