/** Plot v2 (DESIGN.md §5): axis models, `Plot`, `Plots` and the layers. Re-exported from '@lab/viz'. */
export { useAxis, AxisModel, type AxisOptions, type AxisInterval } from './axis'
export { niceRange } from './ticks'
export { Plot, type PlotProps } from './Plot'
export { Plots, type PlotsProps } from './Plots'
export { AxisToolbar } from './AxisToolbar'
export { defineLayer, type CommonProps, type LayerDef, type LayerContext, type LayerOutput, type Orient } from './layer'
export { distributionRange, signedArea, supportOf, type SignedAreaResult } from './probability'
export {
  Annotation,
  Area,
  Bars,
  Curve,
  Points,
  Rug,
  Segments,
  SignedArea,
  Vectors,
  type AnnotationProps,
  type AreaProps,
  type BarsProps,
  type CurveProps,
  type PointsProps,
  type RugProps,
  type SegmentsProps,
  type SignedAreaProps,
  type VectorsProps,
} from './layers/marks'
export {
  Density,
  Histogram,
  Mass,
  SupportBand,
  type DensityProps,
  type HistogramProps,
  type MassProps,
  type SupportBandProps,
} from './layers/distributions'
export { Contours, Raster, type ContoursProps, type RasterProps } from './layers/raster'
export { Probe, type ProbeProps } from './layers/probe'
