/** The lab's visual layer. Everything outside src/viz imports from '@lab/viz' only. */
export { EChart, type EChartProps, type EChartClick, type PlotPointer } from './EChart'
export { XYChart, type XYChartProps, type XYSeries, type Segment, type HoldFit } from './XYChart'
export { Heatmap, type HeatmapProps, type HeatmapOverlay } from './Heatmap'
export { Readout, Readouts } from './Readout'
export { ViewportControls } from './ViewportControls'
export { useViewport, zoomRange, panRange, extentOf, equalUnits, type Range, type Viewport } from './viewport'
export { ChartSize } from './ChartSize'
export {
  FrameContext,
  DEFAULT_HEIGHT,
  useChartHeight,
  useElementSize,
  type FrameContextValue,
  type HoverInfo,
  type HoverRow,
} from './frame'
export type { Handle, Vec2 } from './handles'
export type { Vector } from './vectors'
export { useScaleColor } from './useScaleColor'
export { formatNumber, formatPower, niceStep, snapToStep, stepDecimals } from './format'
export { Subplots, Panel, type SubplotsProps } from './Subplots'
export { type Share } from './subplot-context'
