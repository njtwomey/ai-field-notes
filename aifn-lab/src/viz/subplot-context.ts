/**
 * What a chart learns from the `Subplots` grid around it (see Subplots.tsx): its place in the grid, which of its axes
 * are shared (and their shared zoom and fitted range), the plot margins that align it with its column and row, and the
 * hover group. Charts outside a Subplots get null and behave as before.
 */
import { createContext, useContext } from 'react'
import { formatNumber, formatPower } from './format'
import { GRID } from './theme'
import type { AxisStore, Range } from './viewport'

/** Share an axis across every panel (`true`), across each row, or down each column. */
export type Share = boolean | 'row' | 'col'

export type HoldSettings = { rescaleOnChange?: boolean; axisKey?: string | number; holdFit?: 'initial' | 'union' }

export type Margins = { left: number; right: number; top: number; bottom: number }

export type PanelReport = {
  /** The panel's data extent on each axis (series and vector ends), and whether it came from an explicit range. */
  x?: { range: Range; explicit: boolean }
  y?: { range: Range; explicit: boolean }
  /** The width in pixels of the panel's widest y tick label. */
  labelWidth: number
  /** Equal units: the grid sizes this panel's height from its fitted ranges (`fit`). */
  equal?: boolean
  /** The panel's own unzoomed ranges, for sizing an equal-aspect panel. */
  fit?: { x?: Range; y?: Range }
  /** Pixels of the panel outside its ECharts box (its toolbar row). */
  chrome?: number
}

export type SubplotContextValue = {
  row: number
  col: number
  rows: number
  cols: number
  /** Whether this panel's x or y is shared with other panels. */
  shared: { x: boolean; y: boolean }
  /** Show tick labels on this axis (false for inner panels of a shared axis). */
  labels: { x: boolean; y: boolean }
  /** False hides an axis's tick labels but keeps its name (set on a Panel, e.g. a scatter matrix's diagonal). */
  ticks?: { x?: boolean; y?: boolean }
  /** Show this axis's zoom controls here: a shared axis has one set, on one of its panels. */
  controls: { x: boolean; y: boolean }
  /** Bumped by the grid's "fit all"; panels reset their own zoom when it changes. */
  resetKey: number
  /** The shared zoom of each shared axis. */
  stores: { x?: AxisStore; y?: AxisStore }
  /** The unzoomed range of each shared axis: the union of its panels' extents (held still while a handle is dragged). */
  fitted: { x?: Range; y?: Range }
  /** Plot margins that align this panel's plot area with its column (left, right) and row (top, bottom). */
  margins: Margins
  /** The y-axis name gap that clears the column's widest tick labels. */
  yNameGap: number
  hoverGroup?: string
  /** A dense grid (a scatter matrix): charts draw no legend. */
  dense?: boolean
  /** `equal` (set on a Panel): the grid sizes this panel so its units are equal (see subplot-layout.ts). */
  aspect?: 'equal' | 'fit'
  /** The grid has sized this panel for equal units: its y range follows from its x range and its pixel size. */
  sizedEqual: boolean
  /** Held axes for every chart in the grid (or the panel): see `XYChartProps.rescaleOnChange`. */
  hold: HoldSettings
  report: (key: string, report: PanelReport | null) => void
  setDragging: (dragging: boolean) => void
}

export const SubplotContext = createContext<SubplotContextValue | null>(null)

export function useSubplot(): SubplotContextValue | null {
  return useContext(SubplotContext)
}

/** The group a panel's axis belongs to under a sharing rule, or null if the axis is its own. */
export function shareGroup(axis: 'x' | 'y', share: Share, row: number, col: number): string | null {
  if (share === false) return null
  if (share === 'row') return `${axis}:r${row}`
  if (share === 'col') return `${axis}:c${col}`
  return `${axis}:all`
}

/** The tick step ECharts picks for a span (1, 2, 3 or 5 times a power of ten). */
function tickStep(span: number): number {
  const raw = span / 5
  const unit = 10 ** Math.floor(Math.log10(raw))
  const f = raw / unit
  return unit * (f <= 1 ? 1 : f <= 2 ? 2 : f <= 3 ? 3 : f <= 5 ? 5 : 10)
}

/** `r` widened outward to whole ticks (whole decades on a log axis), as ECharts' own fit would. */
export function niceRange([lo, hi]: Range, log = false): Range {
  if (log && lo > 0) return [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))]
  if (!(hi > lo)) return [lo - 1, hi + 1]
  const step = tickStep(hi - lo)
  return [Math.floor(lo / step + 1e-9) * step, Math.ceil(hi / step - 1e-9) * step]
}

const FONT = "11px 'Geist Variable', system-ui, sans-serif"
let context: CanvasRenderingContext2D | null | undefined

/** The pixel width of the widest tick label a y axis over `range` shows, measured in the axis font. */
export function tickLabelWidth(range: Range | undefined, log = false): number {
  if (!range || !(range[1] > range[0])) return 24
  const labels: string[] = []
  if (log && range[0] > 0) {
    for (let e = Math.ceil(Math.log10(range[0])); e <= Math.floor(Math.log10(range[1])); e++)
      labels.push(formatPower(10 ** e))
  } else {
    const step = tickStep(range[1] - range[0])
    for (let v = Math.ceil(range[0] / step) * step; v <= range[1] + step * 1e-9; v += step) labels.push(formatNumber(v))
  }
  if (context === undefined)
    context = typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d')
  if (!context) return Math.max(24, ...labels.map((l) => l.length * 6.5))
  context.font = FONT
  return Math.ceil(Math.max(24, ...labels.map((l) => context!.measureText(l).width)))
}

/** Room for the rotated y-axis name, beside the tick labels. */
export const Y_NAME_ROOM = 22
/** Gap between the tick labels and the axis line. */
export const LABEL_GAP = 8

/** The default margins, for panels before they are measured. */
export const DEFAULT_MARGINS: Margins = { ...GRID }
