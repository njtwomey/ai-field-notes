import { useCallback, useEffect, useId, useMemo, useState } from 'react'
import {
  categorical,
  chrome,
  interpolateColors,
  MARKER_SHAPES,
  mute,
  scaleStops,
  seriesColor,
  type Mode,
} from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { contourSegments } from './contours'
import { EChart, type EChartClick, type PlotPointer } from './EChart'
import { formatNumber } from './format'
import { useChartHeight, useElementSize, useFrameData, useFrameHover } from './frame'
import type { Handle } from './handles'
import { tickLabelWidth, useSubplot, type Margins } from './subplot-context'
import { GRID, LINE_WIDTH, MARKER_SIZE } from './theme'
import { vectorLines, type Vector } from './vectors'
import { drawnViewport, useViewport, type Range } from './viewport'
import { ViewportControls } from './ViewportControls'

type CustomApi = {
  value: (dim: number) => number
  coord: (point: number[]) => number[]
  visual: (key: string) => string
}

export type HeatmapOverlay = {
  name: string
  type: 'scatter' | 'line'
  x: ArrayLike<number>
  y: ArrayLike<number>
  group?: ArrayLike<number> | null
  groupNames?: readonly string[]
  /** Fixed categorical slot. Defaults to the overlay's index, shifted past the sequential ramp's hue (slot 0). */
  slot?: number
  /** Lines only: mark every vertex, e.g. each step of an optimiser. */
  showPoints?: boolean
  /** Lines only: a light line, for many stochastic draws such as several chains. */
  thin?: boolean
  /** Ink-coloured diamond, e.g. an optimum. Not a category, so no palette slot. */
  emphasis?: boolean
  /** Lines only: an explicit colour in place of the slot or ink, e.g. an emphasis line that must read on any fill. */
  color?: string
  /** Lines only: the stroke width in pixels (default 2, or 1 when `thin`), e.g. a wider halo drawn under a line. */
  width?: number
  /** Lines only: dashed, e.g. a reference contour. */
  dashed?: boolean
  /** Scatter only: colour each point by its value on the heatmap's own scale, with an ink outline. */
  values?: ArrayLike<number>
  /** Scatter only: an explicit colour per point (from `useScaleColor`), drawn like `values`. */
  colors?: readonly string[]
  /** Scatter with `values` or `colors`: the marker outline (default ink), e.g. a light ring on a field of like hue. */
  outline?: string
}

export type HeatmapProps = {
  /** Cell centres along x (columns) and y (rows), evenly spaced. */
  x: ArrayLike<number>
  y: ArrayLike<number>
  /** Row-major: z[i][j] is the value at (x[j], y[i]). */
  z: readonly (readonly number[])[]
  xLabel?: string
  yLabel?: string
  /**
   * `sequential` for magnitude, `diverging` for signed values around the middle of `range`, `categorical` for classes:
   * z holds a class index k ≥ 0 drawn in slot k, or -1 (unassigned) and -2 (contested) as neutrals. No colour bar.
   */
  scale?: 'sequential' | 'diverging' | 'categorical'
  /** Categorical only: a name per class index, shown in the tooltip and hover readout. */
  categoryNames?: readonly string[]
  /** The colour scale's ends; by default the data's min and max. */
  range?: [number, number]
  /** Label the colour bar at these values rather than at 3–5 nice ticks. */
  scaleTicks?: readonly number[]
  /** Strength of the cell colours, below 1 to mute the fill under full-strength overlays. */
  fillOpacity?: number
  /** Ink contour lines at `levels` of `field` (a second grid) or of `z`. Pass a memoised object. */
  contours?: { levels: readonly number[]; field?: readonly (readonly number[])[] }
  /** Points or paths over the grid. Their data is applied as a patch, so moving a path does not redraw the grid. */
  overlay?: readonly HeatmapOverlay[]
  /** One highlighted point, e.g. the current parameter setting. Moving it does not redraw the grid. */
  marker?: [number, number]
  /** Draggable handles bound to parameters. See handles.ts. */
  handles?: Handle[]
  /** Arrows over the grid. Ink unless a vector sets `slot`; on a sequential grid use slot 1 or above. */
  vectors?: readonly Vector[]
  /** The centre of a clicked cell, or a clicked overlay point. Prefer handles. */
  onCellClick?: (x: number, y: number) => void
  /** Hover, click and leave positions in data coordinates. */
  onPointer?: (event: PlotPointer) => void
  /** Name of the cell value in tooltips and readouts. */
  valueLabel?: string
  /** Pixels. Inside a Figure the frame sets the height and this is ignored. */
  height?: number
  /**
   * Equal pixel length per unit on both axes. The axes stay on the grid's extent (never padded); the plot area takes the
   * grid's aspect and is centred in the space the chart has. Inside `Subplots`, `Panel aspect="equal"` does the same and
   * lets a single-column grid size the panel's height to fit.
   */
  equalAspect?: boolean
  /** False hides the colour bar. */
  colorBar?: boolean
  /**
   * Refit the colour scale whenever z changes (default true). False holds it (fitted once; `union` grows it), so a
   * change in z reads as a change of colour. `axisKey` refits; the fit button refits too.
   */
  rescaleOnChange?: boolean
  axisKey?: string | number
  holdFit?: 'initial' | 'union'
  /** The zoom and pan toolbar (default true). */
  zoom?: boolean
  ariaLabel?: string
}

const NO_OVERLAY: readonly HeatmapOverlay[] = []

/** A categorical cell's colour: slot k for class k, a light neutral for -1, a darker neutral for -2. */
function categoryColor(mode: Mode, value: number): string {
  if (value === -2) return chrome(mode).muted
  if (value < 0) return chrome(mode).grid
  const colours = categorical(mode)
  return colours[Math.round(value) % colours.length]
}

function categoryName(value: number, names: readonly string[] | undefined): string {
  if (value === -1) return 'unassigned'
  if (value === -2) return 'contested'
  return names?.[value] ?? formatNumber(value)
}

/** Contour lines as one ink line series (segments split by nulls), plus one label per level. */
function contourSeries(
  x: number[],
  y: number[],
  z: readonly (readonly number[])[],
  contours: HeatmapProps['contours'],
  mode: Mode,
) {
  if (!contours) return []
  const { ink, surface } = chrome(mode)
  const lines: (number | null)[][] = []
  const labels: number[][] = []
  for (const level of contours.levels) {
    const segments = contourSegments(x, y, (contours.field ?? z) as number[][], level)
    for (const [a, b] of segments) lines.push(a, b, [null, null])
    const mid = segments[Math.floor(segments.length / 2)]
    if (mid) labels.push([(mid[0][0] + mid[1][0]) / 2, (mid[0][1] + mid[1][1]) / 2, level])
  }
  return [
    {
      id: '__contours',
      name: '__contours',
      type: 'line',
      data: lines,
      connectNulls: false,
      showSymbol: false,
      silent: true,
      clip: true,
      lineStyle: { color: ink, width: 1 },
      tooltip: { show: false },
      // Static: without animation ECharts does not diff every contour point against the last frame on each update.
      animation: false,
      z: 2,
    },
    {
      id: '__contour-labels',
      name: '__contour-labels',
      type: 'scatter',
      data: labels,
      symbolSize: 0,
      silent: true,
      clip: true,
      label: {
        show: true,
        formatter: (p: { value: number[] }) => formatNumber(p.value[2]),
        color: ink,
        fontSize: 10,
        backgroundColor: surface,
        padding: [1, 3],
        borderRadius: 2,
      },
      tooltip: { show: false },
      animation: false,
      z: 4,
    },
  ]
}

type OverlayPoint = number[] | { value: number[]; itemStyle: { color: string } }

/** The colour every point of a series shares, if they share one. */
function groupColor(data: OverlayPoint[]): string | undefined {
  const colors = new Set(data.map((d) => (Array.isArray(d) ? undefined : d.itemStyle.color)))
  const [only] = colors
  return colors.size === 1 ? only : undefined
}
type OverlaySeries = { id: string; data: OverlayPoint[]; style: Record<string, unknown> }

/** Overlays as ECharts series. On a sequential grid overlays start at slot 1, since slot 0 is the ramp's hue. */
function overlaySeries(
  overlay: readonly HeatmapOverlay[],
  mode: Mode,
  scale: HeatmapProps['scale'],
  lo: number,
  hi: number,
): OverlaySeries[] {
  const shift = scale === 'sequential' ? 1 : 0
  const { ink, surface } = chrome(mode)
  const stops = scaleStops(scale === 'diverging' ? 'diverging' : 'sequential', mode)
  return overlay.flatMap((s, index): OverlaySeries[] => {
    const group = s.group
    if (s.values || s.colors) {
      const colorAt = (i: number) =>
        s.colors?.[i] ??
        (scale === 'categorical'
          ? categoryColor(mode, s.values?.[i] ?? -1)
          : interpolateColors(stops, hi > lo ? ((s.values?.[i] ?? lo) - lo) / (hi - lo) : 0))
      // Colour carries the value; `group` only picks the marker shape. Every named group keeps a series, so the legend
      // (drawn in ink) does not change as points move.
      const groups = group
        ? (s.groupNames?.map((_, g) => g) ?? [...new Set(Array.from(group))].sort((a, b) => a - b))
        : [0]
      return groups.map((g) => {
        const data: OverlayPoint[] = []
        for (let i = 0; i < s.x.length; i++)
          if (!group || group[i] === g) data.push({ value: [s.x[i], s.y[i]], itemStyle: { color: colorAt(i) } })
        return {
          id: `overlay-${index}-values-${g}`,
          data,
          style: {
            name: group ? (s.groupNames?.[g] ?? `${s.name} ${g}`) : s.name,
            type: 'scatter',
            symbol: MARKER_SHAPES[g % MARKER_SHAPES.length],
            symbolSize: MARKER_SIZE,
            // The series colour (its legend icon) is the group's own when every point of it shares one, e.g. a class.
            itemStyle: {
              color: groupColor(data) ?? ink,
              borderColor: s.outline ?? ink,
              borderWidth: s.outline ? 1.5 : 1,
            },
            clip: true,
            z: 3,
          },
        }
      })
    }
    const groups = group ? [...new Set(Array.from(group))].sort((a, b) => a - b) : [null]
    return groups.map((g) => {
      const slot = g !== null ? g + shift : (s.slot ?? index + shift)
      const line = s.type === 'line'
      const color = (line && s.color) || (s.emphasis ? ink : seriesColor(mode, slot))
      const data: OverlayPoint[] = []
      for (let i = 0; i < s.x.length; i++) if (g === null || group![i] === g) data.push([s.x[i], s.y[i]])
      return {
        id: `overlay-${index}-${g ?? 'all'}`,
        data,
        style: {
          name: g === null ? s.name : (s.groupNames?.[g] ?? `${s.name} ${g}`),
          type: line ? 'line' : 'scatter',
          symbol: s.emphasis ? 'diamond' : line ? 'circle' : MARKER_SHAPES[slot % MARKER_SHAPES.length],
          symbolSize: s.emphasis ? 16 : line ? 5 : MARKER_SIZE,
          showSymbol: !line || !!s.showPoints,
          itemStyle: { color, borderColor: surface, borderWidth: s.emphasis ? 2 : 1 },
          lineStyle: {
            color,
            width: s.width ?? (s.thin ? 1 : LINE_WIDTH),
            opacity: s.thin ? 0.5 : 1,
            type: s.dashed ? 'dashed' : 'solid',
          },
          clip: true,
          z: s.emphasis ? 4 : 3,
        },
      }
    })
  })
}

/**
 * The cells drawn once into an offscreen canvas, `k` pixels per cell, so the grid is one image to ECharts: a drag that
 * patches an overlay re-renders one element instead of every cell. `k` follows the cells' size on screen (in device
 * pixels), so the image is never scaled up far enough to blur the cell edges. Non-finite cells stay transparent.
 */
function drawRaster(
  x: readonly number[],
  y: readonly number[],
  z: readonly (readonly number[])[],
  k: number,
  colorOf: (v: number) => string,
): HTMLCanvasElement | null {
  if (typeof document === 'undefined' || !x.length || !y.length) return null
  const canvas = document.createElement('canvas')
  canvas.width = x.length * k
  canvas.height = y.length * k
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  // Colours repeat across cells (categorical classes, equal values), so each is computed once per value.
  const cache = new Map<number, string>()
  // Canvas rows run top down, so increasing y is flipped; columns follow x.
  const flipX = x.length > 1 && x[1] < x[0]
  const flipY = !(y.length > 1 && y[1] < y[0])
  for (let i = 0; i < y.length; i++) {
    const row = flipY ? y.length - 1 - i : i
    for (let j = 0; j < x.length; j++) {
      const v = z[i]?.[j]
      if (!Number.isFinite(v)) continue
      let color = cache.get(v)
      if (color === undefined) cache.set(v, (color = colorOf(v)))
      ctx.fillStyle = color
      ctx.fillRect((flipX ? x.length - 1 - j : j) * k, row * k, k, k)
    }
  }
  return canvas
}

/** The colour bar's geometry: gap from the plot, bar width, tick length, gap before the tick labels. */
const BAR = { gap: 12, width: 10, tick: 4, labelGap: 3 }
/** Room below the toolbar that a grid must allow for (see XYChart's TOOLBAR_CHROME). */
const TOOLBAR_CHROME = 28

/** 3–5 round values (1, 2 or 5 × 10ᵏ apart) inside [lo, hi], for the colour bar's axis; the ends if none fit. */
function colorBarTicks(lo: number, hi: number): number[] {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return Number.isFinite(lo) ? [lo] : []
  const span = hi - lo
  const unit = 10 ** Math.floor(Math.log10(span / 10))
  const at = (step: number) => {
    const out: number[] = []
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step)
      out.push(Number(v.toPrecision(12)))
    return out
  }
  let best: number[] | null = null
  for (const m of [1, 2, 5, 10, 20, 50, 100]) {
    const ticks = at(m * unit)
    if (ticks.length >= 3 && ticks.length <= 5 && (!best || Math.abs(ticks.length - 4) < Math.abs(best.length - 4)))
      best = ticks
  }
  return best ?? [lo, hi]
}

/**
 * Plot margins: the Subplots column's left and top edges when in a grid, room for the x tick labels below, and room
 * for the colour bar and its labels on the right unless it is hidden.
 */
function baseMargins(sub: Margins | undefined, barWidth: number): Margins {
  const m = sub ? { ...sub, bottom: Math.max(sub.bottom, GRID.bottom) } : { ...GRID }
  return barWidth ? { ...m, right: Math.max(m.right, barWidth + 8) } : m
}

/**
 * A value grid on numeric axes with optional overlays, zoomable and pannable. Cells are a custom series rather than
 * the ECharts category heatmap, so overlays share the same continuous axes. Hovering a cell shows its x, y and value.
 */
export function Heatmap({
  x: xIn,
  y: yIn,
  z,
  xLabel,
  yLabel,
  scale = 'sequential',
  categoryNames,
  range,
  scaleTicks,
  fillOpacity = 1,
  contours,
  overlay = NO_OVERLAY,
  marker,
  vectors,
  handles,
  onCellClick,
  onPointer,
  valueLabel = 'value',
  height: ownHeight,
  equalAspect = false,
  colorBar = true,
  zoom = true,
  rescaleOnChange,
  axisKey,
  holdFit,
  ariaLabel,
}: HeatmapProps) {
  const { resolved: mode } = useTheme()
  const height = useChartHeight(ownHeight)
  const [box, size] = useElementSize<HTMLDivElement>()
  const x = useMemo(() => Array.from(xIn), [xIn])
  const y = useMemo(() => Array.from(yIn), [yIn])
  const [rangeLo, rangeHi] = range ?? [undefined, undefined]
  const clickable = !!onCellClick
  const isCategorical = scale === 'categorical'
  const showBar = colorBar && !isCategorical
  const sub = useSubplot()
  const equal = equalAspect || sub?.aspect === 'equal'
  // A value for the memo dependencies, so an inline array does not redraw the grid.
  const namesKey = categoryNames?.join('\u0000')

  const yNameGap = sub?.yNameGap ?? 40
  const dx = x.length > 1 ? x[1] - x[0] : 1
  const dy = y.length > 1 ? y[1] - y[0] : 1
  const bounds = useMemo(
    () => ({
      x: [x[0] - dx / 2, x[x.length - 1] + dx / 2] as Range,
      y: [y[0] - dy / 2, y[y.length - 1] + dy / 2] as Range,
    }),
    [x, y, dx, dy],
  )
  const viewport = useViewport(bounds)
  const [vx0, vx1] = viewport.shown.x ?? bounds.x
  const [vy0, vy1] = viewport.shown.y ?? bounds.y

  const [fitLo, fitHi] = useMemo(() => {
    if (rangeLo !== undefined && rangeHi !== undefined) return [rangeLo, rangeHi]
    let [min, max] = [Infinity, -Infinity]
    for (const row of z) for (const v of row) if (Number.isFinite(v)) [min, max] = [Math.min(min, v), Math.max(max, v)]
    return [rangeLo ?? (Number.isFinite(min) ? min : 0), rangeHi ?? (Number.isFinite(max) ? max : 1)]
  }, [z, rangeLo, rangeHi])
  // Held colour scale: the range taken on mount (or when `axisKey` changes), kept or grown (`union`) as z changes.
  const hold = !(rescaleOnChange ?? true)
  const [held, setHeld] = useState<{ key: unknown; lo: number; hi: number } | null>(null)
  // Updated during render (state derived from props), so the scale never lags a frame.
  let kept = held
  if (hold) {
    if (!held || held.key !== axisKey) kept = { key: axisKey, lo: fitLo, hi: fitHi }
    else if (holdFit === 'union' && (fitLo < held.lo || fitHi > held.hi))
      kept = { key: axisKey, lo: Math.min(held.lo, fitLo), hi: Math.max(held.hi, fitHi) }
    if (kept !== held) setHeld(kept)
  } else kept = null
  const [lo, hi] = kept ? [kept.lo, kept.hi] : [fitLo, fitHi]
  const overlays = useMemo(() => overlaySeries(overlay, mode, scale, lo, hi), [overlay, mode, scale, lo, hi])
  // Structure only: the option depends on this string, not on overlay data.
  const overlayKey = JSON.stringify(overlays.map((s) => [s.id, s.style]))

  // The colour bar's ticks and the room its labels take (the value label sits above the bar).
  const ticksKey = (scaleTicks ?? colorBarTicks(lo, hi)).join(',')
  const barTicks = useMemo(() => (ticksKey ? ticksKey.split(',').map(Number) : []), [ticksKey])
  const barRoom = showBar
    ? BAR.gap + BAR.width + BAR.tick + BAR.labelGap + Math.max(...barTicks.map((v) => formatNumber(v).length * 6.5), 12)
    : 0

  // The plot area. Axes span the grid's extent exactly (or the zoomed ranges). With equal units the plot area takes the
  // ranges' aspect and is centred in the room the margins leave, so the raster fills it and nothing is padded.
  // A legend (two or more overlay series) sits at the top right, so the bar's value label moves down below it.
  const base = baseMargins(sub?.margins, barRoom)
  const m = showBar && overlays.length > 1 ? { ...base, top: base.top + 16 } : base
  let grid = m
  const roomW = size.width - m.left - m.right
  const roomH = size.height - m.top - m.bottom
  if (equal && roomW > 0 && roomH > 0 && vx1 > vx0 && vy1 > vy0) {
    const aspect = (vy1 - vy0) / (vx1 - vx0)
    if (roomW * aspect <= roomH) {
      const extra = Math.floor((roomH - roomW * aspect) / 2)
      grid = { ...m, top: m.top + extra, bottom: m.bottom + extra }
    } else {
      const extra = Math.floor((roomW - roomH / aspect) / 2)
      grid = { ...m, left: m.left + extra, right: m.right + extra }
    }
  }
  const { left: gl, right: gr, top: gt, bottom: gb } = grid

  // One cell's size on screen in CSS pixels: the hit area of each cell, and (in device pixels, rounded up to a power of
  // two so that a zoom or resize seldom redraws it) the raster's pixels per cell, at most 4096 pixels a side.
  const plotWidth = Math.max(size.width - gl - gr, 1)
  const plotHeight = Math.max(size.height - gt - gb, 1)
  const cellWidth = size.width > 0 && vx1 > vx0 ? (plotWidth * Math.abs(dx)) / (vx1 - vx0) : 8
  const cellHeight = size.height > 0 && vy1 > vy0 ? (plotHeight * Math.abs(dy)) / (vy1 - vy0) : 8
  const cellW = Math.ceil(cellWidth)
  const cellH = Math.ceil(cellHeight)
  const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1
  const k = Math.min(
    2 ** Math.ceil(Math.log2(Math.max(cellWidth, cellHeight, 1) * dpr)),
    2 ** Math.max(0, Math.floor(Math.log2(4096 / Math.max(x.length, y.length, 1)))),
  )
  // The raster depends on the cells and their colours only, never on overlays, markers or handles.
  const raster = useMemo(() => {
    const { surface } = chrome(mode)
    const stops = scaleStops(scale === 'diverging' ? 'diverging' : 'sequential', mode)
    const colorOf = (v: number) => {
      const c = isCategorical ? categoryColor(mode, v) : interpolateColors(stops, hi > lo ? (v - lo) / (hi - lo) : 0)
      return fillOpacity < 1 ? mute(c, surface, fillOpacity) : c
    }
    return drawRaster(x, y, z, k, colorOf)
  }, [x, y, z, k, lo, hi, scale, isCategorical, mode, fillOpacity])

  const option = useMemo(() => {
    const structure = JSON.parse(overlayKey) as [string, Record<string, unknown>][]
    const { surface, ink } = chrome(mode)
    // The grid's outer edges, half a step beyond the first and last centres.
    const [hx, hy] = [Math.abs(dx) / 2, Math.abs(dy) / 2]
    const [gx0, gx1] = [Math.min(x[0], x[x.length - 1]) - hx, Math.max(x[0], x[x.length - 1]) + hx]
    const [gy0, gy1] = [Math.min(y[0], y[y.length - 1]) - hy, Math.max(y[0], y[y.length - 1]) + hy]
    const cells = y.flatMap((yv, i) => x.map((xv, j) => [xv, yv, z[i][j]]))
    const names = namesKey?.split('\u0000')
    const valueText = (v: number) => (isCategorical ? categoryName(v, names) : formatNumber(v))
    return {
      grid: { left: gl, right: gr, top: gt, bottom: gb },
      legend: {
        show: structure.filter(([, style]) => !String(style.name).startsWith('__')).length > 1,
        // Each entry shows its series' marker; series sharing a name get one entry.
        data: structure
          // Overlays named `__…` (a halo, a hover mark) stay out of the legend, as in XYChart.
          .filter(([, style]) => !String(style.name).startsWith('__'))
          .filter(([, style], i, all) => all.findIndex(([, other]) => other.name === style.name) === i)
          .map(([, style]) => ({
            name: style.name as string,
            ...(style.type === 'scatter' && typeof style.symbol === 'string' ? { icon: style.symbol } : {}),
          })),
      },
      tooltip: {
        trigger: 'item',
        formatter: (p: { seriesName: string; value: number[] }) =>
          p.seriesName === '__grid'
            ? `${xLabel ?? 'x'} ${formatNumber(p.value[0])}, ${yLabel ?? 'y'} ${formatNumber(p.value[1])}<br/>${valueLabel}: <b>${valueText(p.value[2])}</b>`
            : `${p.seriesName}<br/>(${formatNumber(p.value[0])}, ${formatNumber(p.value[1])})`,
      },
      // Axis lines on the plot's edges, never at zero across the raster.
      xAxis: {
        type: 'value',
        name: xLabel,
        min: vx0,
        max: vx1,
        splitLine: { show: false },
        axisLine: { onZero: false },
      },
      yAxis: {
        type: 'value',
        name: yLabel,
        min: vy0,
        max: vy1,
        splitLine: { show: false },
        axisLine: { onZero: false },
        nameGap: yNameGap,
      },
      series: [
        // The cells, as one image spanning the grid (see drawRaster).
        {
          id: '__raster',
          name: '__raster',
          type: 'custom',
          silent: true,
          clip: true,
          data: [[gx0, gy0, lo]],
          encode: { x: 0, y: 1 },
          tooltip: { show: false },
          renderItem: (_: unknown, api: CustomApi) => {
            if (!raster) return null
            const [px0, py0] = api.coord([gx0, gy1])
            const [px1, py1] = api.coord([gx1, gy0])
            return { type: 'image', style: { image: raster, x: px0, y: py0, width: px1 - px0, height: py1 - py0 } }
          },
          z: 1,
        },
        // Hover and click targets: a transparent rectangle per cell in a large-mode scatter, which is one path rather
        // than an element per cell, so the tooltip and cell clicks cost a drag nothing.
        {
          id: '__grid',
          name: '__grid',
          type: 'scatter',
          large: true,
          largeThreshold: 0,
          clip: true,
          data: cells,
          symbol: 'rect',
          symbolSize: [cellW + 1, cellH + 1],
          itemStyle: { opacity: 0 },
          cursor: clickable ? 'pointer' : 'default',
          z: 1,
        },
        ...contourSeries(x, y, z, contours, mode),
        ...structure.map(([id, style]) => ({ id, ...style, data: [] })),
        {
          id: 'marker',
          name: 'current',
          type: 'scatter',
          data: [],
          symbol: 'circle',
          symbolSize: 14,
          clip: true,
          itemStyle: { color: ink, borderColor: surface, borderWidth: 2 },
          // Vectors ride on the marker series as a markLine, which orients arrowheads in screen space.
          markLine: {
            silent: true,
            symbol: ['none', 'arrow'],
            symbolSize: 10,
            label: { show: false },
            lineStyle: { color: ink, width: LINE_WIDTH, type: 'solid' },
            animation: false,
            data: [],
          },
          z: 5,
        },
      ],
    }
  }, [
    x,
    y,
    z,
    dx,
    dy,
    xLabel,
    yLabel,
    isCategorical,
    namesKey,
    lo,
    contours,
    overlayKey,
    valueLabel,
    mode,
    gl,
    gr,
    gt,
    gb,
    yNameGap,
    clickable,
    vx0,
    vx1,
    vy0,
    vy1,
    raster,
    cellW,
    cellH,
  ])

  const [mx, my] = marker ?? [undefined, undefined]
  const patch = useMemo(
    () => ({
      series: [
        ...overlays.map((s) => ({ id: s.id, data: s.data })),
        {
          id: 'marker',
          data: mx === undefined ? [] : [[mx, my]],
          markLine: { data: vectorLines(vectors ?? [], mode, { x: [vx0, vx1], y: [vy0, vy1] }) },
        },
      ],
    }),
    [overlays, mx, my, vectors, mode, vx0, vx1, vy0, vy1],
  )

  const handleClick = onCellClick
    ? (e: EChartClick) => {
        if (e.seriesId === 'marker' || !Array.isArray(e.value) || e.value.length < 2) return
        onCellClick(e.value[0] as number, e.value[1] as number)
      }
    : undefined

  // The frame's hover readout: the cell under the pointer.
  const setHover = useFrameHover()
  const pointer = useCallback(
    (event: PlotPointer) => {
      onPointer?.(event)
      if (event.type !== 'move') return event.type === 'leave' ? setHover(null) : undefined
      const j = Math.round((event.point[0] - x[0]) / dx)
      const i = Math.round((event.point[1] - y[0]) / dy)
      if (!(i >= 0 && i < y.length && j >= 0 && j < x.length)) return setHover(null)
      const v = z[i][j]
      setHover({
        at: `${xLabel ?? 'x'} = ${formatNumber(x[j])}, ${yLabel ?? 'y'} = ${formatNumber(y[i])}`,
        rows: [{ label: valueLabel, value: isCategorical ? categoryName(v, categoryNames) : formatNumber(v) }],
      })
    },
    [onPointer, setHover, x, y, z, dx, dy, xLabel, yLabel, valueLabel, isCategorical, categoryNames],
  )

  useFrameData(
    useCallback(
      () => ({ kind: 'heatmap', xLabel, yLabel, valueLabel, x, y, z }),
      [x, y, z, xLabel, yLabel, valueLabel],
    ),
  )

  // In a Subplots grid: the widest y tick label (for the column's margins) and, with equal units, the grid's extent, so
  // a single-column grid can size this panel's height from it. The axes are the heatmap's own, so no shared extents.
  const reportTo = sub?.report
  const panelKey = useId()
  const labelWidth = Math.ceil(tickLabelWidth([vy0, vy1]) / 8) * 8
  const [bx0, bx1] = bounds.x
  const [by0, by1] = bounds.y
  useEffect(() => {
    reportTo?.(panelKey, {
      labelWidth,
      equal,
      fit: { x: [Math.min(bx0, bx1), Math.max(bx0, bx1)], y: [Math.min(by0, by1), Math.max(by0, by1)] },
      chrome: zoom ? TOOLBAR_CHROME : 0,
    })
  }, [reportTo, panelKey, labelWidth, equal, bx0, bx1, by0, by1, zoom])
  useEffect(() => () => reportTo?.(panelKey, null), [reportTo, panelKey])

  // The toolbar works on the drawn ranges; with equal aspect, zooming one axis zooms both.
  const drawnControls = drawnViewport(viewport, { x: [vx0, vx1], y: [vy0, vy1] }, {}, equal)
  const controls = hold
    ? {
        ...drawnControls,
        zoomed: true,
        reset: () => {
          viewport.reset()
          setHeld(null)
        },
      }
    : drawnControls

  return (
    <div className="flex w-full flex-col gap-1" style={{ height }}>
      {zoom && (
        <ViewportControls
          viewport={controls}
          className={sub ? 'h-6 flex-nowrap overflow-x-auto overflow-y-hidden px-1' : 'px-1'}
        />
      )}
      <div className="flex min-h-0 flex-1">
        <div ref={box} className="relative min-w-0 flex-1">
          <EChart
            option={option}
            patch={patch}
            height="fill"
            className="absolute inset-0"
            onClick={handleClick}
            onPointer={pointer}
            handles={handles}
            ariaLabel={ariaLabel}
            onWheelZoom={zoom ? (factor, [px, py]) => controls.zoom(factor, ['x', 'y'], { x: px, y: py }) : undefined}
            // One rectangle per cell: thousands of marks, which canvas draws far faster than SVG.
            renderer="canvas"
          />
          {showBar && size.width > 0 && (
            <ScaleBar
              stops={scaleStops(scale === 'diverging' ? 'diverging' : 'sequential', mode)}
              lo={lo}
              hi={hi}
              ticks={barTicks}
              label={valueLabel}
              left={size.width - gr + BAR.gap}
              top={gt}
              height={Math.max(size.height - gt - gb, 40)}
              room={gr - BAR.gap}
            />
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * The colour bar: a vertical ramp beside the plot, as tall as the plot area, low values at the bottom, with a small axis
 * of labelled ticks and the value's name above it. Drawn as SVG over the chart's box, so it follows the plot area.
 */
function ScaleBar({
  stops,
  lo,
  hi,
  ticks,
  label,
  left,
  top,
  height,
  room,
}: {
  stops: readonly string[]
  lo: number
  hi: number
  ticks: readonly number[]
  label: string
  left: number
  top: number
  height: number
  room: number
}) {
  // useId can contain characters that are not valid in a url(#…) reference.
  const id = `scale${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const at = (v: number) => height * (1 - (hi > lo ? (v - lo) / (hi - lo) : 0.5))
  const tickX = BAR.width + BAR.tick
  // The value's name starts over the bar; a long name is shifted left to stay inside the chart.
  const labelX = Math.min(0, room - label.length * 6.5)
  return (
    <svg
      width={Math.max(room, 1)}
      height={height + 20}
      role="img"
      aria-label={`colour scale: ${label} from ${formatNumber(lo)} to ${formatNumber(hi)}`}
      className="pointer-events-none absolute overflow-visible"
      style={{ left, top: top - 20 }}
    >
      <defs>
        <linearGradient id={id} x1="0" y1="1" x2="0" y2="0">
          {stops.map((c, i) => (
            <stop key={i} offset={i / (stops.length - 1)} stopColor={c} />
          ))}
        </linearGradient>
      </defs>
      <text x={labelX} y={11} className="fill-muted-foreground text-[11px]">
        {label}
      </text>
      <g transform="translate(0 20)">
        <rect
          x={0}
          y={0}
          width={BAR.width}
          height={height}
          fill={`url(#${id})`}
          className="stroke-border"
          strokeWidth={1}
        />
        {ticks
          .filter((v) => v >= Math.min(lo, hi) - 1e-12 && v <= Math.max(lo, hi) + 1e-12)
          .map((v) => (
            <g key={v}>
              <line x1={BAR.width} x2={tickX} y1={at(v)} y2={at(v)} className="stroke-muted-foreground" />
              <text
                x={tickX + BAR.labelGap}
                y={at(v)}
                dominantBaseline="middle"
                className="fill-muted-foreground text-[11px] tabular-nums"
              >
                {formatNumber(v)}
              </text>
            </g>
          ))}
      </g>
    </svg>
  )
}
