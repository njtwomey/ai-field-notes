import { useCallback, useEffect, useId, useMemo, useState } from 'react'
import { chrome, MARKER_SHAPES, seriesColor, type Mode } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { EChart, type PlotPointer } from './EChart'
import { formatNumber, formatPower } from './format'
import { useChartHeight, useElementSize, useFrameData, useFrameHover, type HoverRow } from './frame'
import type { Handle } from './handles'
import { GRID, LINE_WIDTH, MARKER_SIZE } from './theme'
import { vectorEnds, vectorLines, type Box, type Vector } from './vectors'
import { niceRange, tickLabelWidth, useSubplot } from './subplot-context'
import { drawnViewport, equalUnits, extentOf, fixedUnits, useViewport, type Range } from './viewport'
import { ViewportControls } from './ViewportControls'

/** One plotted series. Colours come from categorical slots, fixed per entity. */
export type XYSeries = {
  name: string
  /** `area` is a line filled down to y = 0; `bar` draws a bar at each x (e.g. a histogram). */
  type: 'line' | 'scatter' | 'bar' | 'area'
  x: ArrayLike<number>
  y: ArrayLike<number>
  /** Scatter only: a categorical group per point, drawn in slot g with marker shape g. */
  group?: readonly number[] | null
  /** Names for group indices, shown in the legend. */
  groupNames?: readonly string[]
  /**
   * Fixed categorical slot. Defaults to the series' position among the non-muted, non-emphasised series. Colour follows
   * the entity, so pass it when series can be toggled. There are 8 slots; fold further series into a muted one.
   */
  slot?: number
  /** Lines only: dashed, e.g. a reference curve. */
  dashed?: boolean
  /** Lines only: fill the region under the line down to y = 0 (same as `type: 'area'`). */
  area?: boolean
  /** Larger ink-coloured marks, e.g. centroids. Not a category, so no palette slot. */
  emphasis?: boolean
  /** Background marks in the muted chrome colour, e.g. the unremarkable majority. Not a category. */
  muted?: boolean
  /**
   * Light marks. Lines: thin and translucent, for many draws of one thing (chains, paths). Bars: translucent in their
   * slot's colour, e.g. a histogram of samples under the density it estimates. Scatter: small marks, for many points
   * in a small panel (a scatter matrix).
   */
  thin?: boolean
  /** Lines only: mark every vertex, e.g. each step of an optimiser. */
  showPoints?: boolean
  /** A data colour from a scale helper (e.g. `useScaleColor`) in place of the slot, for a point on an ordered scale. */
  color?: string
  /**
   * Bars only: a histogram's bars, touching (no gaps) and translucent, so two overlapping histograms (one per class)
   * show through each other.
   */
  histogram?: boolean
  /** Bars and ungrouped scatter only: one colour per point, e.g. each bar coloured by its class. */
  pointColors?: readonly string[]
  /**
   * Lines only: left out of hover (no tooltip row, no readout, no emphasis), e.g. hundreds of curves broken by NaN gaps
   * under the summary curves that the tooltip should report.
   */
  silent?: boolean
}

/** How held axes follow new data (see `XYChartProps.holdFit`). */
export type HoldFit = 'initial' | 'union'

export type Segment = { from: [number, number]; to: [number, number] }

export type XYChartProps = {
  series: readonly XYSeries[]
  /**
   * Series that move on every drag (a tangent, a cursor, a circle that follows a handle): drawn like `series` but sent
   * to ECharts as a patch, so moving them does not redraw the rest. Keep their number and styles fixed; only their
   * data should change. They do not widen the fitted axes (give `xRange` and `yRange`), are left out of the hover
   * readout and the tooltip, and do not support `group`.
   */
  live?: readonly XYSeries[]
  xLabel?: string
  yLabel?: string
  /** The unzoomed x range; omitted, the axis fits the data with nice ticks. */
  xRange?: [number, number]
  /** The unzoomed y range; either end may be undefined to fit that end to the data. */
  yRange?: [number | undefined, number | undefined]
  /** Logarithmic axes. Values must be positive; zoom and pan work in log space. */
  xLog?: boolean
  yLog?: boolean
  /** Thin muted segments under the data, e.g. residuals. */
  segments?: readonly Segment[]
  /** Arrows, e.g. a weight vector normal to a decision boundary. Ink unless a vector sets `slot`. */
  vectors?: readonly Vector[]
  /**
   * `fit` (default): each axis fits the data. `equal`: equal pixel length per unit on both axes, so shapes and angles
   * are true; the chart keeps its size and widens one axis, never hiding data. With both `xRange` and `yRange` given in
   * full, the ranges are kept exactly and the plot area takes their aspect instead, centred (a unit square such as a
   * ROC curve's is drawn square); zooming keeps that plot area. In a Subplots panel with a shared axis,
   * the other axis takes its units from the panel's size, or falls back to `fit` (noted in the toolbar) when that
   * would hide data.
   */
  aspect?: 'equal' | 'fit'
  /** @deprecated Same as `aspect="equal"`. */
  equalAspect?: boolean
  /**
   * Refit the axes whenever the data changes (default true). False holds them: fitted once, then kept while the data
   * changes (e.g. while a parameter slider moves), so a change of shape reads as one. The fit button refits; zoom and
   * pan still work. Inside Subplots the grid or the Panel can set it for every chart.
   */
  rescaleOnChange?: boolean
  /** Refit held axes when this changes, e.g. the family or dataset, but not a parameter. */
  axisKey?: string | number
  /** How held axes follow new data: `initial` (default) keeps the first fit; `union` grows to include new data. */
  holdFit?: HoldFit
  /** Ticks and labels on the x-axis at integers only, e.g. classes 1…K drawn as bars on [0.5, K + 0.5]. */
  integerX?: boolean
  /** Hide axes, ticks and grid lines, e.g. for a grid of people where coordinates mean nothing. */
  bare?: boolean
  /** Draggable handles bound to parameters. See handles.ts. */
  handles?: Handle[]
  /** Clicks anywhere in the plot area, in data coordinates. Prefer handles. */
  onPlotClick?: (point: [number, number]) => void
  /** Hover, click and leave positions in data coordinates. */
  onPointer?: (event: PlotPointer) => void
  /** Pixels. Inside a Figure the frame sets the height and this is ignored. */
  height?: number
  /** The zoom and pan toolbar (default true). Pinch and Ctrl/⌘-scroll zoom whenever it is on. */
  zoom?: boolean
  /** Charts sharing a hover group share the hovered x (e.g. the panels of one trace). */
  hoverGroup?: string
  /** Force the legend on or off; by default it shows when there are two or more names. */
  legend?: boolean
  /**
   * Rectangle brushing: drag in the plot to select a region, reported in data coordinates on release (null on a plain
   * click). Draw the selection yourself, e.g. by fading unselected points and adding the rectangle as a live series.
   */
  onBrush?: (rect: { x: Range; y: Range } | null) => void
  /** `canvas` for thousands of marks (a scatter matrix); default SVG. Fixed at mount. */
  renderer?: 'svg' | 'canvas'
  /** Formats x and y values in tooltips and hover readouts (default `formatNumber`). */
  formatX?: (v: number) => string
  formatY?: (v: number) => string
  ariaLabel?: string
}

/** The toolbar row (24 px) and the gap under it (4 px): a panel's height outside its ECharts box. */
const TOOLBAR_CHROME = 28

const BARE_GRID = { left: 8, right: 8, top: 32, bottom: 8 }
const NONE: readonly never[] = []

/** Series with a line-like shape: the chart hovers these by x (axis trigger) rather than by item. */
const lineLike = (s: XYSeries) => s.type !== 'scatter'

/** The index of the point in `x` nearest to `at`. */
function nearestIndex(x: ArrayLike<number>, at: number): number {
  let best = -1
  let distance = Infinity
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - at)
    if (d < distance) [best, distance] = [i, d]
  }
  return best
}

/** Colour and style decisions for one series, shared by the option and the hover readout. */
function seriesColour(s: XYSeries, slot: number, mode: Mode): string {
  const c = chrome(mode)
  if (s.emphasis) return c.ink
  if (s.muted) return s.type === 'bar' ? c.grid : c.muted
  return s.color ?? seriesColor(mode, slot)
}

/**
 * Line, area, bar and scatter series on shared numeric axes, zoomable and pannable. Grouped scatter series split into
 * one series per group, coloured by slot and shaped by `MARKER_SHAPES`, so identity never relies on colour alone.
 * Line-like charts hover by x: a pointer line follows the cursor, each series' nearest point is marked, and the tooltip
 * lists every series' value there.
 */
export function XYChart({
  series,
  live = NONE,
  xLabel,
  yLabel,
  xRange,
  yRange,
  xLog = false,
  yLog = false,
  segments = NONE,
  vectors = NONE,
  aspect = 'fit',
  equalAspect = false,
  rescaleOnChange,
  axisKey,
  holdFit,
  integerX = false,
  bare = false,
  handles,
  onPlotClick,
  onPointer,
  onBrush,
  renderer,
  height: ownHeight,
  zoom = true,
  hoverGroup,
  legend: legendShown,
  formatX = formatNumber,
  formatY = formatNumber,
  ariaLabel,
}: XYChartProps) {
  const { resolved: mode } = useTheme()
  const height = useChartHeight(ownHeight)
  // Inside a Subplots grid: shared axes, aligned margins, linked hover.
  const sub = useSubplot()
  const [box, size] = useElementSize<HTMLDivElement>()

  // Live series: the option holds their styles (with no data), keyed by value so that new data alone keeps the option;
  // their data goes in the patch.
  const liveKey = JSON.stringify(live.map(({ x: _x, y: _y, ...style }) => style))
  // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the styles' value, not the array's identity
  const liveStyles = useMemo((): XYSeries[] => live.map((s) => ({ ...s, x: [], y: [] })), [liveKey])
  const patch = useMemo(
    () =>
      live.length
        ? { series: live.map((s, i) => ({ id: `__live${i}`, data: Array.from(s.x, (x, k) => [x, s.y[k]]) })) }
        : undefined,
    [live],
  )

  // Slots: default slots count categories only; muted and emphasised series take none.
  const slots = useMemo(() => {
    let category = 0
    return series.map((s) => s.slot ?? (s.muted || s.emphasis ? 0 : category++))
  }, [series])

  // The data's extent: non-live series and vector ends. Live series (a tangent, a circle) never move the axes.
  const [ex0, ex1] = useMemo(
    () => extentOf([...series.map((s) => s.x), vectorEnds(vectors).map((p) => p[0])], xLog) ?? [],
    [series, vectors, xLog],
  )
  const [ey0, ey1] = useMemo(
    () => extentOf([...series.map((s) => s.y), vectorEnds(vectors).map((p) => p[1])], yLog) ?? [],
    [series, vectors, yLog],
  )
  const fromZero = !yLog && series.some((s) => s.type === 'bar' || s.type === 'area' || s.area)

  // The unzoomed ranges: explicit ends win; otherwise the data's extent rounded out to whole ticks (bars and areas
  // include zero). The chart fits its own axes rather than leaving it to ECharts, which would also fit live series.
  // `drawn` (what ECharts reports) is only a fallback for a chart with no finite data.
  const [drawn, setDrawn] = useState<{ x?: Range; y?: Range }>({})
  const [x0, x1] = xRange ?? []
  const [y0, y1] = yRange ?? []
  const fitX = useMemo(
    (): Range | undefined => (ex0 !== undefined && ex1 !== undefined ? niceRange([ex0, ex1], xLog) : undefined),
    [ex0, ex1, xLog],
  )
  const fitY = useMemo(
    (): Range | undefined =>
      ey0 !== undefined && ey1 !== undefined
        ? niceRange(fromZero ? [Math.min(0, ey0), Math.max(0, ey1)] : [ey0, ey1], yLog)
        : undefined,
    [ey0, ey1, yLog, fromZero],
  )
  // Held axes: the fit taken on mount (or when `axisKey` changes), kept, or grown to include new data with `union`.
  const hold = !(rescaleOnChange ?? sub?.hold.rescaleOnChange ?? true)
  const holdMode = holdFit ?? sub?.hold.holdFit ?? 'initial'
  const holdKey = axisKey ?? sub?.hold.axisKey
  const [held, setHeld] = useState<{ key: unknown; x?: Range; y?: Range } | null>(null)
  // Updated during render (React's pattern for state derived from props), not in an effect, so it never lags a frame.
  let kept = held
  if (hold) {
    if (!held || held.key !== holdKey) kept = { key: holdKey, x: fitX, y: fitY }
    else if (holdMode === 'union') {
      const x = union(held.x, fitX)
      const y = union(held.y, fitY)
      if (!same(x, held.x) || !same(y, held.y)) kept = { key: holdKey, x, y }
    } else if ((!held.x && fitX) || (!held.y && fitY)) kept = { key: holdKey, x: held.x ?? fitX, y: held.y ?? fitY }
    if (kept !== held) setHeld(kept)
  } else kept = null
  const baseX = kept?.x ?? fitX
  const baseY = kept?.y ?? fitY

  const own = (lo: number | undefined, hi: number | undefined, fit?: Range, fallback?: Range): Range | undefined =>
    lo !== undefined && hi !== undefined ? [lo, hi] : fit ? [lo ?? fit[0], hi ?? fit[1]] : fallback
  const fitted = {
    x: sub?.shared.x ? sub.fitted.x : own(x0, x1, baseX, drawn.x),
    y: sub?.shared.y ? sub.fitted.y : own(y0, y1, baseY, drawn.y),
  }
  const viewport = useViewport(fitted, { x: xLog, y: yLog }, sub?.stores)
  const resetKey = sub?.resetKey ?? 0
  const reset = viewport.reset
  useEffect(() => {
    if (!resetKey) return
    reset()
    setHeld(null)
  }, [resetKey, reset])

  // The ranges handed to ECharts: zoomed, else shared, explicit or fitted.
  let [vx0, vx1] = viewport.shown.x ?? [x0, x1]
  let [vy0, vy1] = viewport.shown.y ?? [y0, y1]
  const base = sub ? sub.margins : bare ? BARE_GRID : GRID
  const equal = aspect === 'equal' || equalAspect || sub?.aspect === 'equal'
  // Equal units on fully explicit ranges (e.g. ROC and PR on [0, 1]²) keep those ranges exactly: the plot area takes
  // their aspect (a square for a unit square) and is centred in the chart's box, as a Heatmap's is. Only a chart on
  // its own; in a Subplots grid the grid sizes the panel.
  const fixedEqual =
    equal && !sub && !xLog && !yLog && x0 !== undefined && x1 !== undefined && y0 !== undefined && y1 !== undefined
  let [padX, padY] = [0, 0]
  if (fixedEqual && size.width > 0 && size.height > 0 && x1 !== x0 && y1 !== y0) {
    const w = size.width - base.left - base.right
    const h = size.height - base.top - base.bottom
    const ratio = Math.abs(y1 - y0) / Math.abs(x1 - x0)
    if (w * ratio <= h) padY = Math.max(0, Math.round((h - w * ratio) / 2))
    else padX = Math.max(0, Math.round((w - h / ratio) / 2))
  }
  const grid = useMemo(
    () => ({ left: base.left + padX, right: base.right + padX, top: base.top + padY, bottom: base.bottom + padY }),
    [base.left, base.right, base.top, base.bottom, padX, padY],
  )
  const inGrid = !!sub
  const dense = !!sub?.dense
  const hideX = !!sub && !sub.labels.x
  const hideY = !!sub && !sub.labels.y
  // An axis of its own inside a shared grid (the diagonal of a scatter matrix) may keep its name but drop its ticks.
  const noXTicks = !!sub && sub.ticks?.x === false
  const noYTicks = !!sub && sub.ticks?.y === false
  const yNameGap = sub ? sub.yNameGap : 40

  // Equal units only ever widen an axis, so they never hide data. On its own a chart widens whichever axis has room.
  // In a Subplots grid a shared axis keeps its range and the other axis takes its units from the panel's size; if that
  // range would not hold the data, the panel falls back to fitted, unequal axes and says so.
  let unequal = false
  if (equal && size.width > 0 && size.height > 0) {
    const plot = { width: size.width - grid.left - grid.right, height: size.height - grid.top - grid.bottom }
    const sx = !!sub?.shared.x
    const sy = !!sub?.shared.y
    if (sub?.sizedEqual) {
      // The grid sized this panel from its fitted ranges, so y follows from x and the pixel size: always equal.
      const r = fixedUnits(viewport.shown.x, viewport.shown.y, plot, 'x')
      if (r) [[vx0, vx1], [vy0, vy1]] = r
    } else if (sx && sy) unequal = true
    else if (sx || sy) {
      const r = fixedUnits(viewport.shown.x, viewport.shown.y, plot, sx ? 'x' : 'y')
      const derived = r?.[sx ? 1 : 0]
      const need = viewport.shown[sx ? 'y' : 'x']
      const slack = need ? (need[1] - need[0]) * 1e-9 : 0
      if (r && derived && need && derived[0] <= need[0] + slack && derived[1] >= need[1] - slack)
        [[vx0, vx1], [vy0, vy1]] = r
      else unequal = true
    } else if (!fixedEqual || viewport.view.x || viewport.view.y) {
      // Fixed ranges already have equal units in their letterboxed plot area until the reader zooms.
      const r = equalUnits(viewport.shown.x, viewport.shown.y, plot)
      if (r) [[vx0, vx1], [vy0, vy1]] = r
    }
  }

  // The box the plot shows now, for clipping vectors and for the toolbar's range fields.
  const bx0 = vx0 ?? drawn.x?.[0]
  const bx1 = vx1 ?? drawn.x?.[1]
  const by0 = vy0 ?? drawn.y?.[0]
  const by1 = vy1 ?? drawn.y?.[1]
  const plotBox = useMemo(
    (): Box | undefined =>
      bx0 !== undefined && bx1 !== undefined && by0 !== undefined && by1 !== undefined
        ? { x: [bx0, bx1], y: [by0, by1], xLog, yLog }
        : undefined,
    [bx0, bx1, by0, by1, xLog, yLog],
  )

  const explicitX = (x0 !== undefined && x1 !== undefined) || !!sub?.shared.x
  const { option, hoverSeries } = useMemo(() => {
    const c = chrome(mode)
    const out: Record<string, unknown>[] = []
    const hoverSeries: { label: string; color: string; x: ArrayLike<number>; y: ArrayLike<number> }[] = []
    ;[...series, ...liveStyles].forEach((s, index) => {
      const liveIndex = index - series.length
      const slot = liveIndex >= 0 ? (s.slot ?? 0) : slots[index]
      if (liveIndex >= 0) {
        const before = out.length
        drawSeries(s, slot, false)
        for (let k = before; k < out.length; k++) out[k].id = `__live${liveIndex}`
        return
      }
      drawSeries(s, slot, true)
    })
    function drawSeries(s: XYSeries, slot: number, hover: boolean) {
      if (s.type === 'scatter' && s.group) {
        const group = s.group
        const groups = [...new Set(Array.from(group))].sort((a, b) => a - b)
        for (const g of groups) {
          const data: number[][] = []
          for (let i = 0; i < s.x.length; i++) if (group[i] === g) data.push([s.x[i], s.y[i]])
          out.push(
            scatter(s.groupNames?.[g] ?? `${s.name} ${g}`, data, seriesColor(mode, g), g, s.emphasis, mode, s.thin),
          )
        }
        return
      }
      const points = Array.from(s.x, (x, i) => [x, s.y[i]])
      const data = s.pointColors
        ? points.map((value, i) => ({ value, itemStyle: { color: s.pointColors![i] } }))
        : points
      const color = seriesColour(s, slot, mode)
      if (hover && lineLike(s) && !s.silent) hoverSeries.push({ label: s.name, color, x: s.x, y: s.y })
      if (s.type === 'bar') {
        out.push({
          name: s.name,
          type: 'bar',
          data,
          barWidth: s.histogram ? '100%' : '92%',
          barGap: '-100%',
          itemStyle: {
            color,
            opacity: s.histogram ? 0.6 : s.thin ? 0.35 : 1,
            borderRadius: s.histogram ? 0 : [2, 2, 0, 0],
          },
          emphasis: { disabled: true },
          z: 1,
        })
      } else if (s.type === 'scatter') {
        const marks = scatter(s.name, points, color, s.emphasis ? 3 : 0, s.emphasis, mode, s.thin)
        // Marks coloured along a scale get an ink outline, since they may match their surroundings.
        out.push(
          s.pointColors
            ? { ...marks, data, itemStyle: { ...marks.itemStyle, color: c.ink, borderColor: c.ink, borderWidth: 1 } }
            : marks,
        )
      } else {
        const faint = s.muted || s.thin
        out.push({
          ...(s.silent ? { id: `__silent${out.length}`, silent: true } : {}),
          name: s.name,
          type: 'line',
          data,
          showSymbol: !!s.showPoints,
          symbolSize: s.showPoints ? 5 : 7,
          smooth: false,
          lineStyle: {
            width: faint ? 1 : LINE_WIDTH,
            color,
            opacity: s.thin ? 0.45 : s.muted ? 0.7 : 1,
            type: s.dashed ? 'dashed' : 'solid',
          },
          itemStyle: { color },
          emphasis: { focus: 'none', scale: 1.4 },
          ...(s.area || s.type === 'area' ? { areaStyle: { color, opacity: 0.25 } } : {}),
          z: faint ? 2 : 3,
        })
      }
    }
    if (segments.length) {
      out.unshift({
        name: '__segments',
        type: 'line',
        data: segments.flatMap((s) => [s.from, s.to, [null, null]]),
        connectNulls: false,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 1, color: c.muted, opacity: 0.6 },
        tooltip: { show: false },
        z: 1,
      })
    }
    if (vectors.length) {
      // Invisible vector ends, so axes that fit the data also fit every arrow.
      out.push({
        name: '__vector-ends',
        type: 'scatter',
        data: vectorEnds(vectors),
        symbolSize: 0,
        silent: true,
        tooltip: { show: false },
      })
      // markLine orients arrowheads in screen space, so they point correctly whatever the axis scales. Vectors are
      // clipped to the box shown now (zoomed, explicit or fitted), since ECharts drops a markLine that leaves it.
      out.push({
        name: '__vectors',
        type: 'line',
        data: [],
        silent: true,
        tooltip: { show: false },
        markLine: {
          silent: true,
          symbol: ['none', 'arrow'],
          symbolSize: 10,
          label: { show: false },
          lineStyle: { color: c.ink, width: LINE_WIDTH, type: 'solid' },
          animation: false,
          data: vectorLines(vectors, mode, plotBox),
        },
        z: 6,
      })
    }
    // Series sharing a name (e.g. several draws of one thing) get one legend entry, which toggles them together.
    const names = [...new Set(out.filter((s) => !String(s.name).startsWith('__')).map((s) => s.name as string))]
    const byAxis = hoverSeries.length > 0
    const valueAxis = (log: boolean) =>
      log ? { type: 'log', axisLabel: { formatter: formatPower } } : { type: 'value' }
    const option = {
      // ECharts 6 shrinks the grid to fit axis labels and names (`outerBounds`). Where the margins are set here (equal
      // units, a Subplots grid's aligned margins) that must not happen, or units and alignment drift by a few pixels.
      ...(equal || bare || inGrid ? { grid: { ...grid, outerBoundsMode: 'none' } } : {}),
      legend: { data: names, show: legendShown ?? (names.length > 1 && !dense) },
      tooltip: byAxis
        ? {
            trigger: 'axis',
            axisPointer: { type: 'line', snap: false, label: { show: false } },
            formatter: (params: AxisParam[]) => axisTooltip(params, xLabel, formatX, formatY),
          }
        : {
            trigger: 'item',
            formatter: (p: { seriesName: string; value: number[]; marker: string }) =>
              `${p.marker}${escape(p.seriesName)}<br/>(${formatX(p.value[0])}, ${formatY(p.value[1])})`,
          },
      xAxis: {
        ...valueAxis(xLog),
        name: xLabel,
        min: vx0,
        max: vx1,
        scale: true,
        // ECharts 6 widens a value axis so bars fit (`containShape`); an explicit or shared range must stay exact, so
        // that it lines up with the panels it is shared with.
        ...(explicitX ? { containShape: false } : {}),
        show: !bare,
        // Ticks every half unit (ECharts starts them at the axis minimum, a half-integer), labelled at integers only.
        ...(integerX
          ? {
              interval: 0.5,
              axisLabel: { formatter: (v: number) => (Number.isInteger(v) ? String(v) : '') },
              splitLine: { show: false },
            }
          : {}),
        // Inner panels of a shared x show no tick labels or name; the bottom row carries them.
        ...(hideX ? { name: undefined, axisLabel: { show: false } } : noXTicks ? { axisLabel: { show: false } } : {}),
        // A dense grid's panels sit nearly edge to edge: the end labels align inwards rather than spill into a neighbour.
        ...(dense && !hideX && !noXTicks && !integerX
          ? { axisLabel: { alignMinLabel: 'left', alignMaxLabel: 'right' } }
          : {}),
      },
      yAxis: {
        ...valueAxis(yLog),
        name: yLabel,
        min: vy0,
        max: vy1,
        scale: true,
        nameGap: yNameGap,
        show: !bare,
        ...(hideY ? { name: undefined, axisLabel: { show: false } } : noYTicks ? { axisLabel: { show: false } } : {}),
      },
      series: out,
    }
    return { option, hoverSeries }
  }, [
    series,
    liveStyles,
    slots,
    segments,
    vectors,
    plotBox,
    xLabel,
    yLabel,
    vx0,
    vx1,
    vy0,
    vy1,
    xLog,
    yLog,
    equal,
    bare,
    integerX,
    legendShown,
    formatX,
    formatY,
    mode,
    inGrid,
    dense,
    grid,
    hideX,
    hideY,
    noXTicks,
    noYTicks,
    yNameGap,
    explicitX,
  ])

  // The frame's hover readout: every line-like series' value at the hovered x.
  const setHover = useFrameHover()
  const onAxisHover = useCallback(
    (x: number | null) => {
      if (x === null) return setHover(null)
      const rows: HoverRow[] = []
      const seen = new Set<string>()
      for (const s of hoverSeries) {
        if (seen.has(s.label)) continue
        seen.add(s.label)
        const i = nearestIndex(s.x, x)
        if (i >= 0) rows.push({ label: s.label, value: formatY(s.y[i]), color: s.color })
      }
      setHover({ at: `${xLabel ?? 'x'} = ${formatX(x)}`, rows })
    },
    [hoverSeries, setHover, xLabel, formatX, formatY],
  )

  useFrameData(
    useCallback(
      () => ({
        kind: 'xy',
        xLabel,
        yLabel,
        series: series.map((s) => ({
          name: s.name,
          type: s.type,
          x: Array.from(s.x),
          y: Array.from(s.y),
          ...(s.group ? { group: Array.from(s.group) } : {}),
        })),
      }),
      [series, xLabel, yLabel],
    ),
  )

  const onExtents = (e: { x: Range; y: Range }) =>
    setDrawn((d) => {
      const x = viewport.view.x ? d.x : e.x
      const y = viewport.view.y ? d.y : e.y
      return same(x, d.x) && same(y, d.y) ? d : { x, y }
    })

  // The toolbar works on the drawn ranges, so its fields show what the plot shows; equal units zoom both axes together.
  const drawnControls = drawnViewport(
    viewport,
    { x: plotBox?.x, y: plotBox?.y },
    { x: xLog, y: yLog },
    equal && !unequal,
  )
  // With held axes the fit button also refits to the data now, so it is always available.
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
  const toolbarAxes = (['x', 'y'] as const).filter((axis) => !sub || sub.controls[axis])

  // Report this panel's extents and tick label width to its Subplots grid, and its unzoomed ranges for sizing.
  const [fx0, fx1] = fitted.x ?? []
  const [fy0, fy1] = own(y0, y1, baseY, drawn.y) ?? []
  const reportTo = sub?.report
  const panelKey = useId()
  // Measured on the y range the chart is asked to show (zoomed or fitted), never the range equal units derive from the
  // panel's pixel size: that size follows from the margins this width sets, so measuring it would feed the grid's output
  // back into its input. Rounded up to 8 px, so that a small change of labels does not move the margins.
  const labelWidth = Math.ceil(tickLabelWidth(viewport.shown.y ?? drawn.y, yLog) / 8) * 8
  useEffect(() => {
    if (!reportTo) return
    const axis = (explicit: [number | undefined, number | undefined], lo?: number, hi?: number) =>
      explicit[0] !== undefined && explicit[1] !== undefined
        ? { range: [explicit[0], explicit[1]] as Range, explicit: true }
        : lo !== undefined && hi !== undefined
          ? { range: [explicit[0] ?? lo, explicit[1] ?? hi] as Range, explicit: false }
          : undefined
    // Held axes report their held range, so a shared axis holds too.
    const heldAxis = (explicit: [number | undefined, number | undefined], r?: Range) =>
      explicit[0] !== undefined && explicit[1] !== undefined
        ? { range: [explicit[0], explicit[1]] as Range, explicit: true }
        : r
          ? { range: r, explicit: true }
          : undefined
    reportTo(panelKey, {
      x: hold ? heldAxis([x0, x1], baseX) : axis([x0, x1], ex0, ex1),
      y: hold ? heldAxis([y0, y1], baseY) : axis([y0, y1], ey0, ey1),
      labelWidth,
      equal,
      fit: {
        x: fx0 !== undefined && fx1 !== undefined ? [fx0, fx1] : undefined,
        y: fy0 !== undefined && fy1 !== undefined ? [fy0, fy1] : undefined,
      },
      chrome: zoom ? TOOLBAR_CHROME : 0,
    })
  }, [
    reportTo,
    panelKey,
    x0,
    x1,
    y0,
    y1,
    ex0,
    ex1,
    ey0,
    ey1,
    labelWidth,
    hold,
    baseX,
    baseY,
    equal,
    zoom,
    fx0,
    fx1,
    fy0,
    fy1,
  ])
  useEffect(() => () => reportTo?.(panelKey, null), [reportTo, panelKey])

  return (
    <div className="flex w-full shrink-0 flex-col gap-1 overflow-hidden" style={{ height }}>
      {zoom && (
        <ViewportControls
          viewport={controls}
          log={{ x: xLog, y: yLog }}
          axes={toolbarAxes}
          bothAxes={!sub}
          note={unequal ? 'units not equal' : undefined}
          // A fixed-height toolbar in a grid, so the grid can size the plot below it exactly (TOOLBAR_CHROME).
          className={sub ? 'h-6 flex-nowrap overflow-x-auto overflow-y-hidden px-1' : 'px-1'}
        />
      )}
      <div ref={box} className="relative min-h-0 flex-1">
        <EChart
          option={option}
          patch={patch}
          height="fill"
          className="absolute inset-0"
          ariaLabel={ariaLabel}
          handles={handles}
          onPlotClick={onPlotClick}
          onPointer={onPointer}
          onBrush={onBrush}
          renderer={renderer}
          onExtents={onExtents}
          onWheelZoom={zoom ? (factor, [x, y]) => controls.zoom(factor, ['x', 'y'], { x, y }) : undefined}
          onAxisHover={hoverSeries.length ? onAxisHover : undefined}
          hoverGroup={hoverGroup ?? sub?.hoverGroup}
          onDragChange={sub?.setDragging}
        />
      </div>
    </div>
  )
}

/** The smallest range holding both. */
const union = (a: Range | undefined, b: Range | undefined): Range | undefined =>
  a && b ? [Math.min(a[0], b[0]), Math.max(a[1], b[1])] : (a ?? b)

const same = (a: Range | undefined, b: Range | undefined) => a === b || (!!a && !!b && a[0] === b[0] && a[1] === b[1])

type AxisParam = {
  seriesId?: string
  seriesName: string
  seriesType: string
  marker: string
  value: number[] | { value: number[] }
  axisValue: number
}

const escape = (s: string) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!)

/** The axis tooltip: the hovered x, then each line-like series' value at its nearest point (one row per name). */
function axisTooltip(
  params: AxisParam[],
  xLabel: string | undefined,
  fx: (v: number) => string,
  fy: (v: number) => string,
) {
  if (!params.length) return ''
  const seen = new Set<string>()
  const rows = params.flatMap((p) => {
    if (p.seriesName.startsWith('__') || /^__(live|silent)/.test(p.seriesId ?? '') || p.seriesType === 'scatter')
      return []
    if (seen.has(p.seriesName)) return []
    seen.add(p.seriesName)
    const value = Array.isArray(p.value) ? p.value : p.value.value
    return [
      `<div style="display:flex;gap:12px;justify-content:space-between"><span>${p.marker}${escape(p.seriesName)}</span>` +
        `<span style="font-variant-numeric:tabular-nums;font-weight:500">${fy(value[1])}</span></div>`,
    ]
  })
  const head = `<div style="opacity:0.7;margin-bottom:2px">${escape(xLabel ?? 'x')} = ${fx(params[0].axisValue)}</div>`
  return head + rows.join('')
}

function scatter(
  name: string,
  data: unknown[],
  color: string,
  shape: number,
  emphasis: boolean | undefined,
  mode: Mode,
  small = false,
) {
  return {
    name,
    type: 'scatter',
    data,
    symbol: MARKER_SHAPES[shape % MARKER_SHAPES.length],
    symbolSize: emphasis ? (small ? 11 : 16) : small ? 5 : MARKER_SIZE,
    itemStyle: {
      color,
      opacity: emphasis ? 1 : small ? 0.75 : 0.85,
      // A surface-coloured ring separates overlapping marks.
      borderColor: chrome(mode).surface,
      borderWidth: emphasis ? 2 : small ? 0.5 : 1,
    },
    z: emphasis ? 5 : 2,
  }
}
