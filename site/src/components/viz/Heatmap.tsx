import { useId, useMemo } from 'react'
import { useTheme } from '@/components/theme-provider'
import type { Series } from '@/generated/contracts'
import { contourSegments } from '@/lib/math/contours'
import { EChart, type EChartClick } from './EChart'
import type { Handle } from './handles'
import {
  chrome,
  diverging,
  interpolateColors,
  MARKER_SHAPES,
  mute,
  seriesColor,
  sequential,
  type Mode,
} from './palette'
import { formatNumber, LINE_WIDTH, MARKER_SIZE } from './theme'

type CustomApi = {
  value: (dim: number) => number
  coord: (point: number[]) => number[]
  visual: (key: string) => string
}

export type HeatmapOverlay = Omit<Series, 'group'> & {
  group?: number[] | null
  groupNames?: string[]
  /** Fixed categorical slot. Defaults to the overlay's index, shifted past the heatmap's own hue (see below). */
  slot?: number
  /** Lines only: mark every vertex, e.g. each step of an optimiser. */
  showPoints?: boolean
  /** Ink-coloured diamond, e.g. an optimum. Not a category, so no palette slot. */
  emphasis?: boolean
  /**
   * Scatter only: colour each point by its value on the heatmap's own colour scale, with an ink outline so it stays
   * visible on a cell of the same colour, e.g. the true class of a point over predicted-class regions. `group` then sets
   * only the marker shape.
   */
  values?: number[]
  /**
   * Scatter only: an explicit palette colour per point (from a palette helper such as `useScaleColor`), drawn like
   * `values` with an ink outline, for points coloured on a different scale from the cells.
   */
  colors?: string[]
}

export type HeatmapProps = {
  x: number[]
  y: number[]
  /** Row-major: z[i][j] is the value at (x[j], y[i]). */
  z: number[][]
  xLabel?: string
  yLabel?: string
  /** `sequential` for magnitude, `diverging` for signed values around `midpoint`. */
  scale?: 'sequential' | 'diverging'
  range?: [number, number]
  /**
   * Label the colour bar at these values, e.g. the classes 1…K of an ordinal map. ECharts labels a continuous bar only
   * at its ends, so the bar is then drawn beside the chart with a tick at each value.
   */
  scaleTicks?: number[]
  /** Strength of the cell colours, below 1 to mute the fill towards the surface under full-strength overlays. */
  fillOpacity?: number
  /**
   * Ink contour lines at `levels`, each labelled once with its level, of `field` (a second field on the same grid,
   * e.g. an expected value over predicted-class cells) or of `z` itself. Pass a memoised object.
   */
  contours?: { levels: number[]; field?: number[][] }
  /**
   * Points or paths drawn over the grid. Their data is applied as a patch, so moving a path does not redraw the grid.
   * Changing the overlays' names, types or colours does redraw it.
   */
  overlay?: HeatmapOverlay[]
  /** A single highlighted point, e.g. the current parameter setting. Moving it does not redraw the grid. */
  marker?: [number, number]
  /** Draggable handles bound to parameters. See handles.ts. */
  handles?: Handle[]
  /** Ink arrows over the grid, e.g. a gradient at the marker. Updated without redrawing the grid. */
  vectors?: { from: [number, number]; to: [number, number] }[]
  /** Called with the centre of a clicked cell, or a clicked overlay point. Cells show a pointer cursor when set. */
  onCellClick?: (x: number, y: number) => void
  valueLabel?: string
  height?: number
  ariaLabel?: string
}

const NO_OVERLAY: HeatmapOverlay[] = []

/** Contour lines as one ink line series (segments split by null points), plus one label per level. */
function contourSeries(x: number[], y: number[], z: number[][], contours: HeatmapProps['contours'], mode: Mode) {
  if (!contours) return []
  const ink = chrome(mode).ink
  const lines: (number | null)[][] = []
  const labels: number[][] = []
  for (const level of contours.levels) {
    const segments = contourSegments(x, y, contours.field ?? z, level)
    for (const [a, b] of segments) lines.push(a, b, [null, null])
    // Label the level once, at the middle segment of its scan order.
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
      lineStyle: { color: ink, width: 1 },
      tooltip: { show: false },
      z: 2,
    },
    {
      id: '__contour-labels',
      name: '__contour-labels',
      type: 'scatter',
      data: labels,
      symbolSize: 0,
      silent: true,
      label: {
        show: true,
        formatter: (p: { value: number[] }) => formatNumber(p.value[2]),
        color: ink,
        fontSize: 10,
        backgroundColor: chrome(mode).surface,
        padding: [1, 3],
        borderRadius: 2,
      },
      tooltip: { show: false },
      z: 4,
    },
  ]
}

type OverlayPoint = number[] | { value: number[]; itemStyle: { color: string } }
type OverlaySeries = { id: string; data: OverlayPoint[]; style: Record<string, unknown> }

/**
 * The sequential ramp is slot 0's hue (blue), so on a sequential heatmap overlays start at slot 1; otherwise a blue
 * path would vanish into the grid.
 */
function overlaySeries(
  overlay: HeatmapOverlay[],
  mode: Mode,
  scale: HeatmapProps['scale'],
  lo: number,
  hi: number,
): OverlaySeries[] {
  const shift = scale === 'sequential' ? 1 : 0
  const ink = chrome(mode).ink
  const surface = chrome(mode).surface
  const stops = scale === 'diverging' ? diverging(mode) : sequential
  return overlay.flatMap((s, index): OverlaySeries[] => {
    if (s.values || s.colors) {
      const colorAt = (i: number) =>
        s.colors?.[i] ?? interpolateColors(stops, hi > lo ? ((s.values?.[i] ?? lo) - lo) / (hi - lo) : 0)
      // With values, colour carries the value and `group` only picks the marker shape. Every named group keeps a
      // series, so the legend (drawn in ink, since colour means something else) does not change as points move.
      const groups = s.group ? (s.groupNames?.map((_, g) => g) ?? [...new Set(s.group)].sort((a, b) => a - b)) : [0]
      return groups.map((g) => ({
        id: `overlay-${index}-values-${g}`,
        data: s.x.flatMap((xv, i) =>
          !s.group || s.group[i] === g
            ? [
                {
                  value: [xv, s.y[i]],
                  itemStyle: { color: colorAt(i) },
                },
              ]
            : [],
        ),
        style: {
          name: s.group ? (s.groupNames?.[g] ?? `${s.name} ${g}`) : s.name,
          type: 'scatter',
          symbol: MARKER_SHAPES[g % MARKER_SHAPES.length],
          symbolSize: MARKER_SIZE,
          itemStyle: { color: ink, borderColor: ink, borderWidth: 1 },
          clip: true,
          z: 3,
        },
      }))
    }
    const groups = s.group ? [...new Set(s.group)].sort((a, b) => a - b) : [null]
    return groups.map((g) => {
      const slot = g !== null ? g + shift : (s.slot ?? index + shift)
      const color = s.emphasis ? ink : seriesColor(mode, slot)
      const name = g === null ? s.name : (s.groupNames?.[g] ?? `${s.name} ${g}`)
      const line = s.type === 'line'
      return {
        id: `overlay-${index}-${g ?? 'all'}`,
        data: s.x.flatMap((xv, i) => (g === null || s.group![i] === g ? [[xv, s.y[i]]] : [])),
        style: {
          name,
          type: line ? 'line' : 'scatter',
          symbol: s.emphasis ? 'diamond' : line ? 'circle' : MARKER_SHAPES[slot % MARKER_SHAPES.length],
          symbolSize: s.emphasis ? 16 : line ? 5 : MARKER_SIZE,
          showSymbol: !line || !!s.showPoints,
          itemStyle: { color, borderColor: surface, borderWidth: s.emphasis ? 2 : 1 },
          lineStyle: { color, width: LINE_WIDTH },
          // Clip paths that leave the grid, e.g. a diverging optimiser.
          clip: true,
          z: s.emphasis ? 4 : 3,
        },
      }
    })
  })
}

/**
 * A value grid on numeric axes with optional overlays. Cells are a custom series rather than the ECharts category
 * heatmap, so overlays share the same continuous axes.
 */
export function Heatmap({
  x,
  y,
  z,
  xLabel,
  yLabel,
  scale = 'sequential',
  range,
  scaleTicks,
  fillOpacity = 1,
  contours,
  overlay = NO_OVERLAY,
  marker,
  vectors,
  handles,
  onCellClick,
  valueLabel = 'value',
  height = 360,
  ariaLabel,
}: HeatmapProps) {
  const { resolved: mode } = useTheme()
  const [rangeLo, rangeHi] = range ?? [undefined, undefined]
  const clickable = !!onCellClick

  // The colour scale's ends, shared by the grid and by overlays coloured by value.
  const [lo, hi] = useMemo(() => {
    if (rangeLo !== undefined && rangeHi !== undefined) return [rangeLo, rangeHi]
    const flat = z.flat()
    return [rangeLo ?? Math.min(...flat), rangeHi ?? Math.max(...flat)]
  }, [z, rangeLo, rangeHi])
  const overlays = useMemo(() => overlaySeries(overlay, mode, scale, lo, hi), [overlay, mode, scale, lo, hi])
  // Structure only. The option below depends on this string, not on overlay data.
  const overlayKey = JSON.stringify(overlays.map((s) => [s.id, s.style]))

  // Everything except overlay data and the marker. Depends on values, not arrays, so inline props stay cached.
  const option = useMemo(() => {
    const structure = JSON.parse(overlayKey) as [string, Record<string, unknown>][]
    const stops = scale === 'diverging' ? diverging(mode) : sequential
    const surface = chrome(mode).surface
    const dx = x.length > 1 ? x[1] - x[0] : 1
    const dy = y.length > 1 ? y[1] - y[0] : 1
    const cells = y.flatMap((yv, i) => x.map((xv, j) => [xv, yv, z[i][j]]))
    return {
      grid: { right: scaleTicks ? 16 : 72 },
      legend: {
        show: structure.length > 1,
        // Each entry shows its series' marker, so shapes that encode groups read correctly in the legend.
        data: structure.map(([, style]) => ({
          name: style.name as string,
          ...(style.type === 'scatter' && typeof style.symbol === 'string' ? { icon: style.symbol } : {}),
        })),
      },
      tooltip: {
        trigger: 'item',
        formatter: (p: { seriesName: string; value: number[] }) =>
          p.seriesName === '__grid'
            ? `(${formatNumber(p.value[0])}, ${formatNumber(p.value[1])})<br/>${valueLabel}: ${formatNumber(p.value[2])}`
            : `${p.seriesName}<br/>(${formatNumber(p.value[0])}, ${formatNumber(p.value[1])})`,
      },
      visualMap: {
        min: lo,
        max: hi,
        show: !scaleTicks,
        dimension: 2,
        seriesIndex: 0,
        calculable: false,
        orient: 'vertical',
        right: 0,
        top: 'middle',
        itemHeight: height - 140,
        itemWidth: 10,
        inRange: { color: stops },
        textStyle: { color: chrome(mode).muted, fontSize: 11 },
        formatter: (v: number) => formatNumber(v),
      },
      xAxis: {
        type: 'value',
        name: xLabel,
        min: x[0] - dx / 2,
        max: x[x.length - 1] + dx / 2,
        splitLine: { show: false },
        // Bounds sit half a cell outside the data; their labels would read 47.5 rather than 47.
        axisLabel: { showMinLabel: false, showMaxLabel: false },
      },
      yAxis: {
        type: 'value',
        name: yLabel,
        min: y[0] - dy / 2,
        max: y[y.length - 1] + dy / 2,
        splitLine: { show: false },
        axisLabel: { showMinLabel: false, showMaxLabel: false },
      },
      series: [
        {
          name: '__grid',
          type: 'custom',
          // Draw every cell in one pass. Above a few thousand cells ECharts otherwise paints in chunks across frames,
          // which flickers on each update (and stalls entirely in a background tab).
          progressive: 0,
          data: cells,
          encode: { x: 0, y: 1, tooltip: 2 },
          cursor: clickable ? 'pointer' : 'default',
          // Each cell is a rectangle spanning half a grid step either side of its centre, so cells tile exactly.
          renderItem: (_: unknown, api: CustomApi) => {
            const cx = api.value(0)
            const cy = api.value(1)
            const [x0, y0] = api.coord([cx - dx / 2, cy + dy / 2])
            const [x1, y1] = api.coord([cx + dx / 2, cy - dy / 2])
            return {
              type: 'rect',
              shape: { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 },
              style: {
                fill: fillOpacity < 1 ? mute(api.visual('color'), surface, fillOpacity) : api.visual('color'),
                stroke: 'none',
              },
            }
          },
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
          itemStyle: { color: chrome(mode).ink, borderColor: chrome(mode).surface, borderWidth: 2 },
          // Vectors ride on the marker series as a markLine, which orients arrowheads in screen space.
          markLine: {
            silent: true,
            symbol: ['none', 'arrow'],
            symbolSize: 10,
            label: { show: false },
            lineStyle: { color: chrome(mode).ink, width: LINE_WIDTH, type: 'solid' },
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
    xLabel,
    yLabel,
    scale,
    lo,
    hi,
    scaleTicks,
    fillOpacity,
    contours,
    overlayKey,
    valueLabel,
    mode,
    height,
    clickable,
  ])

  const [mx, my] = marker ?? [undefined, undefined]
  const patch = useMemo(
    () => ({
      series: [
        ...overlays.map((s) => ({ id: s.id, data: s.data })),
        {
          id: 'marker',
          data: mx === undefined ? [] : [[mx, my]],
          markLine: { data: (vectors ?? []).map((v) => [{ coord: v.from }, { coord: v.to }]) },
        },
      ],
    }),
    [overlays, mx, my, vectors],
  )

  // Clicks on overlay points (e.g. an optimum drawn on top of the grid) select that point too.
  const handleClick = onCellClick
    ? (e: EChartClick) => {
        if (e.seriesId === 'marker' || !Array.isArray(e.value) || e.value.length < 2) return
        onCellClick(e.value[0] as number, e.value[1] as number)
      }
    : undefined

  const chart = (
    <EChart
      option={option}
      patch={patch}
      onClick={handleClick}
      handles={handles}
      height={height}
      ariaLabel={ariaLabel}
      // One rectangle per cell: thousands of marks, which canvas draws far faster than SVG.
      renderer="canvas"
    />
  )
  if (!scaleTicks) return chart
  return (
    <div className="flex items-center">
      <div className="min-w-0 flex-1">{chart}</div>
      <ScaleBar
        stops={scale === 'diverging' ? diverging(mode) : sequential}
        lo={lo}
        hi={hi}
        ticks={scaleTicks}
        height={height}
      />
    </div>
  )
}

/** A vertical colour bar with labelled ticks, drawn as SVG beside the chart; low values at the bottom. */
function ScaleBar({
  stops,
  lo,
  hi,
  ticks,
  height,
}: {
  stops: string[]
  lo: number
  hi: number
  ticks: number[]
  height: number
}) {
  // useId can contain characters that are not valid in a url(#…) reference.
  const id = `scale${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const bar = height - 140
  const top = 70
  const at = (v: number) => top + bar * (1 - (hi > lo ? (v - lo) / (hi - lo) : 0.5))
  return (
    <svg width={40} height={height} role="img" aria-label="colour scale" className="shrink-0">
      <defs>
        <linearGradient id={id} x1="0" y1="1" x2="0" y2="0">
          {stops.map((c, i) => (
            <stop key={i} offset={i / (stops.length - 1)} stopColor={c} />
          ))}
        </linearGradient>
      </defs>
      <rect x={4} y={top} width={10} height={bar} fill={`url(#${id})`} rx={2} />
      {ticks.map((v) => (
        <g key={v}>
          <line x1={14} x2={18} y1={at(v)} y2={at(v)} className="stroke-muted-foreground" />
          <text x={21} y={at(v)} dominantBaseline="middle" className="fill-muted-foreground text-[11px]">
            {formatNumber(v)}
          </text>
        </g>
      ))}
    </svg>
  )
}
