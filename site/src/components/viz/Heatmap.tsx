import { useMemo } from 'react'
import { useTheme } from '@/components/theme-provider'
import type { Series } from '@/generated/contracts'
import { EChart, type EChartClick } from './EChart'
import type { Handle } from './handles'
import { chrome, diverging, MARKER_SHAPES, seriesColor, sequential, type Mode } from './palette'
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

type OverlaySeries = { id: string; data: number[][]; style: Record<string, unknown> }

/**
 * The sequential ramp is slot 0's hue (blue), so on a sequential heatmap overlays start at slot 1; otherwise a blue
 * path would vanish into the grid.
 */
function overlaySeries(overlay: HeatmapOverlay[], mode: Mode, scale: HeatmapProps['scale']): OverlaySeries[] {
  const shift = scale === 'sequential' ? 1 : 0
  const ink = chrome(mode).ink
  const surface = chrome(mode).surface
  return overlay.flatMap((s, index) => {
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

  const overlays = useMemo(() => overlaySeries(overlay, mode, scale), [overlay, mode, scale])
  // Structure only. The option below depends on this string, not on overlay data.
  const overlayKey = JSON.stringify(overlays.map((s) => [s.id, s.style]))

  // Everything except overlay data and the marker. Depends on values, not arrays, so inline props stay cached.
  const option = useMemo(() => {
    const structure = JSON.parse(overlayKey) as [string, Record<string, unknown>][]
    const flat = z.flat()
    const lo = rangeLo ?? Math.min(...flat)
    const hi = rangeHi ?? Math.max(...flat)
    const dx = x.length > 1 ? x[1] - x[0] : 1
    const dy = y.length > 1 ? y[1] - y[0] : 1
    const cells = y.flatMap((yv, i) => x.map((xv, j) => [xv, yv, z[i][j]]))
    return {
      grid: { right: 72 },
      legend: { show: structure.length > 1, data: structure.map(([, style]) => style.name as string) },
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
        dimension: 2,
        seriesIndex: 0,
        calculable: false,
        orient: 'vertical',
        right: 0,
        top: 'middle',
        itemHeight: height - 140,
        itemWidth: 10,
        inRange: { color: scale === 'diverging' ? diverging(mode) : sequential },
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
              style: { fill: api.visual('color'), stroke: 'none' },
            }
          },
          z: 1,
        },
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
  }, [x, y, z, xLabel, yLabel, scale, rangeLo, rangeHi, overlayKey, valueLabel, mode, height, clickable])

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

  return (
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
}
