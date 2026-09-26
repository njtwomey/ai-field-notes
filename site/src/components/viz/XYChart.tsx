import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTheme } from '@/components/theme-provider'
import type { Series } from '@/generated/contracts'
import { EChart } from './EChart'
import type { Handle } from './handles'
import { chrome, MARKER_SHAPES, seriesColor } from './palette'
import { formatNumber, LINE_WIDTH, MARKER_SIZE } from './theme'

/** A contract `Series` (group optional) plus display options that only the site needs. */
export type XYSeries = Omit<Series, 'group' | 'type'> & {
  /** `bar` draws a bar at each x, e.g. a histogram; bars take the width of the gap between x values. */
  type: Series['type'] | 'bar'
  group?: number[] | null
  /**
   * Fixed categorical slot. Defaults to the series' position among the non-muted, non-emphasised series. Colour follows
   * the entity, so pass it when filtering. There are 8 slots; fold further series into a muted "other" series.
   */
  slot?: number
  /** Draw as a dashed reference line (lines only). */
  dashed?: boolean
  /** Fill the region under a line down to y = 0, e.g. a shaded tail probability (lines only). */
  area?: boolean
  /** Larger ink-coloured markers, e.g. centroids. Not a category, so no palette slot. */
  emphasis?: boolean
  /** Background marks in the muted chrome colour, e.g. the unremarkable majority. Not a category. */
  muted?: boolean
  /** Names for group indices, shown in the legend. */
  groupNames?: string[]
}

export type Segment = { from: [number, number]; to: [number, number] }

export type XYChartProps = {
  series: XYSeries[]
  xLabel?: string
  yLabel?: string
  xRange?: [number, number]
  /** Either end may be undefined to let that end fit the data. */
  yRange?: [number | undefined, number | undefined]
  /** Logarithmic y-axis, e.g. for loss curves that fall by orders of magnitude. Values must be positive. */
  yLog?: boolean
  /** Thin muted segments drawn under the data, e.g. residuals. */
  segments?: Segment[]
  /** Ink arrows, e.g. a weight vector normal to a decision boundary. */
  vectors?: Segment[]
  /**
   * Equal pixel length per unit on both axes, so shapes and angles are true (an ellipse keeps its proportions, a
   * normal vector looks perpendicular). Requires `xRange` and `yRange`. The chart sets its own height from its width
   * and ignores `height`.
   */
  equalAspect?: boolean
  /** Hide axes, ticks and grid lines, e.g. for a grid of people where coordinates mean nothing. */
  bare?: boolean
  /** Draggable handles bound to parameters. See handles.ts. */
  handles?: Handle[]
  /** Clicks anywhere in the plot area, in data coordinates. See EChart. */
  onPlotClick?: (point: [number, number]) => void
  height?: number
  ariaLabel?: string
}

/**
 * Scatter and line series on shared numeric axes. Grouped scatter series split into one ECharts series per group,
 * coloured by categorical slot and shaped by `MARKER_SHAPES`, so identity never relies on colour alone.
 */
export function XYChart({
  series,
  xLabel,
  yLabel,
  xRange,
  yRange,
  yLog,
  segments,
  vectors,
  equalAspect,
  onPlotClick,
  handles,
  bare,
  height = 320,
  ariaLabel,
}: XYChartProps) {
  const { resolved: mode } = useTheme()
  const [x0, x1] = xRange ?? []
  const [y0, y1] = yRange ?? []
  const option = useMemo(() => {
    const c = chrome(mode)
    const out: Record<string, unknown>[] = []
    // Default slots count categories only: muted and emphasised series are not categories and take no palette slot.
    let category = 0
    series.forEach((s) => {
      const slot = s.slot ?? (s.muted || s.emphasis ? 0 : category++)
      if (s.type === 'scatter' && s.group) {
        const groups = [...new Set(s.group)].sort((a, b) => a - b)
        for (const g of groups) {
          const data = s.x.flatMap((x, i) => (s.group![i] === g ? [[x, s.y[i]]] : []))
          out.push(scatter(`${s.groupNames?.[g] ?? `${s.name} ${g}`}`, data, seriesColor(mode, g), g, s.emphasis))
        }
        return
      }
      const data = s.x.map((x, i) => [x, s.y[i]])
      if (s.type === 'bar') {
        out.push({
          name: s.name,
          type: 'bar',
          data,
          barWidth: '92%',
          barGap: '-100%',
          itemStyle: { color: s.muted ? chrome(mode).grid : seriesColor(mode, slot), borderRadius: [2, 2, 0, 0] },
          z: 1,
        })
      } else if (s.type === 'scatter') {
        // Emphasised marks (e.g. centroids) are ink-coloured so they never read as another category.
        const color = s.emphasis ? chrome(mode).ink : s.muted ? chrome(mode).grid : seriesColor(mode, slot)
        out.push(scatter(s.name, data, color, s.emphasis ? 3 : 0, s.emphasis, mode))
      } else {
        // Muted lines sit behind the data in the chrome colour; emphasised lines are ink. Neither uses a palette slot.
        const color = s.emphasis ? c.ink : s.muted ? c.muted : seriesColor(mode, slot)
        out.push({
          name: s.name,
          type: 'line',
          data,
          showSymbol: false,
          smooth: false,
          lineStyle: {
            width: s.muted ? 1 : LINE_WIDTH,
            color,
            opacity: s.muted ? 0.7 : 1,
            type: s.dashed ? 'dashed' : 'solid',
          },
          itemStyle: { color },
          ...(s.area ? { areaStyle: { color, opacity: 0.3 } } : {}),
          z: s.muted ? 2 : 3,
        })
      }
    })
    if (segments?.length) {
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
    if (vectors?.length) {
      // markLine orients arrowheads in screen space, so they point correctly whatever the axis scales.
      out.push({
        name: '__vectors',
        type: 'line',
        data: [],
        silent: true,
        markLine: {
          silent: true,
          symbol: ['none', 'arrow'],
          symbolSize: 10,
          label: { show: false },
          lineStyle: { color: c.ink, width: LINE_WIDTH, type: 'solid' },
          animation: false,
          data: vectors.map((v) => [{ coord: v.from }, { coord: v.to }]),
        },
        z: 6,
      })
    }
    const legend = out.filter((s) => !String(s.name).startsWith('__')).map((s) => s.name as string)
    return {
      // Equal horizontal and vertical margin totals, so the plot area has exactly the container's proportions.
      ...(equalAspect ? { grid: ASPECT_GRID } : {}),
      ...(bare ? { grid: { left: 8, right: 8, top: 32, bottom: 8 } } : {}),
      legend: { data: legend, show: legend.length > 1 },
      tooltip: {
        trigger: 'item',
        formatter: (p: { seriesName: string; value: number[] }) =>
          `${p.seriesName}<br/>(${formatNumber(p.value[0])}, ${formatNumber(p.value[1])})`,
      },
      xAxis: { type: 'value', name: xLabel, min: x0, max: x1, scale: true, show: !bare },
      yAxis: yLog
        ? {
            type: 'log',
            name: yLabel,
            min: y0,
            max: y1,
            nameGap: 44,
            axisLabel: { formatter: (v: number) => formatPower(v) },
          }
        : { type: 'value', name: yLabel, min: y0, max: y1, scale: true, nameGap: 36, show: !bare },
      series: out,
    }
  }, [series, segments, vectors, xLabel, yLabel, x0, x1, y0, y1, yLog, equalAspect, bare, mode])

  // For equal aspect, derive the height from the measured width: plot height / plot width = y span / x span.
  const wrapper = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    if (!equalAspect || !wrapper.current) return
    const el = wrapper.current
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [equalAspect])

  if (!equalAspect)
    return <EChart option={option} height={height} ariaLabel={ariaLabel} onPlotClick={onPlotClick} handles={handles} />
  if (x0 === undefined || x1 === undefined || y0 === undefined || y1 === undefined) {
    throw new Error('XYChart equalAspect needs xRange and yRange')
  }
  const plotWidth = Math.max(width - ASPECT_MARGIN, 0)
  const aspectHeight = Math.round((plotWidth * (y1 - y0)) / (x1 - x0)) + ASPECT_MARGIN
  return (
    <div ref={wrapper} className="w-full">
      {width > 0 && (
        <EChart
          option={option}
          height={aspectHeight}
          ariaLabel={ariaLabel}
          onPlotClick={onPlotClick}
          handles={handles}
        />
      )}
    </div>
  )
}

const ASPECT_GRID = { left: 52, right: 28, top: 36, bottom: 44 }
/** Horizontal and vertical margin totals of ASPECT_GRID; both are 80. */
const ASPECT_MARGIN = 80

/** Log-axis tick label: 10⁻³ rather than 0.001. */
function formatPower(v: number): string {
  const e = Math.round(Math.log10(v))
  if (Math.abs(v - 10 ** e) > 1e-9 * 10 ** e) return formatNumber(v)
  const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹'
  const digits = String(Math.abs(e))
    .split('')
    .map((d) => sup[Number(d)])
  return e === 0 ? '1' : `10${e < 0 ? '⁻' : ''}${digits.join('')}`
}

function scatter(
  name: string,
  data: number[][],
  color: string,
  shape: number,
  emphasis?: boolean,
  mode: 'light' | 'dark' = 'light',
) {
  return {
    name,
    type: 'scatter',
    data,
    symbol: MARKER_SHAPES[shape % MARKER_SHAPES.length],
    symbolSize: emphasis ? 16 : MARKER_SIZE,
    itemStyle: {
      color,
      opacity: emphasis ? 1 : 0.85,
      // A surface-coloured ring separates overlapping marks.
      borderColor: chrome(mode).surface,
      borderWidth: emphasis ? 2 : 1,
    },
    z: emphasis ? 5 : 2,
  }
}
