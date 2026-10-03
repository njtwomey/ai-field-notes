import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTheme } from '@render/design/theme'
import { EChart } from '@render/viz/EChart'
import type { Handle } from '@render/viz/handles'
import { chrome, MARKER_SHAPES, seriesColor } from '@render/design/palette'
import { vectorLines, type Vector } from '@render/viz/vectors'
import { LINE_WIDTH, MARKER_SIZE } from '@render/viz/theme'
import { formatNumber } from '@render/viz/format'

export type Series = {
  name: string
  type: 'scatter' | 'line'
  x: number[]
  y: number[]
  group?: number[] | null
}

export type XYSeries = Omit<Series, 'group' | 'type'> & {
  type: Series['type'] | 'bar'
  group?: number[] | null
  slot?: number
  dashed?: boolean
  area?: boolean
  emphasis?: boolean
  muted?: boolean
  thin?: boolean
  groupNames?: string[]
  color?: string
  pointColors?: string[]
}

export type Segment = { from: [number, number]; to: [number, number] }

export type XYRect = {
  name?: string
  x0: number
  x1: number
  y0: number
  y1: number
  color?: string
  fill?: string
  stroke?: string
  strokeWidth?: number
  dashed?: boolean
  label?: string
  labelSub?: string
  labelColor?: string
}

export type XYChartProps = {
  series: XYSeries[]
  xLabel?: string
  yLabel?: string
  xRange?: [number, number]
  yRange?: [number | undefined, number | undefined]
  yLog?: boolean
  segments?: Segment[]
  vectors?: Vector[]
  rects?: XYRect[]
  equalAspect?: boolean
  integerX?: boolean
  bare?: boolean
  handles?: Handle[]
  onPlotClick?: (point: [number, number]) => void
  height?: number
  ariaLabel?: string
}

export function XYChart({
  series,
  xLabel,
  yLabel,
  xRange,
  yRange,
  yLog,
  segments,
  vectors,
  rects,
  equalAspect,
  onPlotClick,
  handles,
  integerX,
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
    let category = 0
    series.forEach((s) => {
      const slot = s.slot ?? (s.muted || s.emphasis ? 0 : category++)
      if (s.type === 'scatter' && s.group) {
        const groups = [...new Set(s.group)].sort((a, b) => a - b)
        for (const g of groups) {
          const data = s.x.flatMap((x, i) => (s.group![i] === g ? [[x, s.y[i]]] : []))
          out.push(scatter(`${s.groupNames?.[g] ?? `${s.name} ${g}`}`, data, seriesColor(mode, g), g, s.emphasis, mode))
        }
        return
      }
      const points = s.x.map((x, i) => [x, s.y[i]])
      const data = s.pointColors
        ? points.map((value, i) => ({ value, itemStyle: { color: s.pointColors![i] } }))
        : points
      const own = s.color ?? seriesColor(mode, slot)
      if (s.type === 'bar') {
        out.push({
          name: s.name,
          type: 'bar',
          data,
          barWidth: '92%',
          barGap: '-100%',
          itemStyle: { color: s.muted ? chrome(mode).grid : own, borderRadius: [2, 2, 0, 0] },
          z: 1,
        })
      } else if (s.type === 'scatter') {
        const color = s.emphasis ? chrome(mode).ink : s.muted ? chrome(mode).grid : own
        const seriesItem = scatter(s.name, points, color, s.emphasis ? 3 : 0, s.emphasis, mode)
        out.push(
          s.pointColors
            ? { ...seriesItem, data, itemStyle: { ...seriesItem.itemStyle, color: c.ink, borderColor: c.ink, borderWidth: 1 } }
            : seriesItem,
        )
      } else {
        const color = s.emphasis ? c.ink : s.muted ? c.muted : own
        out.push({
          name: s.name,
          type: 'line',
          data,
          showSymbol: false,
          smooth: false,
          lineStyle: {
            width: s.muted || s.thin ? 1 : LINE_WIDTH,
            color,
            opacity: s.thin ? 0.45 : s.muted ? 0.7 : 1,
            type: s.dashed ? 'dashed' : 'solid',
          },
          itemStyle: { color },
          ...(s.area ? { areaStyle: { color, opacity: 0.3 } } : {}),
          z: s.muted || s.thin ? 2 : 3,
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
    if (rects?.length) {
      out.unshift({
        name: '__rects',
        type: 'custom',
        data: rects.map((r, i) => [r.x0, r.x1, r.y0, r.y1, i]),
        silent: true,
        clip: true,
        renderItem: (
          _params: unknown,
          api: {
            value: (dim: number) => number
            coord: (pt: [number, number]) => [number, number]
          },
        ) => {
          const idx = api.value(4)
          const r = rects[idx]
          if (!r) return
          const p0 = api.coord([r.x0, r.y0])
          const p1 = api.coord([r.x1, r.y1])
          const rx = Math.min(p0[0], p1[0])
          const ry = Math.min(p0[1], p1[1])
          const rw = Math.abs(p1[0] - p0[0])
          const rh = Math.abs(p1[1] - p0[1])

          const groupChildren: Record<string, unknown>[] = [
            {
              type: 'rect',
              shape: { x: rx, y: ry, width: rw, height: rh },
              style: {
                fill: r.fill ?? 'transparent',
                stroke: r.stroke ?? r.color ?? '#3b82f6',
                lineWidth: r.strokeWidth ?? 1.5,
                lineDash: r.dashed ? [4, 4] : undefined,
              },
            },
          ]

          if (r.label && rw > 28 && rh > 28) {
            groupChildren.push({
              type: 'text',
              style: {
                text: r.labelSub ? `${r.label}\n${r.labelSub}` : r.label,
                x: rx + rw / 2,
                y: ry + rh / 2,
                fill: r.labelColor ?? (mode === 'dark' ? '#f3f4f6' : '#1f2937'),
                font: 'bold 11px sans-serif',
                align: 'center',
                verticalAlign: 'middle',
              },
            })
          }

          return {
            type: 'group',
            children: groupChildren,
          }
        },
        z: 2,
      })
    }
    if (vectors?.length) {
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
          data: vectorLines(vectors, mode),
        },
        z: 6,
      })
    }
    const legend = [...new Set(out.filter((s) => !String(s.name).startsWith('__')).map((s) => s.name as string))]
    return {
      ...(equalAspect || bare ? { grid: bare ? BARE_GRID : ASPECT_GRID } : {}),
      legend: { data: legend, show: legend.length > 1 },
      tooltip: {
        trigger: 'item',
        formatter: (p: { seriesName: string; value: number[] }) =>
          `${p.seriesName}<br/>(${formatNumber(p.value[0])}, ${formatNumber(p.value[1])})`,
      },
      xAxis: {
        type: 'value',
        name: xLabel,
        min: x0,
        max: x1,
        scale: true,
        show: !bare,
        ...(integerX
          ? {
              interval: 0.5,
              axisLabel: { formatter: (v: number) => (Number.isInteger(v) ? String(v) : '') },
              splitLine: { show: false },
            }
          : {}),
      },
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
  }, [series, segments, vectors, rects, xLabel, yLabel, x0, x1, y0, y1, yLog, equalAspect, bare, integerX, mode])

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
  const grid = bare ? BARE_GRID : ASPECT_GRID
  const plotWidth = Math.max(width - grid.left - grid.right, 0)
  const aspectHeight = Math.round((plotWidth * (y1 - y0)) / (x1 - x0)) + grid.top + grid.bottom
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
const BARE_GRID = { left: 8, right: 8, top: 32, bottom: 8 }

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
      borderColor: chrome(mode).surface,
      borderWidth: emphasis ? 2 : 1,
    },
    z: emphasis ? 5 : 2,
  }
}
