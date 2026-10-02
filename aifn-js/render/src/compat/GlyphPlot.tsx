import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTheme } from '@render/design/theme'
import { EChart } from '@render/viz/EChart'
import type { Handle, Vec2 } from '@render/viz/handles'
import { chrome, seriesColor } from '@render/design/palette'

/**
 * One filled shape: closed contours of [x, y] points. Contours are filled with the nonzero rule, so a contour wound
 * the opposite way to its enclosing contour is a hole, as in a font's glyph outlines.
 */
export type GlyphShape = {
  contours: Vec2[][]
  /** Fill: 'ink' (the text colour, the default), 'muted', or a categorical palette slot. */
  tone?: 'ink' | 'muted' | number
  /** Fill opacity, e.g. for a ghost of a reference shape drawn behind. */
  opacity?: number
}

export type GlyphPlotProps = {
  shapes: GlyphShape[]
  xRange: [number, number]
  yRange: [number, number]
  /** Thin horizontal guide lines, e.g. a baseline and a cap height. */
  guides?: number[]
  /** Draggable handles bound to parameters. See handles.ts. */
  handles?: Handle[]
  ariaLabel?: string
}

type CustomApi = { value: (dim: number) => number; coord: (point: number[]) => number[] }

const GRID = { left: 4, right: 4, top: 4, bottom: 4 }

/**
 * Filled vector shapes, such as generated glyphs, on hidden axes with equal pixel length per unit. The height follows
 * from the width and the ranges. Shapes are drawn as SVG paths, so counters stay open and edges stay sharp.
 */
export function GlyphPlot({ shapes, xRange, yRange, guides, handles, ariaLabel }: GlyphPlotProps) {
  const { resolved: mode } = useTheme()
  const [x0, x1] = xRange
  const [y0, y1] = yRange
  const guideKey = (guides ?? []).join(',')

  const option = useMemo(() => {
    const c = chrome(mode)
    const fill = (tone: GlyphShape['tone']) =>
      tone === undefined || tone === 'ink' ? c.ink : tone === 'muted' ? c.muted : seriesColor(mode, tone)
    const lines = guideKey ? guideKey.split(',').map(Number) : []
    return {
      grid: GRID,
      tooltip: { show: false },
      legend: { show: false },
      xAxis: { type: 'value', min: x0, max: x1, show: false },
      yAxis: { type: 'value', min: y0, max: y1, show: false },
      series: [
        {
          name: '__guides',
          type: 'custom',
          silent: true,
          data: lines.map((y) => [y]),
          renderItem: (_: unknown, api: CustomApi) => {
            const [ax, ay] = api.coord([x0, api.value(0)])
            const [bx] = api.coord([x1, api.value(0)])
            return {
              type: 'line',
              shape: { x1: ax, y1: ay, x2: bx, y2: ay },
              style: { stroke: c.grid, lineWidth: 1 },
            }
          },
          z: 1,
        },
        {
          name: '__shapes',
          type: 'custom',
          silent: true,
          progressive: 0,
          data: shapes.map((_, i) => [i]),
          renderItem: (_: unknown, api: CustomApi) => {
            const s = shapes[api.value(0)]
            const d = s.contours
              .map((contour) => {
                const pts = contour.map((p) => api.coord(p))
                return `M${pts.map(([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`).join('L')}Z`
              })
              .join('')
            return {
              type: 'path',
              shape: { pathData: d },
              style: { fill: fill(s.tone), opacity: s.opacity ?? 1, stroke: 'none' },
            }
          },
          z: 2,
        },
      ],
    }
  }, [shapes, x0, x1, y0, y1, guideKey, mode])

  const wrapper = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    if (!wrapper.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)))
    observer.observe(wrapper.current)
    return () => observer.disconnect()
  }, [])
  const plotWidth = Math.max(width - GRID.left - GRID.right, 0)
  const height = Math.round((plotWidth * (y1 - y0)) / (x1 - x0)) + GRID.top + GRID.bottom

  return (
    <div ref={wrapper} className="w-full">
      {width > 0 && <EChart option={option} height={height} ariaLabel={ariaLabel} handles={handles} />}
    </div>
  )
}
