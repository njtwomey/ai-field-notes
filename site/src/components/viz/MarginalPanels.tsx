import { useMemo } from 'react'
import { useTheme } from '@/components/theme-provider'
import { EChart, type PlotPointer } from './EChart'
import type { Handle, Vec2 } from './handles'
import { chrome, mute, seriesColor, type Mode } from './palette'
import { LINE_WIDTH } from './theme'

/**
 * The three panels. `main` plots v against u = x. `bottom` shares its x-axis and plots a value (usually a density)
 * against x. `right` shares main's y-axis and plots a value against y, drawn rotated: y runs up, the value runs right.
 */
export type Panel = 'main' | 'bottom' | 'right'

/**
 * A mark in panel coordinates (u, v): (x, y) in main, (x, value) in bottom, (y, value) in right. The right panel swaps
 * them when drawing, so a density over y is written the same way as a density over x.
 */
export type PanelMark =
  /** A polyline, e.g. a density curve. Named lines appear in the legend. */
  | { kind: 'line'; points: Vec2[]; name?: string; slot?: number; emphasis?: boolean; dashed?: boolean; width?: number }
  /** A histogram: bar i spans [edges[i], edges[i + 1]] with height heights[i]. Drawn in a muted tint of its slot. */
  | { kind: 'bars'; edges: number[]; heights: number[]; name?: string; slot?: number }
  /** A filled polygon, e.g. the mass under a curve. Ink unless a slot is given. */
  | { kind: 'fill'; points: Vec2[]; slot?: number; opacity?: number }
  /** An axis-aligned band, u in [u0, u1] and v in [v0, v1]. */
  | { kind: 'rect'; u: Vec2; v: Vec2; slot?: number; opacity?: number }
  /** Separate straight segments, e.g. guide lines. Ink unless a slot is given. */
  | { kind: 'segments'; segments: [Vec2, Vec2][]; slot?: number; opacity?: number; width?: number; dashed?: boolean }
  /** Small dots, e.g. single samples. */
  | { kind: 'dots'; points: Vec2[]; slot?: number; emphasis?: boolean; size?: number }

export type PanelMarks = Partial<Record<Panel, PanelMark[]>>

export type PanelPointer =
  { type: 'move' | 'click'; panel: Panel; point: Vec2 } | { type: 'leave'; panel?: never; point?: never }

export type MarginalPanelsProps = {
  /** Data marks. Keep this object referentially stable (useMemo): a new one redraws the whole chart. */
  marks: PanelMarks
  /**
   * Fast-changing marks drawn above the data, e.g. a hover highlight. Applied as a patch, so updating them leaves the
   * data marks untouched.
   */
  overlay?: PanelMarks
  xRange: Vec2
  yRange: Vec2
  /** Top of the bottom and right panels' value axes; both start at 0. */
  bottomMax: number
  rightMax: number
  xLabel?: string
  yLabel?: string
  bottomLabel?: string
  rightLabel?: string
  /** Draggable handles in the main panel. See handles.ts. */
  handles?: Handle[]
  /** Hover, click and leave, in the panel coordinates of the panel under the pointer. */
  onPointer?: (event: PanelPointer) => void
  height?: number
  ariaLabel?: string
}

const PANELS: Panel[] = ['main', 'bottom', 'right']
const LEFT = 52
const RIGHT = 12
const TOP = 36
const BOTTOM = 44
const GAP = 6
/** Share of the width left of the right panel, and of the height given to the bottom panel. */
const MAIN_WIDTH = 0.72
const BOTTOM_SHARE = 0.24

/**
 * A function panel with a marginal panel below it (sharing x) and a rotated marginal to its right (sharing y), drawn in
 * one ECharts instance with three grids, so the shared axes align exactly at every width. The bottom-right corner is
 * left empty. The layout for a change of variables: x below, y = g(x) in the middle, the density of y on the right.
 */
export function MarginalPanels({
  marks,
  overlay,
  xRange,
  yRange,
  bottomMax,
  rightMax,
  xLabel,
  yLabel,
  bottomLabel,
  rightLabel,
  handles,
  onPointer,
  height = 520,
  ariaLabel,
}: MarginalPanelsProps) {
  const { resolved: mode } = useTheme()
  const [x0, x1] = xRange
  const [y0, y1] = yRange

  const option = useMemo(() => {
    const plot = height - TOP - BOTTOM - GAP
    const bottomHeight = Math.round(plot * BOTTOM_SHARE)
    const mainBottom = BOTTOM + bottomHeight + GAP
    const split = `${Math.round((1 - MAIN_WIDTH) * 100)}%`
    const grid = [
      { left: LEFT, right: split, top: TOP, bottom: mainBottom },
      { left: LEFT, right: split, top: height - BOTTOM - bottomHeight, bottom: BOTTOM },
      { left: `${Math.round(MAIN_WIDTH * 100) + 1}%`, right: RIGHT, top: TOP, bottom: mainBottom },
    ]
    const hidden = { axisLabel: { show: false } }
    const value = { splitNumber: 2, scale: false }
    const xAxis = [
      { type: 'value', gridIndex: 0, min: x0, max: x1, ...hidden },
      { type: 'value', gridIndex: 1, min: x0, max: x1, name: xLabel },
      { type: 'value', gridIndex: 2, min: 0, max: rightMax, name: rightLabel, ...value },
    ]
    const yAxis = [
      // Labels at the ends that meet across the gap would collide: main drops its lowest, bottom its highest.
      { type: 'value', gridIndex: 0, min: y0, max: y1, name: yLabel, nameGap: 36, axisLabel: { showMinLabel: false } },
      {
        type: 'value',
        gridIndex: 1,
        min: 0,
        max: bottomMax,
        name: bottomLabel,
        nameGap: 36,
        axisLabel: { showMaxLabel: false },
        ...value,
      },
      { type: 'value', gridIndex: 2, min: y0, max: y1, ...hidden },
    ]
    const series = PANELS.flatMap((panel) => (marks[panel] ?? []).map((m, i) => markSeries(m, panel, i, mode)))
    // One overlay series per panel, declared empty here and filled by the patch.
    for (const panel of PANELS) series.push(overlaySeries(panel, [], mode))
    const legend = series.flatMap((s) => (s.name && !String(s.name).startsWith('__') ? [s.name as string] : []))
    return {
      grid,
      xAxis,
      yAxis,
      series,
      legend: { data: legend, show: legend.length > 0 },
      tooltip: { show: false },
    }
  }, [marks, x0, x1, y0, y1, bottomMax, rightMax, xLabel, yLabel, bottomLabel, rightLabel, height, mode])

  const patch = useMemo(
    () => ({ series: PANELS.map((panel) => overlaySeries(panel, overlay?.[panel] ?? [], mode)) }),
    [overlay, mode],
  )

  const handler = useMemo(
    () =>
      onPointer &&
      ((e: PlotPointer) => {
        if (e.type === 'leave') return onPointer(e)
        const panel = PANELS[e.grid]
        if (!panel) return
        const [a, b] = e.point
        onPointer({ type: e.type, panel, point: panel === 'right' ? [b, a] : [a, b] })
      }),
    [onPointer],
  )

  return (
    <EChart option={option} patch={patch} height={height} handles={handles} onPointer={handler} ariaLabel={ariaLabel} />
  )
}

const axes = (panel: Panel) => {
  const i = PANELS.indexOf(panel)
  return { xAxisIndex: i, yAxisIndex: i }
}

/** Panel coordinates to the grid's data coordinates: the right panel is rotated. */
const toData = (panel: Panel, [u, v]: Vec2): Vec2 => (panel === 'right' ? [v, u] : [u, v])

function markColor(m: { slot?: number; emphasis?: boolean }, mode: Mode) {
  return m.slot === undefined || m.emphasis ? chrome(mode).ink : seriesColor(mode, m.slot)
}

/** One data mark as an ECharts series: lines as line series (for the legend), everything else as a custom series. */
function markSeries(m: PanelMark, panel: Panel, i: number, mode: Mode): Record<string, unknown> {
  if (m.kind === 'line') {
    const color = markColor(m, mode)
    return {
      id: `${panel}-${i}`,
      name: m.name ?? `__${panel}-${i}`,
      type: 'line',
      ...axes(panel),
      data: m.points.map((p) => toData(panel, p)),
      showSymbol: false,
      silent: true,
      animation: false,
      lineStyle: { color, width: m.width ?? LINE_WIDTH, type: m.dashed ? 'dashed' : 'solid' },
      itemStyle: { color },
      z: 3,
    }
  }
  const series = customSeries(`${panel}-${i}`, panel, [m], mode, m.kind === 'bars' ? 1 : 2)
  // The legend takes a custom series' colour from its itemStyle.
  if (m.kind === 'bars' && m.name) return { ...series, name: m.name, itemStyle: { color: barColor(m, mode) } }
  return series
}

function overlaySeries(panel: Panel, marks: PanelMark[], mode: Mode) {
  return customSeries(`__overlay-${panel}`, panel, marks, mode, 4)
}

const barColor = (m: { slot?: number }, mode: Mode) =>
  mute(m.slot === undefined ? chrome(mode).muted : seriesColor(mode, m.slot), chrome(mode).surface, 0.4)

type RenderApi = { coord: (point: number[]) => number[] }

/** A custom series that draws `marks` as one group of shapes, clipped to its grid. */
function customSeries(id: string, panel: Panel, marks: PanelMark[], mode: Mode, z: number) {
  const ink = chrome(mode).ink
  return {
    id,
    name: `__${id}`,
    type: 'custom',
    ...axes(panel),
    silent: true,
    clip: true,
    animation: false,
    z,
    // One placeholder datum, so renderItem runs once and draws every mark. The axes have fixed ranges, so its value
    // does not move them.
    data: marks.length ? [[0, 0]] : [],
    renderItem: (_params: unknown, api: RenderApi) => {
      const px = (p: Vec2) => api.coord(toData(panel, p))
      const children = marks.flatMap((m): Record<string, unknown>[] => {
        const color = markColor(m as { slot?: number }, mode)
        switch (m.kind) {
          case 'bars':
            return m.heights.map((h, j) => {
              const a = px([m.edges[j], 0])
              const b = px([m.edges[j + 1], h])
              return {
                type: 'rect',
                shape: box(a, b),
                style: { fill: barColor(m, mode) },
              }
            })
          case 'fill':
            return [
              {
                type: 'polygon',
                shape: { points: m.points.map(px) },
                style: { fill: color, opacity: m.opacity ?? 0.3 },
              },
            ]
          case 'rect':
            return [
              {
                type: 'rect',
                shape: box(px([m.u[0], m.v[0]]), px([m.u[1], m.v[1]])),
                style: { fill: color, opacity: m.opacity ?? 0.08 },
              },
            ]
          case 'line':
            return [
              {
                type: 'polyline',
                shape: { points: m.points.map(px) },
                style: {
                  stroke: color,
                  fill: 'none',
                  lineWidth: m.width ?? LINE_WIDTH,
                  lineDash: m.dashed ? [4, 3] : undefined,
                },
              },
            ]
          case 'segments':
            return m.segments.map(([a, b]) => ({
              type: 'line',
              shape: line(px(a), px(b)),
              style: {
                stroke: color,
                lineWidth: m.width ?? 1,
                opacity: m.opacity ?? 1,
                lineDash: m.dashed ? [4, 3] : undefined,
              },
            }))
          case 'dots':
            return m.points.map((p) => {
              const [cx, cy] = px(p)
              return {
                type: 'circle',
                shape: { cx, cy, r: m.size ?? 3 },
                style: { fill: m.emphasis ? ink : color, stroke: chrome(mode).surface, lineWidth: 1 },
              }
            })
        }
      })
      return { type: 'group', children }
    },
  }
}

const line = (a: number[], b: number[]) => ({ x1: a[0], y1: a[1], x2: b[0], y2: b[1] })

/** The rectangle spanned by two pixel corners, in any order. */
function box(a: number[], b: number[]) {
  return {
    x: Math.min(a[0], b[0]),
    y: Math.min(a[1], b[1]),
    width: Math.abs(b[0] - a[0]),
    height: Math.abs(b[1] - a[1]),
  }
}
