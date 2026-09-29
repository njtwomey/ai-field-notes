import { useMemo } from 'react'
import { useTheme } from '@/components/theme-provider'
import { EChart, type PlotPointer } from './EChart'
import type { Handle } from './handles'
import { chrome, diverging, interpolateColors, seriesColor, sequential } from './palette'
import { LINE_WIDTH } from './theme'

/** A path over the image in pixel coordinates (x = column, y = row), e.g. a seam or the outline of a region. */
export type ImagePlotLine = {
  name: string
  x: number[]
  y: number[]
  /** Fixed categorical slot for the line colour. */
  slot?: number
  /** Ink colour instead of a palette slot, e.g. a hovered path. */
  emphasis?: boolean
  dashed?: boolean
}

export type ImagePlotProps = {
  /** Raster size in pixels. */
  width: number
  height: number
  /** Colour pixels, row-major RGB in [0, 1]: `rgb[3 * (r * width + c) + k]` for channel k. */
  rgb?: ArrayLike<number>
  /** Or one value per pixel, row-major, coloured on the site's `sequential` or `diverging` scale over `range`. */
  values?: ArrayLike<number>
  scale?: 'sequential' | 'diverging'
  range?: [number, number]
  /**
   * Columns spanned by the x axis, at least `width`. A wider axis leaves space to the right of the image, so images of
   * different widths drawn at the same `xExtent` share one pixel size.
   */
  xExtent?: number
  /** Paths over the image. Their points are applied as a patch; their names and styles are part of the chart. */
  lines?: ImagePlotLine[]
  /** Draggable handles bound to parameters, in pixel coordinates. See handles.ts. */
  handles?: Handle[]
  /** Hover and click positions in pixel coordinates (column, row). */
  onPointer?: (event: PlotPointer) => void
  ariaLabel?: string
}

// Each pixel is drawn as a UPSCALE × UPSCALE block, so the browser's image smoothing only softens block edges.
const UPSCALE = 4
const NO_LINES: ImagePlotLine[] = []

/** 256 RGB triples along evenly spaced colour stops, as ECharts' continuous visualMap would colour them. */
function lookupTable(stops: string[]): Uint8ClampedArray {
  const table = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = (interpolateColors(stops, i / 255).match(/\d+/g) ?? ['0', '0', '0']).map(Number)
    table.set([r, g, b], 3 * i)
  }
  return table
}

/** The raster drawn into a canvas, upscaled by nearest neighbour. */
function rasterCanvas(
  width: number,
  height: number,
  pixel: (i: number, out: Uint8ClampedArray, o: number) => void,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = width * UPSCALE
  canvas.height = height * UPSCALE
  const ctx = canvas.getContext('2d')
  if (!ctx) return canvas
  const data = ctx.createImageData(canvas.width, canvas.height)
  const px = data.data
  const rgba = new Uint8ClampedArray(4)
  rgba[3] = 255
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++) {
      pixel(r * width + c, rgba, 0)
      for (let dr = 0; dr < UPSCALE; dr++) {
        let o = 4 * ((r * UPSCALE + dr) * canvas.width + c * UPSCALE)
        for (let dc = 0; dc < UPSCALE; dc++, o += 4) px.set(rgba, o)
      }
    }
  ctx.putImageData(data, 0, 0)
  return canvas
}

/**
 * A raster image on pixel axes (rows downward), in colour or on a colour scale, with paths and handles on top. Use it
 * for images too large to draw as one mark per cell, which is what `Heatmap` does. The raster is one canvas image, so
 * a new image redraws in a millisecond or two. Axes are hidden: the pixels are the content.
 */
export function ImagePlot({
  width,
  height,
  rgb,
  values,
  scale = 'sequential',
  range,
  xExtent,
  lines = NO_LINES,
  handles,
  onPointer,
  ariaLabel,
}: ImagePlotProps) {
  const { resolved: mode } = useTheme()
  const extent = Math.max(xExtent ?? width, width)
  const [rangeLo, rangeHi] = range ?? [undefined, undefined]

  const canvas = useMemo(() => {
    if (typeof document === 'undefined') return null
    if (rgb)
      return rasterCanvas(width, height, (i, out) => {
        out[0] = 255 * rgb[3 * i]
        out[1] = 255 * rgb[3 * i + 1]
        out[2] = 255 * rgb[3 * i + 2]
      })
    if (!values) return null
    let lo = rangeLo ?? Infinity
    let hi = rangeHi ?? -Infinity
    if (rangeLo === undefined || rangeHi === undefined)
      for (let i = 0; i < width * height; i++) {
        if (rangeLo === undefined) lo = Math.min(lo, values[i])
        if (rangeHi === undefined) hi = Math.max(hi, values[i])
      }
    const table = lookupTable(scale === 'diverging' ? diverging(mode) : sequential)
    const span = hi > lo ? hi - lo : 1
    return rasterCanvas(width, height, (i, out) => {
      const k = 3 * Math.round(255 * Math.min(Math.max((values[i] - lo) / span, 0), 1))
      out[0] = table[k]
      out[1] = table[k + 1]
      out[2] = table[k + 2]
    })
  }, [width, height, rgb, values, scale, rangeLo, rangeHi, mode])

  // Structure only; the points go in the patch, so moving a path does not redraw the image.
  const lineKey = JSON.stringify(lines.map((l) => [l.name, l.slot ?? 0, !!l.emphasis, !!l.dashed]))

  const option = useMemo(() => {
    const structure = JSON.parse(lineKey) as [string, number, boolean, boolean][]
    const ink = chrome(mode).ink
    return {
      animation: false,
      grid: { left: 0, right: 0, top: 0, bottom: 0 },
      tooltip: { show: false },
      legend: { show: false },
      xAxis: { type: 'value', min: -0.5, max: extent - 0.5, show: false },
      yAxis: { type: 'value', min: -0.5, max: height - 0.5, inverse: true, show: false },
      series: [
        {
          id: '__raster',
          type: 'custom',
          silent: true,
          data: [[0]],
          renderItem: (_: unknown, api: { coord: (p: number[]) => number[] }) => {
            const [x0, y0] = api.coord([-0.5, -0.5])
            const [x1, y1] = api.coord([width - 0.5, height - 0.5])
            return canvas
              ? { type: 'image', style: { image: canvas, x: x0, y: y0, width: x1 - x0, height: y1 - y0 } }
              : { type: 'group', children: [] }
          },
          z: 1,
        },
        ...structure.map(([name, slot, emphasis, dashed], i) => {
          const color = emphasis ? ink : seriesColor(mode, slot)
          return {
            id: `line-${i}`,
            name,
            type: 'line',
            data: [],
            silent: true,
            showSymbol: false,
            lineStyle: { color, width: LINE_WIDTH, type: dashed ? 'dashed' : 'solid' },
            z: emphasis ? 4 : 3,
          }
        }),
      ],
    }
  }, [canvas, width, height, extent, lineKey, mode])

  const patch = useMemo(
    () => ({ series: lines.map((l, i) => ({ id: `line-${i}`, data: l.x.map((x, j) => [x, l.y[j]]) })) }),
    [lines],
  )

  return (
    <div className="w-full" style={{ aspectRatio: `${extent} / ${height}` }}>
      <EChart
        option={option}
        patch={patch}
        handles={handles}
        onPointer={onPointer}
        height="auto"
        className="h-full"
        ariaLabel={ariaLabel}
        renderer="canvas"
      />
    </div>
  )
}
