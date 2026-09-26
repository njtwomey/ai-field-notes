import { useEffect, useRef } from 'react'
import { useTheme } from '@/components/theme-provider'
import { cn } from '@/lib/utils'
import { echarts, type EChartsOption } from './echarts'
import { baseOption } from './theme'

export type EChartProps = {
  /** Chart-specific option, merged over the site theme from `baseOption`. */
  option: EChartsOption
  /** Pixels, or 'auto' to size from CSS (e.g. an aspect-ratio class). */
  height?: number | 'auto'
  className?: string
  /**
   * Small, frequently changing update merged onto the chart without re-rendering the rest, e.g. a moving marker.
   * Series in a patch are matched by `id`, so the base option must declare them with the same `id`.
   */
  patch?: EChartsOption
  /** Receives ECharts click events (`seriesId`, `seriesName`, `value`, ...). */
  onClick?: (event: EChartClick) => void
  /**
   * Receives clicks anywhere inside the plot area, in data coordinates. Use it when marks are too thin to hit, e.g.
   * to pick a point on a line by its x value.
   */
  onPlotClick?: (point: [number, number]) => void
  /** Accessible description of what the chart shows. */
  ariaLabel?: string
  /** False for charts without x/y axes (graphs), so the theme's default axes and grid are left out. */
  cartesian?: boolean
}

export type EChartClick = {
  seriesId?: string
  seriesName?: string
  value?: unknown
  dataIndex: number
  /** 'node' or 'edge' for graph series. */
  dataType?: string
  data?: unknown
}

type Plain = Record<string, unknown>
const isPlain = (v: unknown): v is Plain => typeof v === 'object' && v !== null && !Array.isArray(v)

/** Deep-merge `over` onto `base`. An array over a plain object merges the object into every element (axis lists). */
function merge(base: unknown, over: unknown): unknown {
  if (isPlain(base) && Array.isArray(over)) return over.map((item) => merge(base, item))
  if (!isPlain(base) || !isPlain(over)) return over === undefined ? base : over
  const out: Plain = { ...base }
  for (const [k, v] of Object.entries(over)) out[k] = merge(base[k], v)
  return out
}

/**
 * The only component that touches ECharts directly. All chart components build an option and render this.
 */
export function EChart({
  option,
  patch,
  onClick,
  onPlotClick,
  height = 320,
  className,
  ariaLabel,
  cartesian = true,
}: EChartProps) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const onClickRef = useRef(onClick)
  const onPlotClickRef = useRef(onPlotClick)
  const { resolved } = useTheme()

  useEffect(() => {
    onClickRef.current = onClick
    onPlotClickRef.current = onPlotClick
  })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const instance = echarts.init(el, undefined, { renderer: 'svg' })
    chart.current = instance
    instance.on('click', (e) => onClickRef.current?.(e as unknown as EChartClick))
    instance.getZr().on('click', (e) => {
      const pixel = [e.offsetX, e.offsetY]
      if (!onPlotClickRef.current || !instance.containPixel({ gridIndex: 0 }, pixel)) return
      const [x, y] = instance.convertFromPixel({ gridIndex: 0 }, pixel) as number[]
      onPlotClickRef.current([x, y])
    })
    let size = ''
    const observer = new ResizeObserver(([entry]) => {
      const next = `${Math.round(entry.contentRect.width)}x${Math.round(entry.contentRect.height)}`
      if (next === size) return
      size = next
      instance.resize({ animation: { duration: 0 } })
    })
    observer.observe(el)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  // Full replace when the base option changes. Keep `option` referentially stable (useMemo) or every render redraws.
  useEffect(() => {
    const base: Record<string, unknown> = { ...baseOption(resolved) }
    if (!cartesian) {
      delete base.xAxis
      delete base.yAxis
      delete base.grid
    }
    chart.current?.setOption(merge(base, option) as EChartsOption, { notMerge: true, lazyUpdate: true })
  }, [option, resolved, cartesian])

  // Merge-only update after the base option; runs alone when only the patch changes.
  useEffect(() => {
    if (patch) chart.current?.setOption(patch, { lazyUpdate: true })
  }, [patch, option, resolved])

  return (
    <div
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      className={cn('w-full', onPlotClick && 'cursor-pointer', className)}
      style={height === 'auto' ? undefined : { height }}
    />
  )
}
