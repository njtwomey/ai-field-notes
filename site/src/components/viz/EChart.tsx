import { useEffect, useRef } from 'react'
import { useTheme } from '@/components/theme-provider'
import { cn } from '@/lib/utils'
import { echarts, type EChartsOption } from './echarts'
import { GRAB_RADIUS, type Handle } from './handles'
import { chrome, type Mode } from './palette'
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
  /**
   * Draggable handles bound to parameters (see handles.ts). With one handle, pressing anywhere on the plot moves it;
   * with several, the nearest within GRAB_RADIUS pixels is grabbed. While a handle is dragged the axes are frozen,
   * so a range that depends on the dragged value cannot rescale under the pointer.
   */
  handles?: Handle[]
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

const HANDLES_ID = '__handles'

type Extents = { x: [number, number]; y: [number, number] }

/** Everything the chart is drawn from. Kept in a ref so pointer handlers and effects share one source. */
type Inputs = {
  option: EChartsOption
  patch?: EChartsOption
  handles?: Handle[]
  mode: Mode
  cartesian: boolean
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
  handles,
}: EChartProps) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const onClickRef = useRef(onClick)
  const onPlotClickRef = useRef(onPlotClick)
  const { resolved } = useTheme()
  const inputs = useRef<Inputs>({ option, patch, handles, mode: resolved, cartesian })
  const frozen = useRef<Extents | null>(null)

  useEffect(() => {
    onClickRef.current = onClick
    onPlotClickRef.current = onPlotClick
    inputs.current = { option, patch, handles, mode: resolved, cartesian }
  })

  /**
   * Draw from the latest inputs. `full` replaces the base option (with frozen axes during a drag); otherwise only the
   * patch and the handles are merged in, which leaves the rest of the chart untouched.
   */
  const render = useRef((full: boolean) => {
    const instance = chart.current
    if (!instance) return
    const { option, patch, handles, mode, cartesian } = inputs.current
    if (full) {
      const base: Plain = { ...baseOption(mode) }
      if (!cartesian) {
        delete base.xAxis
        delete base.yAxis
        delete base.grid
      }
      const merged = merge(base, option) as Plain
      if (frozen.current && cartesian) {
        const { x, y } = frozen.current
        merged.xAxis = merge(merged.xAxis, { min: x[0], max: x[1] })
        merged.yAxis = merge(merged.yAxis, { min: y[0], max: y[1] })
      }
      if (handles && cartesian) merged.series = [...asArray(merged.series), handlesSeries(mode)]
      instance.setOption(merged as EChartsOption, { notMerge: true })
    }
    if (patch) instance.setOption(patch)
    if (handles && cartesian) instance.setOption({ series: [handlesPatch(handles)] })
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
    const detach = attachHandles(instance, () => inputs.current.handles, {
      start: () => {
        frozen.current = axisExtents(instance)
      },
      end: () => {
        if (!frozen.current) return
        frozen.current = null
        render.current(true)
      },
    })
    render.current(true)
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
      detach()
      instance.dispose()
      chart.current = null
    }
  }, [])

  // Keep `option` referentially stable (useMemo) or every render redraws the whole chart.
  useEffect(() => render.current(true), [option, resolved, cartesian])
  useEffect(() => render.current(false), [patch, handles])

  return (
    <div
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      className={cn('w-full', onPlotClick && 'cursor-pointer', handles && 'touch-none select-none', className)}
      style={height === 'auto' ? undefined : { height }}
    />
  )
}

const asArray = (v: unknown): unknown[] => (Array.isArray(v) ? v : v ? [v] : [])

/** The series that draws every handle: points as ink markers, x and y handles as dashed guide lines. */
function handlesSeries(mode: Mode) {
  const c = chrome(mode)
  return {
    id: HANDLES_ID,
    type: 'scatter',
    data: [],
    silent: true,
    symbol: 'circle',
    symbolSize: 14,
    itemStyle: { color: c.ink, borderColor: c.surface, borderWidth: 2 },
    z: 20,
    markLine: {
      silent: true,
      symbol: ['none', 'none'],
      animation: false,
      lineStyle: { color: c.ink, width: 1.5, type: 'dashed' },
      label: { color: c.ink, fontSize: 11, position: 'end' },
      data: [],
    },
  }
}

function handlesPatch(handles: Handle[]) {
  return {
    id: HANDLES_ID,
    data: handles.flatMap((h) => (h.kind === 'point' ? [h.at] : [])),
    markLine: {
      data: handles.flatMap((h) =>
        h.kind === 'point' ? [] : [{ [h.kind === 'x' ? 'xAxis' : 'yAxis']: h.at, label: { formatter: h.label ?? '' } }],
      ),
    },
  }
}

type AxisModel = { axis?: { scale: { getExtent: () => [number, number] } } }

/** Current extents of the first grid's axes, as the axis scales computed them (so nice ticks survive a freeze). */
function axisExtents(instance: echarts.ECharts): Extents | null {
  const model = (
    instance as unknown as { getModel: () => { getComponent: (type: string, i: number) => unknown } }
  ).getModel()
  const x = (model.getComponent('xAxis', 0) as AxisModel | undefined)?.axis?.scale.getExtent()
  const y = (model.getComponent('yAxis', 0) as AxisModel | undefined)?.axis?.scale.getExtent()
  return x && y ? { x: [...x], y: [...y] } : null
}

/**
 * Pointer logic for handles. Grab on press (the nearest handle, or the only one), follow the pointer while pressed,
 * release on pointer-up anywhere. Movement is batched to one update per frame; the final position is always applied.
 *
 * The active handle is tracked by its index, and its setter is read from the latest handles on every move: widgets
 * rebuild the handles array each render, so an old closure could hold stale state.
 */
function attachHandles(
  instance: echarts.ECharts,
  current: () => Handle[] | undefined,
  hooks: { start: () => void; end: () => void },
): () => void {
  const zr = instance.getZr()
  let active: number | null = null
  let frame = 0
  let pending: number[] | null = null

  const toData = (pixel: number[]) => instance.convertFromPixel({ gridIndex: 0 }, pixel) as number[]
  const toPixel = (point: number[]) => instance.convertToPixel({ gridIndex: 0 }, point) as number[]
  const inside = (pixel: number[]) => instance.containPixel({ gridIndex: 0 }, pixel)

  const distance = (h: Handle, pixel: number[]) => {
    const here = toData(pixel)
    if (h.kind === 'point') {
      const [x, y] = toPixel(h.at)
      return Math.hypot(x - pixel[0], y - pixel[1])
    }
    if (h.kind === 'x') return Math.abs(toPixel([h.at, here[1]])[0] - pixel[0])
    return Math.abs(toPixel([here[0], h.at])[1] - pixel[1])
  }

  const nearest = (pixel: number[]): number | null => {
    const list = current()
    if (!list?.length || !inside(pixel)) return null
    if (list.length === 1) return 0
    let best: number | null = null
    let bestDistance = Infinity
    list.forEach((h, i) => {
      const d = distance(h, pixel)
      if (d < bestDistance) {
        best = i
        bestDistance = d
      }
    })
    return bestDistance <= GRAB_RADIUS ? best : null
  }

  const apply = (pixel: number[]) => {
    const h = active === null ? undefined : current()?.[active]
    if (!h) return
    const [x, y] = toData(pixel)
    if (h.kind === 'point') h.onDrag([x, y])
    else if (h.kind === 'x') h.onDrag(x)
    else h.onDrag(y)
  }

  const cursorFor = (i: number | null) => {
    const h = i === null ? undefined : current()?.[i]
    return !h ? 'default' : h.kind === 'x' ? 'col-resize' : h.kind === 'y' ? 'row-resize' : 'grab'
  }

  // Plain DOM pointer events with pointer capture: the drag keeps tracking outside the chart, and works for touch.
  const el = instance.getDom()
  const local = (e: PointerEvent) => {
    const rect = el.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  const down = (e: PointerEvent) => {
    if (e.button !== 0) return
    active = nearest(local(e))
    if (active === null) return
    e.preventDefault()
    try {
      el.setPointerCapture(e.pointerId)
    } catch {
      // Synthetic events have no active pointer to capture; the drag still works while the pointer stays inside.
    }
    hooks.start()
    zr.setCursorStyle('grabbing')
    apply(local(e))
  }
  const move = (e: PointerEvent) => {
    if (active === null) {
      if (current()?.length) zr.setCursorStyle(cursorFor(nearest(local(e))))
      return
    }
    pending = local(e)
    if (!frame)
      frame = requestAnimationFrame(() => {
        frame = 0
        if (pending) apply(pending)
        pending = null
      })
  }
  const release = (e: PointerEvent) => {
    if (active === null) return
    cancelAnimationFrame(frame)
    frame = 0
    pending = null
    apply(local(e))
    active = null
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
    hooks.end()
  }
  const listeners = { pointerdown: down, pointermove: move, pointerup: release, pointercancel: release }
  for (const [type, fn] of Object.entries(listeners)) el.addEventListener(type, fn as EventListener)
  return () => {
    cancelAnimationFrame(frame)
    for (const [type, fn] of Object.entries(listeners)) el.removeEventListener(type, fn as EventListener)
  }
}
