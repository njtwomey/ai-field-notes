/**
 * Zoom and pan. A viewport is the pair of axis ranges a chart shows. Zoom scales a range about a point, pan shifts it
 * by a fraction of its span; on a log axis both happen in log space, so a zoomed log axis keeps whole decades
 * proportional.
 */
import { useCallback, useMemo, useState } from 'react'

export type Range = [number, number]

const toLog = (log: boolean) => (v: number) => (log ? Math.log10(v) : v)
const fromLog = (log: boolean) => (v: number) => (log ? 10 ** v : v)

/** A usable range: finite, increasing, and positive on a log axis. */
export function validRange(r: Range | undefined, log = false): r is Range {
  return !!r && Number.isFinite(r[0]) && Number.isFinite(r[1]) && r[0] < r[1] && (!log || r[0] > 0)
}

/** `r` scaled about `about` (default its centre) by `factor`: above 1 zooms out, below 1 zooms in. */
export function zoomRange(r: Range, factor: number, log = false, about?: number): Range {
  if (log && !(r[0] > 0)) return r
  const [f, g] = [toLog(log), fromLog(log)]
  const [a, b] = [f(r[0]), f(r[1])]
  const c = about !== undefined && Number.isFinite(f(about)) ? f(about) : (a + b) / 2
  return [g(c + (a - c) * factor), g(c + (b - c) * factor)]
}

/** `r` shifted by `fraction` of its span: positive moves the view right (x) or up (y). */
export function panRange(r: Range, fraction: number, log = false): Range {
  if (log && !(r[0] > 0)) return r
  const [f, g] = [toLog(log), fromLog(log)]
  const d = (f(r[1]) - f(r[0])) * fraction
  return [g(f(r[0]) + d), g(f(r[1]) + d)]
}

/** The finite extent of some value arrays (positive values only on a log axis), or undefined if there are none. */
export function extentOf(arrays: Iterable<ArrayLike<number>>, log = false): Range | undefined {
  let lo = Infinity
  let hi = -Infinity
  for (const values of arrays)
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (!Number.isFinite(v) || (log && v <= 0)) continue
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  if (!Number.isFinite(lo)) return undefined
  return lo === hi ? (log ? [lo / 10, hi * 10] : [lo - 1, hi + 1]) : [lo, hi]
}

export type Axis = 'x' | 'y'

export type Viewport = {
  /** The zoomed range of each axis, or undefined where the axis follows the chart's own fit. */
  view: { x?: Range; y?: Range }
  /** The range each axis shows now: the zoomed range, or else the fitted one. */
  shown: { x?: Range; y?: Range }
  zoomed: boolean
  set: (axis: Axis, r: Range) => void
  /** Zoom one axis, or both, by `factor` about an optional point. */
  zoom: (factor: number, axes?: Axis[], about?: { x?: number; y?: number }) => void
  pan: (axis: Axis, fraction: number) => void
  /** Back to the fitted view. */
  reset: () => void
}

/**
 * Zoom state for a chart. `fitted` is what each axis shows unzoomed (the chart's explicit range, or the extent the
 * chart drew); zooming starts from it. The zoom survives data changes (e.g. scrubbing a trace) until `reset`.
 */
/** An axis whose zoom lives outside the chart, e.g. shared by the panels of a `Subplots`. */
export type AxisStore = { view: Range | undefined; set: (r: Range | undefined) => void }

export function useViewport(
  fitted: { x?: Range; y?: Range },
  log: { x?: boolean; y?: boolean } = {},
  shared: { x?: AxisStore; y?: AxisStore } = {},
): Viewport {
  const [own, setOwn] = useState<{ x?: Range; y?: Range }>({})
  const logX = !!log.x
  const logY = !!log.y
  const sx = shared.x
  const sy = shared.y
  const vx = sx ? sx.view : own.x
  const vy = sy ? sy.view : own.y
  const view = useMemo(() => ({ x: vx, y: vy }), [vx, vy])
  const [fx0, fx1] = fitted.x ?? []
  const [fy0, fy1] = fitted.y ?? []
  const shown = useMemo(
    () => ({
      x: view.x ?? (fx0 !== undefined && fx1 !== undefined ? ([fx0, fx1] as Range) : undefined),
      y: view.y ?? (fy0 !== undefined && fy1 !== undefined ? ([fy0, fy1] as Range) : undefined),
    }),
    [view, fx0, fx1, fy0, fy1],
  )
  const isLog = useCallback((axis: Axis) => (axis === 'x' ? logX : logY), [logX, logY])
  /** Write one axis's zoom: to the shared store if it has one, else to this chart's own state. */
  const write = useCallback(
    (axis: Axis, r: Range | undefined) => {
      const store = axis === 'x' ? sx : sy
      if (store) store.set(r)
      else setOwn((v) => ({ ...v, [axis]: r }))
    },
    [sx, sy],
  )
  const set = useCallback(
    (axis: Axis, r: Range) => {
      if (validRange(r, isLog(axis))) write(axis, r)
    },
    [isLog, write],
  )
  const zoom = useCallback(
    (factor: number, axes: Axis[] = ['x', 'y'], about: { x?: number; y?: number } = {}) => {
      for (const axis of axes) {
        const r = shown[axis]
        if (validRange(r, isLog(axis))) write(axis, zoomRange(r, factor, isLog(axis), about[axis]))
      }
    },
    [shown, isLog, write],
  )
  const pan = useCallback(
    (axis: Axis, fraction: number) => {
      const r = shown[axis]
      if (validRange(r, isLog(axis))) write(axis, panRange(r, fraction, isLog(axis)))
    },
    [shown, isLog, write],
  )
  const reset = useCallback(() => {
    write('x', undefined)
    write('y', undefined)
  }, [write])
  const zoomed = view.x !== undefined || view.y !== undefined
  return useMemo(() => ({ view, shown, zoomed, set, zoom, pan, reset }), [view, shown, zoomed, set, zoom, pan, reset])
}

/** Ranges widened about their centres so both axes have the same units per pixel in a plot of `plot` pixels. */
export function equalUnits(
  x: Range | undefined,
  y: Range | undefined,
  plot: { width: number; height: number },
): [Range, Range] | null {
  if (!validRange(x) || !validRange(y) || plot.width <= 0 || plot.height <= 0) return null
  const unit = Math.max((x[1] - x[0]) / plot.width, (y[1] - y[0]) / plot.height)
  const widen = (r: Range, pixels: number): Range => {
    const c = (r[0] + r[1]) / 2
    return [c - (unit * pixels) / 2, c + (unit * pixels) / 2]
  }
  return [widen(x, plot.width), widen(y, plot.height)]
}

/**
 * Equal units per pixel with one axis held: the `fix` axis keeps its range and the other takes the same units per pixel
 * about its centre, e.g. an equal-aspect panel whose x is shared with the panels below it.
 */
export function fixedUnits(
  x: Range | undefined,
  y: Range | undefined,
  plot: { width: number; height: number },
  fix: 'x' | 'y',
): [Range, Range] | null {
  if (!validRange(x) || !validRange(y) || plot.width <= 0 || plot.height <= 0) return null
  const unit = fix === 'x' ? (x[1] - x[0]) / plot.width : (y[1] - y[0]) / plot.height
  const about = (r: Range, pixels: number): Range => {
    const c = (r[0] + r[1]) / 2
    return [c - (unit * pixels) / 2, c + (unit * pixels) / 2]
  }
  return fix === 'x' ? [x, about(y, plot.height)] : [about(x, plot.width), y]
}

/**
 * A viewport whose zoom, pan and range fields work on the ranges the chart actually draws (`drawn`), which may differ
 * from the viewport's own (an equal-aspect chart widens one axis). With `together`, zooming one axis zooms both, so an
 * equal-aspect chart keeps its aspect.
 */
export function drawnViewport(
  viewport: Viewport,
  drawn: { x?: Range; y?: Range },
  log: { x?: boolean; y?: boolean },
  together: boolean,
): Viewport {
  const isLog = (axis: Axis) => !!(axis === 'x' ? log.x : log.y)
  return {
    ...viewport,
    shown: drawn,
    zoom: (factor, axes = ['x', 'y'], about = {}) => {
      for (const axis of together ? (['x', 'y'] as Axis[]) : axes) {
        const r = drawn[axis]
        if (validRange(r, isLog(axis))) viewport.set(axis, zoomRange(r, factor, isLog(axis), about[axis]))
      }
    },
    pan: (axis, fraction) => {
      const r = drawn[axis]
      if (validRange(r, isLog(axis))) viewport.set(axis, panRange(r, fraction, isLog(axis)))
    },
  }
}
