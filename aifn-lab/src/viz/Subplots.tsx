import { Children, useCallback, useContext, useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { Maximize } from 'lucide-react'
import { Button } from '@lab/ui/button'
import { cn } from '@lab/lib/utils'
import { FrameContext, useChartHeight, useElementSize } from './frame'
import { layoutColumn } from './subplot-layout'
import {
  DEFAULT_MARGINS,
  LABEL_GAP,
  niceRange,
  shareGroup,
  SubplotContext,
  Y_NAME_ROOM,
  type HoldSettings,
  type PanelReport,
  type Share,
  type SubplotContextValue,
} from './subplot-context'
import { GRID } from './theme'
import type { AxisStore, Range } from './viewport'

export type SubplotsProps = {
  rows?: number
  cols?: number
  /** Share x across every panel (`true`), across each row or down each column. Shared x shows ticks on the bottom row only. */
  sharex?: Share
  /** Share y likewise. Shared y shows ticks on the left column only. */
  sharey?: Share
  /** Relative row heights, e.g. [2, 1]; the rows split the frame's height. */
  heightRatios?: readonly number[]
  /** Relative column widths. */
  widthRatios?: readonly number[]
  /** Link hover across every panel: true, or the name of a hover group shared with charts elsewhere. */
  hoverGroup?: boolean | string
  /** Total height in pixels outside a Figure; inside one the frame sets it. */
  height?: number
  /** Log axes of the shared axes, so their fitted range rounds to whole decades. */
  xLog?: boolean
  yLog?: boolean
  /** Hold every chart's axes while data changes (see `XYChartProps.rescaleOnChange`); a Panel may override. */
  rescaleOnChange?: boolean
  /** Refit held axes when this changes. */
  axisKey?: string | number
  /** `initial` (default) or `union`: how held axes follow new data. */
  holdFit?: 'initial' | 'union'
  /**
   * With an equal-aspect panel in a single column: `frame` (default) splits the rest of the frame among the other rows
   * by `heightRatios`; `equal` makes each other row's plot `heightRatios[i] / heightRatios[equal row]` times as tall as
   * the equal panel's plot, e.g. `[1, 0.22]` for a short strip under a square.
   */
  ratiosOf?: 'frame' | 'equal'
  /** The grid's toolbar row (the fit-all button; default true). Hide it when no panel can zoom. */
  toolbar?: boolean
  /**
   * Panels stacked with a shared x sit nearly edge to edge: a 2 px gap, no x tick room under inner panels and only
   * tick-label room above lower ones. For a strip that belongs to the panel above it (bin counts under a diagram).
   */
  tight?: boolean
  /**
   * A scatter matrix: every plot area the same size (columns and rows absorb their own tick-label margins), and panels
   * without tick labels nearly edge to edge. Panels in a dense grid draw no legend.
   */
  dense?: boolean
  /** One `Panel` per cell, in row-major order. */
  children: ReactNode
  className?: string
}

const GAP = 8
/** The least height of a panel beside an equal-aspect one. */
const MIN_PANEL = 160
const TOOLBAR = 28
/** The bottom margin of a panel whose x tick labels are hidden. */
const INNER_BOTTOM = 14
/** In a tight grid: the gap between panels, an inner panel's bottom margin, and a lower panel's top margin. */
const TIGHT_GAP = 2
const TIGHT_BOTTOM = 6
const TIGHT_TOP = 8
/** In a dense grid: the gap between cells and the margin on each side of a plot without tick labels. */
const DENSE_GAP = 2
const DENSE_MARGIN = 5

type Reports = ReadonlyMap<string, { row: number; col: number; report: PanelReport }>

/**
 * A grid of charts, like matplotlib's `subplots`: panels in a column share left and right plot edges, panels in a row
 * share top and bottom edges, shared axes have one range (zoom, pan and fit on any panel move all), and hover can be
 * linked. The grid splits the height its frame gives it by `heightRatios`. Put one `Panel` per cell, each holding one
 * XYChart. An `equalAspect` panel keeps the shared axis's range and widens (or narrows) its own other axis.
 */
export function Subplots({
  rows = 1,
  cols = 1,
  sharex = false,
  sharey = false,
  heightRatios,
  widthRatios,
  hoverGroup,
  height: ownHeight,
  xLog = false,
  yLog = false,
  rescaleOnChange,
  axisKey,
  holdFit,
  ratiosOf = 'frame',
  toolbar = true,
  tight = false,
  dense = false,
  children,
  className,
}: SubplotsProps) {
  const gap = dense ? DENSE_GAP : tight ? TIGHT_GAP : GAP
  const toolbarHeight = toolbar ? TOOLBAR : 0
  const height = useChartHeight(ownHeight)
  const generated = `subplots${useId()}`
  const group = hoverGroup === true ? generated : hoverGroup || undefined

  // Shared zoom, one entry per shared group.
  const [views, setViews] = useState<Readonly<Record<string, Range | undefined>>>({})
  const [resetKey, setResetKey] = useState(0)
  const fitAll = () => {
    setViews({})
    setResetKey((k) => k + 1)
  }
  // What each panel reports: its data extents and label width.
  const [reports, setReports] = useState<Reports>(new Map())
  // While a handle is dragged, fitted shared ranges hold still: a snapshot taken when the drag starts.
  const [frozen, setFrozen] = useState<Record<string, Range> | null>(null)
  const latest = useRef<Record<string, Range>>({})

  const fittedAll = useMemo(() => {
    const out: Record<string, Range> = {}
    const acc: Record<string, { lo: number; hi: number; explicit: boolean; log: boolean }> = {}
    for (const { row, col, report } of reports.values())
      for (const axis of ['x', 'y'] as const) {
        const key = shareGroup(axis, axis === 'x' ? sharex : sharey, row, col)
        const e = report[axis]
        if (!key || !e) continue
        const a = (acc[key] ??= { lo: Infinity, hi: -Infinity, explicit: true, log: axis === 'x' ? xLog : yLog })
        a.lo = Math.min(a.lo, e.range[0])
        a.hi = Math.max(a.hi, e.range[1])
        a.explicit &&= e.explicit
      }
    for (const [key, a] of Object.entries(acc))
      if (Number.isFinite(a.lo)) out[key] = a.explicit ? [a.lo, a.hi] : niceRange([a.lo, a.hi], a.log)
    return out
  }, [reports, sharex, sharey, xLog, yLog])
  useEffect(() => {
    latest.current = fittedAll
  })
  const setDragging = useCallback((d: boolean) => setFrozen(d ? latest.current : null), [])
  const fitted = frozen ?? fittedAll

  // Tick label widths, as the widest in each column (shared y hides them away from the left column).
  const columnLabels = useMemo(() => {
    const widths = new Array<number>(cols).fill(0)
    for (const { col, report } of reports.values()) widths[col] = Math.max(widths[col], report.labelWidth)
    return widths
  }, [reports, cols])

  const report = useCallback((key: string, r: PanelReport | null, row = 0, col = 0) => {
    setReports((m) => {
      const old = m.get(key)
      if (!r) {
        if (!old) return m
        const next = new Map(m)
        next.delete(key)
        return next
      }
      if (old && old.row === row && old.col === col && JSON.stringify(old.report) === JSON.stringify(r)) return m
      return new Map(m).set(key, { row, col, report: r })
    })
  }, [])
  // One reporter per cell, stable across renders. A panel's report effect depends on its reporter, and its cleanup
  // withdraws the report: a new function each render would withdraw and re-add every report on every render, and each
  // re-add renders the grid again, without end.
  const [reporters] = useState(() => new Map<string, SubplotContextValue['report']>())
  const reporterAt = (row: number, col: number) => {
    const at = `${row},${col}`
    let f = reporters.get(at)
    if (!f) reporters.set(at, (f = (key, r) => report(key, r, row, col)))
    return f
  }
  const setView = useCallback(
    (key: string, r: Range | undefined) => setViews((v) => (v[key] === r ? v : { ...v, [key]: r })),
    [],
  )

  const cells = Children.toArray(children)
  const [outerBox, outerSize] = useElementSize<HTMLDivElement>()
  const ratios = useMemo(
    () => Array.from({ length: rows }, (_, i) => heightRatios?.[i] ?? 1),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the ratios' values
    [rows, heightRatios?.join(',')],
  )
  const labelsAt = (row: number, col: number) => ({
    x: !(sharex === true || sharex === 'col') || row === rows - 1,
    y: !(sharey === true || sharey === 'row') || col === 0,
  })
  const marginsAt = (row: number, col: number) => {
    const labels = labelsAt(row, col)
    const labelWidth = columnLabels[col]
    const stacked = tight && (sharex === true || sharex === 'col')
    const left = labels.y ? (labelWidth ? labelWidth + LABEL_GAP + Y_NAME_ROOM : DEFAULT_MARGINS.left) : 16
    if (dense)
      return {
        left: labels.y ? left : DENSE_MARGIN,
        right: DENSE_MARGIN,
        top: DENSE_MARGIN,
        bottom: labels.x ? GRID.bottom : DENSE_MARGIN,
      }
    return {
      left,
      right: GRID.right,
      top: stacked && row > 0 ? TIGHT_TOP : GRID.top,
      bottom: labels.x ? GRID.bottom : stacked ? TIGHT_BOTTOM : INNER_BOTTOM,
    }
  }

  const ratioHeights = useMemo(() => {
    const total = ratios.reduce((a, b) => a + b, 0)
    // The toolbar row takes its share of the frame's height, so the grid never overflows the frame.
    const free = Math.max(height - toolbarHeight - gap * rows, rows * 60)
    if (!dense) return ratios.map((r) => Math.round((free * r) / total))
    // Dense: the plot areas share what the rows' own margins leave, so every plot is as tall as its ratio says.
    const edges = ratios.map((_, row) => {
      const m = marginsAt(row, 0)
      return m.top + m.bottom
    })
    const plots = Math.max(free - edges.reduce((a, b) => a + b, 0), rows * 20)
    return ratios.map((r, row) => Math.round(edges[row] + (plots * r) / total))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- marginsAt reads the values listed
  }, [ratios, rows, height, toolbarHeight, gap, dense, columnLabels, sharex, sharey])
  const columns = dense
    ? (() => {
        // Each column is its share of the plot width plus its own margins, so every plot area is equally wide.
        const edges = Array.from({ length: cols }, (_, col) => {
          const m = marginsAt(rows - 1, col)
          return m.left + m.right
        })
        const fixed = edges.reduce((a, b) => a + b, 0) + gap * (cols - 1)
        const total = Array.from({ length: cols }, (_, i) => widthRatios?.[i] ?? 1).reduce((a, b) => a + b, 0)
        return edges.map((e, i) => `calc((100% - ${fixed}px) * ${(widthRatios?.[i] ?? 1) / total} + ${e}px)`).join(' ')
      })()
    : Array.from({ length: cols }, (_, i) => `minmax(0, ${widthRatios?.[i] ?? 1}fr)`).join(' ')

  // Equal units: in a single column, the panel that asks for them is sized from its fitted ranges, the others share
  // the rest of the frame, and the column narrows (centred) if the total would pass the cap.
  const equal = [...reports.values()].find((r) => r.report.equal && r.col === 0)
  const layout = (() => {
    if (cols !== 1 || !equal || outerSize.width <= 0) return null
    const gx = shareGroup('x', sharex, equal.row, 0)
    const xr = gx ? fitted[gx] : equal.report.fit?.x
    const yr = equal.report.fit?.y
    if (!xr || !yr || !(xr[1] > xr[0]) || !(yr[1] > yr[0])) return null
    const chrome = Array.from({ length: rows }, (_, row) => {
      const r = [...reports.values()].find((v) => v.row === row && v.col === 0)
      return r?.report.chrome ?? 0
    })
    const m = Array.from({ length: rows }, (_, row) => marginsAt(row, 0))
    const cap = typeof window === 'undefined' ? height * 1.5 : 0.85 * window.innerHeight - toolbarHeight
    return layoutColumn({
      width: outerSize.width,
      frameHeight: height - toolbarHeight,
      cap,
      ratios,
      equal: { row: equal.row, xSpan: xr[1] - xr[0], ySpan: yr[1] - yr[0] },
      margins: { left: m[0].left, right: m[0].right, top: m.map((v) => v.top), bottom: m.map((v) => v.bottom) },
      chrome,
      gap,
      minHeight: ratiosOf === 'equal' ? 40 : MIN_PANEL,
      ratiosOf,
    })
  })()
  const heights = layout?.heights ?? ratioHeights

  const panel = (index: number): SubplotContextValue => {
    const row = Math.floor(index / cols)
    const col = index % cols
    const gx = shareGroup('x', sharex, row, col)
    const gy = shareGroup('y', sharey, row, col)
    const store = (key: string | null): AxisStore | undefined =>
      key ? { view: views[key], set: (r) => setView(key, r) } : undefined
    // Tick labels: hidden on inner panels along the direction an axis is shared.
    const labels = labelsAt(row, col)
    // Controls: one set per shared axis, on the panel that carries its labels (bottom-left for an axis shared by all).
    const controls = {
      x: !gx || (sharex === 'row' ? col === 0 : sharex === 'col' ? row === rows - 1 : row === rows - 1 && col === 0),
      y: !gy || (sharey === 'col' ? row === 0 : sharey === 'row' ? col === 0 : col === 0 && row === 0),
    }
    const labelWidth = columnLabels[col]
    return {
      row,
      col,
      rows,
      cols,
      shared: { x: !!gx, y: !!gy },
      labels,
      controls,
      resetKey,
      stores: { x: store(gx), y: store(gy) },
      fitted: { x: gx ? fitted[gx] : undefined, y: gy ? fitted[gy] : undefined },
      margins: marginsAt(row, col),
      sizedEqual: !!layout && equal?.row === row,
      yNameGap: (labelWidth || 32) + LABEL_GAP,
      hoverGroup: group,
      dense,
      hold: { rescaleOnChange, axisKey, holdFit },
      report: reporterAt(row, col),
      setDragging,
    }
  }

  return (
    <div ref={outerBox} className={cn('flex w-full flex-col', className)} style={{ gap }}>
      {toolbar && (
        <div className="flex items-center justify-end gap-2" style={{ height: TOOLBAR - GAP }}>
          <Button
            variant="outline"
            size="icon-xs"
            onClick={fitAll}
            aria-label="Auto-scale: fit every panel to its data"
            title="Auto-scale: fit every panel to its data"
          >
            <Maximize />
          </Button>
        </div>
      )}
      <div
        className="mx-auto grid w-full"
        style={{
          gridTemplateColumns: columns,
          gridTemplateRows: heights.map((h) => `${h}px`).join(' '),
          gap,
          // A column narrowed to fit an equal-aspect panel under the cap, centred.
          ...(layout && layout.width < outerSize.width ? { width: layout.width } : {}),
        }}
      >
        {cells.map((child, i) => (
          <PanelSlot key={i} value={panel(i)} height={heights[Math.floor(i / cols)] ?? heights[0]}>
            {child}
          </PanelSlot>
        ))}
      </div>
    </div>
  )
}

/** Gives one cell its subplot context and its row's height. */
function PanelSlot({ value, height, children }: { value: SubplotContextValue; height: number; children: ReactNode }) {
  const outer = useContext(FrameContext)
  const frame = useMemo(() => ({ ...outer, height }), [outer, height])
  return (
    <SubplotContext.Provider value={value}>
      <FrameContext.Provider value={frame}>{children}</FrameContext.Provider>
    </SubplotContext.Provider>
  )
}

/**
 * One cell of a `Subplots` grid, holding one chart. Its hold settings override the grid's for its chart. `aspect="equal"`
 * asks the grid to size this panel so its units are equal with both axes on their fitted ranges (a single column only).
 */
export function Panel({
  children,
  className,
  aspect,
  share,
  ticks,
  ...hold
}: {
  children: ReactNode
  className?: string
  aspect?: 'equal' | 'fit'
  /**
   * `false` on an axis gives this panel an axis of its own although the grid shares it, e.g. the count axis of a
   * scatter matrix's diagonal histograms. The panel keeps its place in the alignment but neither reads nor widens the
   * shared range, and its tick labels do not widen the column.
   */
  share?: { x?: boolean; y?: boolean }
  /** `false` on an axis hides its tick labels but keeps its name where the grid shows it. */
  ticks?: { x?: boolean; y?: boolean }
} & HoldSettings) {
  const outer = useContext(SubplotContext)
  const { rescaleOnChange, axisKey, holdFit } = hold
  const ownX = share?.x === false
  const ownY = share?.y === false
  const ticksX = ticks?.x
  const ticksY = ticks?.y
  // Stable while the cell's reporter is (see `reporters` in Subplots): a new function each render would withdraw and
  // re-add the chart's report on every render of the grid.
  const outerReport = outer?.report
  const report = useMemo(
    (): SubplotContextValue['report'] | undefined =>
      outerReport &&
      (ownX || ownY
        ? (key, r) =>
            outerReport(
              key,
              r && { ...r, ...(ownX ? { x: undefined } : {}), ...(ownY ? { y: undefined, labelWidth: 0 } : {}) },
            )
        : outerReport),
    [outerReport, ownX, ownY],
  )
  const value = useMemo((): SubplotContextValue | null => {
    if (!outer || !report) return null
    return {
      ...outer,
      aspect: aspect ?? outer.aspect,
      shared: { x: outer.shared.x && !ownX, y: outer.shared.y && !ownY },
      stores: { x: ownX ? undefined : outer.stores.x, y: ownY ? undefined : outer.stores.y },
      fitted: { x: ownX ? undefined : outer.fitted.x, y: ownY ? undefined : outer.fitted.y },
      ticks: { x: ticksX ?? outer.ticks?.x, y: ticksY ?? outer.ticks?.y },
      report,
      hold: {
        rescaleOnChange: rescaleOnChange ?? outer.hold.rescaleOnChange,
        axisKey: axisKey ?? outer.hold.axisKey,
        holdFit: holdFit ?? outer.hold.holdFit,
      },
    }
  }, [outer, report, aspect, rescaleOnChange, axisKey, holdFit, ownX, ownY, ticksX, ticksY])
  return (
    <div className={cn('min-w-0', className)}>
      <SubplotContext.Provider value={value}>{children}</SubplotContext.Provider>
    </div>
  )
}
