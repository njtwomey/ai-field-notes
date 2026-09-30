import { Check, Copy, Link2 } from 'lucide-react'
import {
  Component,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react'
import { Button } from '@lab/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@lab/ui/select'
import { cn } from '@lab/lib/utils'
import { FrameContext, Readout, Readouts, type FrameContextValue, type HoverInfo } from '@lab/viz'
import { Controls } from './Controls'
import { useFigureId } from './figure-ids'
import { FIGURE_SIZES, FigureScope, type FigureSize } from './figure-size'

type Stored = { preset: FigureSize } | { width: number; height: number }

const MIN_WIDTH = 280
const MIN_HEIGHT = 140
const MAX_HEIGHT = 1600

const storageKey = (scope: string, id: string) => `aifn-lab:figure-size:${scope}/${id}`

function readSize(key: string): Stored | null {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const v = JSON.parse(raw) as Stored
    if ('preset' in v && v.preset in FIGURE_SIZES) return v
    if ('width' in v && Number.isFinite(v.width) && Number.isFinite(v.height)) return v
  } catch {
    // Unavailable storage or a malformed entry: fall back to the default size.
  }
  return null
}

function writeSize(key: string, v: Stored | null) {
  try {
    if (v) localStorage.setItem(key, JSON.stringify(v))
    else localStorage.removeItem(key)
  } catch {
    // Storage unavailable (private mode): the size lasts for this page view only.
  }
}

export type FigureProps = {
  title: string
  /**
   * The figure's anchor (`/<module>/<specimen>#<id>`) and the key of its remembered size; defaults to the title's slug,
   * made unique within the page.
   */
  id?: string
  /** One or two sentences under the title: what the figure shows. */
  description?: ReactNode
  /** Parameter controls, in a responsive grid above the chart area. */
  controls?: ReactNode
  /** Readouts under the chart area (use `Readout` from @lab/viz). */
  readouts?: ReactNode
  /** What to change and what to watch, under everything. */
  caption?: ReactNode
  /**
   * The data behind the figure, for the copy-as-JSON button: a value or a function computing it on demand. By default
   * the button copies what every chart in the figure drew.
   */
  data?: unknown
  /** The initial size before the reader picks one (default M). */
  defaultSize?: FigureSize
  /** Show the values under the pointer as readouts (default true). */
  hoverReadout?: boolean
  children: ReactNode
  className?: string
}

/**
 * The standard frame for every lab figure: title and description, controls, a sized chart area, readouts (including the
 * values under the pointer) and a caption. The frame owns the chart area's size: pick a preset (S, M, L, XL, full
 * width) or drag the corner; the size is remembered per specimen. Charts inside fill the height they are given. The
 * copy button puts the figure's data on the clipboard as JSON.
 */
export function Figure({
  title,
  id,
  description,
  controls,
  readouts,
  caption,
  data,
  defaultSize = 'M',
  hoverReadout = true,
  children,
  className,
}: FigureProps) {
  const scope = useContext(FigureScope)
  const figureId = useFigureId(title, id)
  const key = storageKey(scope, figureId)
  // The size is read once per key; `stored` stays null until the reader changes it.
  const [stored, setStored] = useState<{ key: string; size: Stored | null }>(() => ({ key, size: readSize(key) }))
  const size: Stored = (stored.key === key ? stored.size : readSize(key)) ?? { preset: defaultSize }
  const box = 'preset' in size ? FIGURE_SIZES[size.preset] : size
  const choose = useCallback(
    (next: Stored | null) => {
      setStored({ key, size: next })
      writeSize(key, next)
    },
    [key],
  )

  // Chart data (for export) and hover readouts, registered by the charts inside.
  const charts = useRef(new Map<string, unknown>())
  const [hover, setHoverMap] = useState<ReadonlyMap<string, HoverInfo>>(new Map())
  const setData = useCallback((k: string, d: unknown) => {
    if (d === undefined) charts.current.delete(k)
    else charts.current.set(k, d)
  }, [])
  const setHover = useCallback((k: string, h: HoverInfo | null) => {
    setHoverMap((m) => {
      if (!h && !m.has(k)) return m
      const next = new Map(m)
      if (h) next.set(k, h)
      else next.delete(k)
      return next
    })
  }, [])
  const frame = useMemo<FrameContextValue>(
    () => ({ height: box.height, setData, setHover: hoverReadout ? setHover : undefined }),
    [box.height, setData, setHover, hoverReadout],
  )
  const exportData = () => {
    const value = typeof data === 'function' ? (data as () => unknown)() : data
    if (value !== undefined) return value
    const drawn = [...charts.current.values()].map((d) => (typeof d === 'function' ? (d as () => unknown)() : d))
    return { title, charts: drawn }
  }

  return (
    <section
      id={figureId}
      data-figure-id={figureId}
      // Every part keeps its natural height (shrink-0): the chart area never overlaps the readouts or the caption.
      className={cn(
        'flex shrink-0 scroll-mt-16 flex-col gap-4 rounded-xl border bg-card p-4 text-card-foreground ring-ring/60 transition-shadow duration-500 *:shrink-0 data-highlight:ring-2',
        className,
      )}
    >
      <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0 space-y-1">
          <AnchorTitle id={figureId}>{title}</AnchorTitle>
          {description && <p className="max-w-prose text-xs text-muted-foreground">{description}</p>}
        </div>
        <div className="flex items-center gap-1.5">
          <SizePicker size={size} onChange={choose} />
          <CopyData data={exportData} />
        </div>
      </header>
      {controls && <Controls>{controls}</Controls>}
      <FrameContext.Provider value={frame}>
        <ChartArea width={box.width} height={box.height} onResize={choose}>
          <FigureBoundary>{children}</FigureBoundary>
        </ChartArea>
      </FrameContext.Provider>
      {(readouts || hoverReadout) && (
        <div className="flex flex-col gap-1.5">
          {readouts && <Readouts>{readouts}</Readouts>}
          {hoverReadout && <HoverReadout hover={hover} />}
        </div>
      )}
      {caption && <p className="max-w-prose text-xs text-muted-foreground">{caption}</p>}
    </section>
  )
}

/** The values under the pointer, one line per hovered position (linked charts share one). */
function HoverReadout({ hover }: { hover: ReadonlyMap<string, HoverInfo> }) {
  const lines = new Map<string, HoverInfo['rows']>()
  for (const h of hover.values()) lines.set(h.at, [...(lines.get(h.at) ?? []), ...h.rows])
  return (
    <div className="min-h-4 text-xs" aria-live="off">
      {lines.size === 0 ? (
        <span className="text-muted-foreground/70">Hover a chart to read its values here.</span>
      ) : (
        [...lines].map(([at, rows]) => (
          <Readouts key={at}>
            <span className="font-mono text-muted-foreground tabular-nums">{at}</span>
            {rows.map((r, i) => (
              <Readout key={i} label={r.label} value={r.value} color={r.color} />
            ))}
          </Readouts>
        ))
      )}
    </div>
  )
}

const LABELS: Record<FigureSize | 'custom', string> = {
  S: 'S',
  M: 'M',
  L: 'L',
  XL: 'XL',
  full: 'Full width',
  custom: 'Custom',
}

function SizePicker({ size, onChange }: { size: Stored; onChange: (s: Stored | null) => void }) {
  const value = 'preset' in size ? size.preset : 'custom'
  const describe = (p: FigureSize) => {
    const { width, height } = FIGURE_SIZES[p]
    return `${LABELS[p]} · ${typeof width === 'number' ? width : 'page'} × ${height}`
  }
  return (
    <Select
      value={value}
      onValueChange={(v) => v && v !== 'custom' && onChange({ preset: v as FigureSize })}
      items={[
        ...(Object.keys(FIGURE_SIZES) as FigureSize[]).map((p) => ({ value: p, label: LABELS[p] })),
        { value: 'custom', label: `${'width' in size ? `${size.width} × ${size.height}` : 'Custom'}` },
      ]}
    >
      <SelectTrigger size="sm" className="h-7 text-xs" aria-label="Figure size" title="Figure size">
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="end" alignItemWithTrigger={false}>
        {(Object.keys(FIGURE_SIZES) as FigureSize[]).map((p) => (
          <SelectItem key={p} value={p} className="text-xs">
            {describe(p)}
          </SelectItem>
        ))}
        {value === 'custom' && (
          <SelectItem value="custom" className="text-xs">
            Custom (dragged)
          </SelectItem>
        )}
      </SelectContent>
    </Select>
  )
}

/** Copies the figure's data as JSON; falls back to downloading a file if the clipboard is unavailable. */
function CopyData({ data }: { data: () => unknown }) {
  const [done, setDone] = useState(false)
  const copy = async () => {
    const text = JSON.stringify(data(), (_, v: unknown) => (ArrayBuffer.isView(v) ? Array.from(v as never) : v), 2)
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
      const a = Object.assign(document.createElement('a'), { href: url, download: 'figure-data.json' })
      a.click()
      URL.revokeObjectURL(url)
    }
    setDone(true)
    setTimeout(() => setDone(false), 1500)
  }
  return (
    <Button variant="ghost" size="icon-sm" aria-label="Copy data as JSON" title="Copy data as JSON" onClick={copy}>
      {done ? <Check /> : <Copy />}
    </Button>
  )
}

/**
 * The sized, resizable box the charts sit in. Its width is capped by the page; dragging the corner sets a custom size,
 * saved on release.
 */
function ChartArea({
  width,
  height,
  onResize,
  children,
}: {
  width: number | string
  height: number
  onResize: (s: Stored) => void
  children: ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState<{ width: number; height: number } | null>(null)
  const start = useRef<{ x: number; y: number; width: number; height: number; max: number } | null>(null)

  const down = (e: ReactPointerEvent<HTMLDivElement>) => {
    const el = ref.current
    if (!el || e.button !== 0) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    const max = el.parentElement?.clientWidth ?? Infinity
    start.current = { x: e.clientX, y: e.clientY, width: el.offsetWidth, height, max }
  }
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    const s = start.current
    if (!s) return
    setDrag({
      width: Math.round(Math.min(Math.max(s.width + e.clientX - s.x, MIN_WIDTH), s.max)),
      height: Math.round(Math.min(Math.max(s.height + e.clientY - s.y, MIN_HEIGHT), MAX_HEIGHT)),
    })
  }
  const up = () => {
    start.current = null
    if (drag) onResize(drag)
    setDrag(null)
  }
  const shown = drag ?? { width, height }

  return (
    <div
      ref={ref}
      className="group/area relative h-fit max-w-full"
      style={{ width: shown.width }}
      data-resizing={drag ? '' : undefined}
    >
      <FrameHeight height={shown.height}>{children}</FrameHeight>
      <div
        role="separator"
        aria-label="Resize figure"
        title="Drag to resize"
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
        className="absolute -right-2 -bottom-2 size-4 cursor-nwse-resize touch-none rounded-sm opacity-40 transition-opacity group-hover/area:opacity-100 group-data-resizing/area:opacity-100"
      >
        <svg viewBox="0 0 16 16" className="size-4 text-muted-foreground" aria-hidden>
          <path d="M14 6 6 14M14 10l-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>
      {drag && (
        <div className="pointer-events-none absolute right-3 bottom-3 rounded bg-foreground/80 px-1.5 py-0.5 font-mono text-[11px] text-background tabular-nums">
          {drag.width} × {drag.height}
        </div>
      )}
    </div>
  )
}

/** Passes a live (dragging) height to the charts without rebuilding the frame's other context values. */
function FrameHeight({ height, children }: { height: number; children: ReactNode }) {
  const outer = useContext(FrameContext)
  const value = useMemo(() => ({ ...outer, height }), [outer, height])
  return <FrameContext.Provider value={value}>{children}</FrameContext.Provider>
}

/** Contains a crash inside one figure, so the rest of the page still renders, and names the error. */
class FigureBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div
        role="alert"
        className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive"
      >
        This figure failed to render: {this.state.error.message}
      </div>
    )
  }
}

/**
 * A figure's title with a link to it: the link icon appears on hover, sets the URL's hash to the figure (which scrolls
 * to it and highlights it) and copies the full URL.
 */
function AnchorTitle({ id, children }: { id: string; children: ReactNode }) {
  return (
    <h2 className="group/anchor flex items-center gap-1.5 text-sm leading-snug font-medium">
      <span>{children}</span>
      <a
        href={`#${id}`}
        aria-label="Link to this figure (copies the URL)"
        title="Link to this figure (copies the URL)"
        className="rounded text-muted-foreground opacity-0 transition-opacity group-hover/anchor:opacity-100 hover:text-foreground focus-visible:opacity-100"
        onClick={(e) => {
          e.preventDefault()
          location.hash = id
          navigator.clipboard?.writeText(location.href).catch(() => {})
        }}
      >
        <Link2 className="size-3.5" />
      </a>
    </h2>
  )
}
