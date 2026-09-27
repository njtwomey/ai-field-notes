/**
 * Margin references. Each <Cite> marker registers its element here. <MarginNotes>, in the page's right column, places
 * one note per marker at the marker's height, pushing notes down when they would overlap. Hovering a marker highlights
 * its note, and hovering a note highlights its markers.
 */
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from 'react'
import { formatAuthors, references } from '@/lib/content'
import { cn } from '@/lib/utils'
import { useCurrentNote } from './note-context'
import { ReferenceCard } from './ReferenceCard'
import { SidenoteContext, useSidenotes, type Anchor } from './sidenotes-context'

export function SidenoteProvider({
  children,
  contentRef,
}: {
  children: ReactNode
  contentRef: RefObject<HTMLElement | null>
}) {
  // One map for the provider's lifetime; held in state rather than a ref because the context value exposes it.
  const [anchors] = useState(() => new Map<string, Anchor>())
  const [version, setVersion] = useState(0)
  const [active, setActive] = useState<string | null>(null)
  const frame = useRef(0)

  const register = useCallback(
    (id: string, anchor: Anchor | null) => {
      if (anchor) anchors.set(id, anchor)
      else anchors.delete(id)
      // Batch registrations from one render into one layout pass.
      cancelAnimationFrame(frame.current)
      frame.current = requestAnimationFrame(() => setVersion((v) => v + 1))
    },
    [anchors],
  )

  const value = useMemo(
    () => ({ register, anchors, version, active, setActive, contentRef }),
    [register, anchors, version, active, contentRef],
  )
  return <SidenoteContext.Provider value={value}>{children}</SidenoteContext.Provider>
}

type Placed = { id: string; key: string; top: number; first: boolean }

/** Right-column notes aligned to their citation markers. Render inside the column that sits beside the article. */
export function MarginNotes() {
  const ctx = useSidenotes()
  const note = useCurrentNote()
  const root = useRef<HTMLDivElement>(null)
  const heights = useRef(new Map<string, number>())
  const [layout, setLayout] = useState<{ placed: Placed[]; height: number }>({ placed: [], height: 0 })
  const [tick, setTick] = useState(0)

  // Re-layout when the article or the window changes size, and once web fonts have loaded.
  useEffect(() => {
    const bump = () => setTick((t) => t + 1)
    const observer = new ResizeObserver(bump)
    if (ctx?.contentRef.current) observer.observe(ctx.contentRef.current)
    window.addEventListener('resize', bump)
    document.fonts?.ready.then(bump)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', bump)
    }
  }, [ctx?.contentRef])

  useLayoutEffect(() => {
    if (!ctx || !root.current) return
    const origin = root.current.getBoundingClientRect().top
    const visible = [...ctx.anchors.entries()]
      // Markers inside collapsed sections have no layout box.
      .filter(([, a]) => a.el.isConnected && a.el.getClientRects().length > 0)
      .map(([id, a]) => ({ id, key: a.key, y: a.el.getBoundingClientRect().top - origin }))
      .sort((a, b) => a.y - b.y)
    const seen = new Set<string>()
    let cursor = 0
    const next = visible.map(({ id, key, y }) => {
      const top = Math.max(y - 4, cursor)
      cursor = top + (heights.current.get(id) ?? 48) + 12
      const first = !seen.has(key)
      seen.add(key)
      return { id, key, top: Math.round(top), first }
    })
    const last = next.at(-1)
    const height = last ? last.top + (heights.current.get(last.id) ?? 48) : 0
    setLayout((prev) =>
      prev.height === height && JSON.stringify(prev.placed) === JSON.stringify(next) ? prev : { placed: next, height },
    )
  }, [ctx, ctx?.version, tick])

  // Heights are known only after the notes render; a change schedules one more layout pass.
  const measure = useCallback((el: HTMLDivElement | null) => {
    if (!el) return
    const id = el.dataset.id!
    const h = el.offsetHeight
    if (heights.current.get(id) !== h) {
      heights.current.set(id, h)
      requestAnimationFrame(() => setTick((t) => t + 1))
    }
  }, [])

  if (!ctx) return null
  const { placed, height } = layout

  return (
    <div ref={root} className="relative" style={{ height }} aria-label="Margin references">
      {placed.map(({ id, key, top, first }) => {
        const ref = references[key]
        if (!ref) return null
        const number = note.cited.indexOf(key) + 1
        return (
          <div
            key={id}
            ref={measure}
            data-id={id}
            onMouseEnter={() => ctx.setActive(key)}
            onMouseLeave={() => ctx.setActive(null)}
            className={cn(
              'absolute inset-x-0 flex gap-2 rounded-md border-l-2 py-1 pr-1 pl-2 text-sm transition-colors',
              ctx.active === key ? 'border-primary bg-muted' : 'border-border',
            )}
            style={{ top }}
          >
            <span className="w-5 shrink-0 text-xs text-muted-foreground tabular-nums">[{number}]</span>
            {first ? (
              <ReferenceCard reference={ref} compact />
            ) : (
              <a
                href={ref.url}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-muted-foreground hover:underline"
              >
                {[formatAuthors(ref.authors), ref.year].filter(Boolean).join(', ')}
              </a>
            )}
          </div>
        )
      })}
    </div>
  )
}
