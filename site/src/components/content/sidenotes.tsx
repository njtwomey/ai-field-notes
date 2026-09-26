/**
 * Margin references. Each <Cite> marker registers its element here. <MarginNotes>, in the page's right column, places
 * one note per marker at the marker's height, pushing notes down when they would overlap. Hovering a marker highlights
 * its note, and hovering a note highlights its markers.
 */
import {
  createContext,
  useCallback,
  useContext,
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

type Anchor = { key: string; el: HTMLElement }

type SidenoteContext = {
  register: (id: string, anchor: Anchor | null) => void
  anchors: Map<string, Anchor>
  version: number
  active: string | null
  setActive: (key: string | null) => void
  /** Element whose size changes should trigger a re-layout: the article column. */
  contentRef: RefObject<HTMLElement | null>
}

const Context = createContext<SidenoteContext | null>(null)

export function SidenoteProvider({
  children,
  contentRef,
}: {
  children: ReactNode
  contentRef: RefObject<HTMLElement | null>
}) {
  const anchors = useRef(new Map<string, Anchor>())
  const [version, setVersion] = useState(0)
  const [active, setActive] = useState<string | null>(null)
  const frame = useRef(0)

  const register = useCallback((id: string, anchor: Anchor | null) => {
    if (anchor) anchors.current.set(id, anchor)
    else anchors.current.delete(id)
    // Batch registrations from one render into one layout pass.
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => setVersion((v) => v + 1))
  }, [])

  const value = useMemo(
    () => ({ register, anchors: anchors.current, version, active, setActive, contentRef }),
    [register, version, active, contentRef],
  )
  return <Context.Provider value={value}>{children}</Context.Provider>
}

export function useSidenotes(): SidenoteContext | null {
  return useContext(Context)
}

type Placed = { id: string; key: string; top: number; first: boolean }

/** Right-column notes aligned to their citation markers. Render inside the column that sits beside the article. */
export function MarginNotes() {
  const ctx = useSidenotes()
  const note = useCurrentNote()
  const root = useRef<HTMLDivElement>(null)
  const heights = useRef(new Map<string, number>())
  const [placed, setPlaced] = useState<Placed[]>([])
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
    setPlaced((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next))
  }, [ctx, ctx?.version, tick])

  // Heights are known only after the notes render; a change schedules one more layout pass.
  const measure = (id: string) => (el: HTMLDivElement | null) => {
    if (!el) return
    const h = el.offsetHeight
    if (heights.current.get(id) !== h) {
      heights.current.set(id, h)
      requestAnimationFrame(() => setTick((t) => t + 1))
    }
  }

  if (!ctx) return null
  const last = placed.at(-1)
  const height = last ? last.top + (heights.current.get(last.id) ?? 48) : 0

  return (
    <div ref={root} className="relative" style={{ height }} aria-label="Margin references">
      {placed.map(({ id, key, top, first }) => {
        const ref = references[key]
        const number = note.cited.indexOf(key) + 1
        return (
          <div
            key={id}
            ref={measure(id)}
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
