import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

export type TocItem = { id: string; text: string; depth: number }

/**
 * Heading index that shows the reader's position. The rail fills down to the current section, the current entry is
 * marked, and its parent section is emphasised. Position comes from useReadingPosition. Lives in the sticky left
 * column.
 */
export function Toc({ items, active, title = 'On this page' }: { items: TocItem[]; active?: string; title?: string }) {
  const activeRef = useRef<HTMLAnchorElement>(null)

  // Keep the current entry visible when the index itself scrolls. Only the index's own scroll box moves:
  // scrollIntoView would also pan a pinch-zoomed page across to the index, away from the text being read.
  useEffect(() => {
    const el = activeRef.current
    const box = el && scrollParent(el)
    if (!el || !box) return
    const top = el.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop
    if (top < box.scrollTop) box.scrollTop = top
    else if (top + el.offsetHeight > box.scrollTop + box.clientHeight)
      box.scrollTop = top + el.offsetHeight - box.clientHeight
  }, [active])

  if (!items.length) return null
  const activeIndex = items.findIndex((i) => i.id === active)
  // The h2 that contains the current entry.
  const parent = items.slice(0, activeIndex + 1).findLast((i) => i.depth === 2)?.id

  return (
    <nav aria-label={title} className="text-sm">
      <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</p>
      <ul>
        {items.map((item, index) => {
          const isActive = item.id === active
          const passed = index < activeIndex
          return (
            <li key={item.id}>
              <a
                ref={isActive ? activeRef : undefined}
                href={`#${item.id}`}
                aria-current={isActive ? 'location' : undefined}
                className={cn(
                  'block border-l-2 py-1 leading-snug transition-colors hover:text-foreground',
                  item.depth > 2 ? 'pl-6 text-xs' : 'pl-3',
                  isActive
                    ? 'border-primary font-medium text-foreground'
                    : passed
                      ? 'border-muted-foreground/40 text-muted-foreground'
                      : 'border-border text-muted-foreground',
                  !isActive && item.id === parent && 'text-foreground',
                )}
              >
                {item.text}
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

/** The nearest ancestor that scrolls vertically on its own, if any (not the page). */
function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
    const overflow = getComputedStyle(p).overflowY
    if ((overflow === 'auto' || overflow === 'scroll') && p.scrollHeight > p.clientHeight) return p
  }
  return null
}
