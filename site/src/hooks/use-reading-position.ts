import { useEffect, useState } from 'react'

export type Heading = { id: string; text: string; depth: number }

/** A heading counts as current once its top passes this far below the viewport top (site header + note bar). */
const READING_LINE = 140

export type ReadingPosition = {
  /** Id of the current heading, or undefined before the first one. */
  active?: string
  /** Fraction of the page scrolled, 0 to 1. */
  progress: number
}

/**
 * The heading the reader is in and how far through the page they are. Shared by the left index and the note bar so
 * the two always agree. Re-measures on scroll, resize and whenever the page's size changes (lazy content, collapsibles).
 */
export function useReadingPosition(headings: Heading[]): ReadingPosition {
  const [position, setPosition] = useState<ReadingPosition>({ progress: 0 })

  useEffect(() => {
    let frame = 0
    const update = () => {
      frame = 0
      const doc = document.documentElement
      const scrollable = doc.scrollHeight - window.innerHeight
      const atBottom = window.scrollY >= scrollable - 4
      let active: string | undefined
      for (const h of headings) {
        const el = document.getElementById(h.id)
        if (!el) continue
        if (el.getBoundingClientRect().top - READING_LINE > 0) break
        active = h.id
      }
      // At the very bottom, the last sections may never reach the reading line.
      if (atBottom && headings.length) active = headings[headings.length - 1].id
      const progress = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0
      setPosition((prev) =>
        prev.active === active && Math.abs(prev.progress - progress) < 0.002 ? prev : { active, progress },
      )
    }
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update)
    }
    update()
    window.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule)
    const observer = new ResizeObserver(schedule)
    observer.observe(document.body)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
      observer.disconnect()
    }
  }, [headings])

  return position
}

/** The current heading and, for a subsection, its parent section, outermost first. */
export function sectionTrail(headings: Heading[], active: string | undefined): Heading[] {
  const index = headings.findIndex((h) => h.id === active)
  if (index < 0) return []
  const current = headings[index]
  if (current.depth <= 2) return [current]
  const parent = headings.slice(0, index).findLast((h) => h.depth === 2)
  return parent ? [parent, current] : [current]
}
