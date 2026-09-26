import { useEffect, useId, useRef } from 'react'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { MARGIN_QUERY, useMediaQuery } from '@/hooks/use-media-query'
import { references } from '@/lib/content'
import { cn } from '@/lib/utils'
import { useCurrentNote } from './note-context'
import { ReferenceCard } from './ReferenceCard'
import { useSidenotes } from './sidenotes'

/**
 * Inline citation: `<Cite id="vaswani2017" />` or several keys `<Cite id="a,b" />`. Markers are numbered in first-use
 * order. On wide screens the reference appears in the right margin beside the line; on narrow screens it appears on
 * hover. Every reference is also listed at the end of the note. Keys are checked at build time.
 */
export function Cite({ id }: { id: string }) {
  const keys = id.split(',').map((k) => k.trim())
  return (
    <sup className="ml-0.5 font-sans">
      {keys.map((key, i) => (
        <span key={key}>
          {i > 0 && <span className="text-muted-foreground">,</span>}
          <Marker citeKey={key} />
        </span>
      ))}
    </sup>
  )
}

function Marker({ citeKey }: { citeKey: string }) {
  const note = useCurrentNote()
  const sidenotes = useSidenotes()
  const wide = useMediaQuery(MARGIN_QUERY)
  const ref = useRef<HTMLAnchorElement>(null)
  const id = useId()
  const register = sidenotes?.register

  useEffect(() => {
    if (!register || !ref.current) return
    register(id, { key: citeKey, el: ref.current })
    return () => register(id, null)
  }, [register, id, citeKey])

  const reference = references[citeKey]
  const label = `[${note.cited.indexOf(citeKey) + 1}]`
  const className = cn(
    'rounded-sm px-0.5 text-[0.7em] font-medium text-primary no-underline transition-colors hover:underline',
    sidenotes?.active === citeKey && 'bg-primary text-primary-foreground',
  )
  const hover = {
    onMouseEnter: () => sidenotes?.setActive(citeKey),
    onMouseLeave: () => sidenotes?.setActive(null),
  }

  if (wide && sidenotes) {
    return (
      <a
        ref={ref}
        href={`#ref-${citeKey}`}
        className={className}
        aria-label={`Reference: ${reference.title}`}
        {...hover}
      >
        {label}
      </a>
    )
  }
  return (
    <HoverCard>
      <HoverCardTrigger
        render={
          <a ref={ref} href={`#ref-${citeKey}`} className={className} aria-label={`Reference: ${reference.title}`} />
        }
      >
        {label}
      </HoverCardTrigger>
      <HoverCardContent className="w-80" side="top">
        <ReferenceCard reference={reference} />
      </HoverCardContent>
    </HoverCard>
  )
}
