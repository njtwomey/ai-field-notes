import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Link } from 'react-router'
import { Button } from '@/components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { categoryTrail, noteUrl, prefetchNote, readingNeighbours, type NoteMeta } from '@/lib/content'
import { cn } from '@/lib/utils'

/** The category a neighbour sits in, named when it differs from the current note's. */
function categoryTitle(n: NoteMeta, from: NoteMeta): string | undefined {
  return n.category === from.category ? undefined : categoryTrail(n.category).at(-1)?.title
}

/** One arrow: a link to the neighbour with its title on hover, or a disabled button at either end of the order. */
function Arrow({ to, from, dir }: { to?: NoteMeta; from: NoteMeta; dir: 'prev' | 'next' }) {
  const Icon = dir === 'prev' ? ChevronLeft : ChevronRight
  const label = dir === 'prev' ? 'Previous note' : 'Next note'
  if (!to)
    return (
      <Button variant="ghost" size="icon-sm" disabled aria-label={label}>
        <Icon />
      </Button>
    )
  const where = categoryTitle(to, from)
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-sm"
            nativeButton={false}
            aria-label={`${label}: ${to.title}`}
            onPointerEnter={() => prefetchNote(to.slug)}
            onFocus={() => prefetchNote(to.slug)}
            render={<Link to={noteUrl(to.slug)} />}
          />
        }
      >
        <Icon />
      </TooltipTrigger>
      <TooltipContent side="bottom">
        <span className="text-muted-foreground">{dir === 'prev' ? 'Previous' : 'Next'}:</span> {to.title}
        {where && <span className="text-muted-foreground"> ({where})</span>}
      </TooltipContent>
    </Tooltip>
  )
}

/** Previous and next arrows as a button group, for the right end of the note bar; they follow content/reading-order.yaml. */
export function ReadingArrows({ note, className }: { note: NoteMeta; className?: string }) {
  const { prev, next } = readingNeighbours(note.slug)
  return (
    <ButtonGroup aria-label="Reading order" className={cn('shrink-0', className)}>
      <Arrow to={prev} from={note} dir="prev" />
      <Arrow to={next} from={note} dir="next" />
    </ButtonGroup>
  )
}

/** The previous and next notes as two cards, for the end of a note. */
export function ReadingFooter({ note }: { note: NoteMeta }) {
  const { prev, next } = readingNeighbours(note.slug)
  if (!prev && !next) return null
  return (
    <nav aria-label="Reading order" className="mt-12 grid grid-cols-1 gap-3 border-t pt-6 sm:grid-cols-2">
      {prev ? <FooterCard to={prev} from={note} dir="prev" /> : <span className="hidden sm:block" />}
      {next && <FooterCard to={next} from={note} dir="next" />}
    </nav>
  )
}

function FooterCard({ to, from, dir }: { to: NoteMeta; from: NoteMeta; dir: 'prev' | 'next' }) {
  const where = categoryTitle(to, from)
  return (
    <Link
      to={noteUrl(to.slug)}
      onPointerEnter={() => prefetchNote(to.slug)}
      className={cn(
        'group flex items-center gap-3 rounded-lg border px-4 py-3 outline-none hover:bg-muted/50',
        'focus-visible:ring-2 focus-visible:ring-ring',
        dir === 'next' && 'flex-row-reverse text-right sm:col-start-2',
      )}
    >
      {dir === 'prev' ? (
        <ChevronLeft className="size-4 shrink-0 text-muted-foreground group-hover:text-foreground" aria-hidden />
      ) : (
        <ChevronRight className="size-4 shrink-0 text-muted-foreground group-hover:text-foreground" aria-hidden />
      )}
      <span className="min-w-0">
        <span className="block text-xs text-muted-foreground">
          {dir === 'prev' ? 'Previous' : 'Next'}
          {where && ` · ${where}`}
        </span>
        <span className="block truncate text-sm font-medium">{to.title}</span>
      </span>
    </Link>
  )
}
