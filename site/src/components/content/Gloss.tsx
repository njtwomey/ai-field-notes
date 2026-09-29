import { BookA, FileText } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { notesBySlug, noteUrl, prefetchNote } from '@/lib/content'
import { glossaryEntry, glossaryKindLabels, glossaryUrl, glossText, type GlossForm } from '@/lib/glossary'
import { MathText } from './MathText'

/**
 * A glossary term: `<Gloss name="kullback-leibler-divergence" />`. Checked at build against content/glossary.yaml.
 *
 * The first use in a note reads "Long Form (SHORT)" and later uses "SHORT"; the build marks the first use (`first`, see
 * plugins/rehype-gloss.ts), so notes never set it. `form` forces one reading, `plural` appends "s" (or uses the entry's
 * `longPlural`), and children replace the words shown, e.g. to inflect a concept. Hovering shows the definition.
 */
export function Gloss({
  name,
  first = false,
  form,
  plural = false,
  children,
}: {
  name: string
  first?: boolean
  form?: GlossForm
  plural?: boolean
  children?: ReactNode
}) {
  const entry = glossaryEntry(name)
  if (!entry) return <span className="text-destructive">{children ?? name}</span>
  const note = entry.note ? notesBySlug.get(entry.note) : undefined
  // Without a note of its own, an entry points at the first note that discusses it.
  const link = note ?? entry.see.flatMap((s) => notesBySlug.get(s) ?? [])[0]
  return (
    <HoverCard>
      <HoverCardTrigger
        render={
          // A term with a note of its own opens that note; any other term only shows its definition on hover.
          note ? (
            <Link
              to={noteUrl(note.slug)}
              onMouseEnter={() => prefetchNote(note.slug)}
              className="underline decoration-muted-foreground/60 decoration-dotted underline-offset-2"
            />
          ) : (
            <span
              tabIndex={0}
              className="cursor-help underline decoration-muted-foreground/60 decoration-dotted underline-offset-2"
            />
          )
        }
      >
        {children ?? <MathText text={glossText(entry, form ?? (first ? 'full' : 'short'), plural)} />}
      </HoverCardTrigger>
      <HoverCardContent className="w-80" side="top">
        <div className="space-y-2">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              {entry.short && (
                <div className="text-sm font-semibold">
                  <MathText text={entry.short} />
                </div>
              )}
              <div className={entry.short ? 'text-xs text-muted-foreground' : 'text-sm font-semibold'}>
                <MathText text={entry.long} />
              </div>
            </div>
            <Badge variant="secondary" className="shrink-0 text-[10px]">
              {entry.sense ?? glossaryKindLabels[entry.kind]}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            <MathText text={entry.definition} />
          </p>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            {link && (
              <Link
                to={noteUrl(link.slug)}
                onMouseEnter={() => prefetchNote(link.slug)}
                className="inline-flex items-center gap-1 hover:underline"
              >
                <FileText className="size-3" aria-hidden />
                {note ? 'Read the note' : 'Discussed in'}: {link.title}
              </Link>
            )}
            <Link to={glossaryUrl(entry.key)} className="inline-flex items-center gap-1 hover:underline">
              <BookA className="size-3" aria-hidden />
              Glossary
            </Link>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
