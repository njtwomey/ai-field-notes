import { ChevronRight } from 'lucide-react'
import { Fragment } from 'react'
import { Link } from 'react-router'
import { TagPill } from '@/components/browse/TagPill'
import { kindIcons } from '@/components/layout/kind-icon'
import { browseUrl, categoryTrail, kindLabels, type NoteMeta } from '@/lib/content'

/**
 * The note's opening, under the <NoteBar> that carries the title: a meta line (category trail ending in this note,
 * kind, status), then the summary set as an abstract with the tags beneath.
 */
export function NoteHeader({ note }: { note: NoteMeta }) {
  const Icon = kindIcons[note.kind]
  return (
    <header className="space-y-4">
      <nav
        aria-label="Breadcrumb"
        className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground"
      >
        {categoryTrail(note.category).map((c) => (
          <Fragment key={c.path}>
            <Link to={browseUrl({ c: c.path })} className="hover:text-foreground">
              {c.title}
            </Link>
            <ChevronRight className="size-3.5" aria-hidden />
          </Fragment>
        ))}
        <span aria-current="page" className="text-foreground">
          {note.title}
        </span>
        <span aria-hidden>·</span>
        <span className="inline-flex items-center gap-1">
          <Icon className="size-3.5" aria-hidden />
          {kindLabels[note.kind]}
        </span>
        {note.status !== 'stable' && (
          <>
            <span aria-hidden>·</span>
            <span className="italic">{note.status}</span>
          </>
        )}
      </nav>
      {/* Set like a paper's abstract: a narrower, centred block of ordinary body text, with keywords beneath. */}
      <div className="mx-auto max-w-2xl space-y-3 py-4">
        <h1 className="pb-2 text-center font-prose text-4xl leading-tight font-bold text-balance">{note.title}</h1>
        <p className="text-justify font-prose text-[1.0625rem] leading-relaxed hyphens-auto">{note.summary}</p>
        {note.tags.length > 0 && (
          <div className="flex flex-wrap justify-center gap-1.5">
            {note.tags.map((t) => (
              <TagPill key={t} tag={t} />
            ))}
          </div>
        )}
      </div>
    </header>
  )
}
