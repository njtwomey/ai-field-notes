import { TagPill } from '@/components/browse/TagPill'
import type { NoteMeta } from '@/lib/content'
import { MathText } from '@/components/content/MathText'

/**
 * The note's opening, under the <NoteBar> that carries the breadcrumb: the centred title (the page's h1), then the
 * summary set as an abstract with the tags beneath.
 */
export function NoteHeader({ note }: { note: NoteMeta }) {
  return (
    <header className="space-y-4">
      {/* Set like a paper's abstract: a narrower, centred block of ordinary body text, with keywords beneath. */}
      <div className="mx-auto max-w-2xl space-y-3 py-4">
        <h1 className="pb-2 text-center font-prose text-4xl leading-tight font-bold text-balance">{note.title}</h1>
        <p className="text-justify font-prose text-[1.0625rem] leading-relaxed hyphens-auto">
          <MathText text={note.summary} />
        </p>
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
