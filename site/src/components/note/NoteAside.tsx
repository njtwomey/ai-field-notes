import { Link } from 'react-router'
import { backlinks, components, notesBySlug, noteUrl, type NoteMeta } from '@/lib/content'

/** Relations to other notes. Sits at the top of the right column, above the margin references. */
export function NoteAside({ note }: { note: NoteMeta }) {
  const listed = [...note.requires, ...note.partOf, ...note.related, ...components(note.slug).map((n) => n.slug)]
  const groups: [string, NoteMeta[]][] = [
    ['Requires', note.requires.map((s) => notesBySlug.get(s)!)],
    ['Part of', note.partOf.map((s) => notesBySlug.get(s)!)],
    ['Components', components(note.slug)],
    ['Related', note.related.map((s) => notesBySlug.get(s)!)],
    // Everything else that points here, excluding notes already shown above.
    ['Referenced by', backlinks(note.slug).filter((n) => !listed.includes(n.slug))],
  ]
  return (
    <div className="space-y-8 text-sm">
      {groups
        .filter(([, list]) => list.length)
        .map(([label, list]) => (
          <section key={label}>
            <h2 className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">{label}</h2>
            <ul className="space-y-1.5">
              {list.map((n) => (
                <li key={n.slug}>
                  <Link to={noteUrl(n.slug)} className="hover:underline">
                    {n.title}
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        ))}
    </div>
  )
}
