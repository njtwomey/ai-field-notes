import { ReferenceCard } from '@/components/content/ReferenceCard'
import { useSidenotes } from '@/components/content/sidenotes'
import { noteReferences, type NoteMeta } from '@/lib/content'
import { cn } from '@/lib/utils'

/** The note's full reference list: inline citations in first-use order, then further reading. Ends every note. */
export function ReferenceList({ note }: { note: NoteMeta }) {
  const refs = noteReferences(note)
  const sidenotes = useSidenotes()
  if (!refs.length) return null
  return (
    <section className="not-prose mt-16 border-t pt-8">
      <h2 id="references" className="mb-4 scroll-mt-32 font-prose text-xl font-semibold">
        References
      </h2>
      <ol className="space-y-3">
        {refs.map(({ key, ref, number }) => (
          <li
            key={key}
            id={`ref-${key}`}
            onMouseEnter={() => sidenotes?.setActive(key)}
            onMouseLeave={() => sidenotes?.setActive(null)}
            className={cn(
              'flex scroll-mt-32 gap-3 rounded-md p-1 transition-colors target:bg-muted',
              sidenotes?.active === key && 'bg-muted',
            )}
          >
            <span className="w-7 shrink-0 pt-0.5 text-xs text-muted-foreground tabular-nums">
              {number ? `[${number}]` : '·'}
            </span>
            <ReferenceCard reference={ref} />
          </li>
        ))}
      </ol>
    </section>
  )
}
