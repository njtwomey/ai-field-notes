import { ArrowLeft, Hash, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router'
import { NoteCard } from '@/components/browse/NoteCard'
import { TagPill } from '@/components/browse/TagPill'
import { Input } from '@/components/ui/input'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { allTags, notes } from '@/lib/content'

/** Every tag as a compact index: grouped A–Z, or ordered by use. Each tag's page lists its notes. */
export function TagsPage() {
  const [query, setQuery] = useState('')
  const [order, setOrder] = useState<'alpha' | 'count'>('alpha')
  const tags = allTags()

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase()
    const shown = tags.filter((t) => !q || t.tag.includes(q))
    if (order === 'count')
      return [['', [...shown].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag))]] as const
    const byLetter = new Map<string, typeof shown>()
    for (const t of shown) byLetter.set(t.tag[0].toUpperCase(), [...(byLetter.get(t.tag[0].toUpperCase()) ?? []), t])
    return [...byLetter.entries()]
  }, [tags, query, order])

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 lg:px-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-prose text-3xl font-bold">Tags</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {tags.length} tags across {notes.length} notes
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative w-56">
            <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter tags"
              className="pl-8"
            />
          </div>
          <ToggleGroup
            value={[order]}
            onValueChange={(v) => v[0] && setOrder(v[0] as 'alpha' | 'count')}
            variant="outline"
            size="sm"
            spacing={0}
          >
            <ToggleGroupItem value="alpha" className="px-3">
              A–Z
            </ToggleGroupItem>
            <ToggleGroupItem value="count" className="px-3">
              Most used
            </ToggleGroupItem>
          </ToggleGroup>
        </div>
      </div>

      {/* A book-style index: flowing columns, letter headings, one line per tag. */}
      <div className="mt-10 columns-1 gap-10 sm:columns-2 lg:columns-3 xl:columns-4">
        {groups.map(([letter, list]) => (
          <section key={letter || 'all'} className="mb-6 break-inside-avoid">
            {letter && <h2 className="mb-1 font-prose text-lg font-bold text-muted-foreground">{letter}</h2>}
            <ul className="text-sm">
              {list.map(({ tag, count }) => (
                <TagEntry key={tag} tag={tag} count={count} />
              ))}
            </ul>
          </section>
        ))}
      </div>
      {groups.length === 0 && (
        <p className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">No tags match.</p>
      )}
    </main>
  )
}

function TagEntry({ tag, count }: { tag: string; count: number }) {
  return (
    <li className="flex items-center gap-2 py-0.5">
      <TagPill tag={tag} />
      <span className="flex-1 border-b border-dotted border-border" aria-hidden />
      <span className="text-xs text-muted-foreground tabular-nums">{count}</span>
    </li>
  )
}

/** One tag: its notes as cards, and the tags that most often appear alongside it. */
export function TagPage() {
  const { tag = '' } = useParams()
  const tagged = notes.filter((n) => n.tags.includes(tag))
  const cooccurring = useMemo(() => {
    const counts = new Map<string, number>()
    for (const n of tagged) for (const t of n.tags) if (t !== tag) counts.set(t, (counts.get(t) ?? 0) + 1)
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  }, [tagged, tag])

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 lg:px-8">
      <Link to="/tags" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-3.5" aria-hidden /> All tags
      </Link>
      <h1 className="mt-3 flex items-center gap-2 font-prose text-3xl font-bold">
        <Hash className="size-6 text-muted-foreground" aria-hidden />
        {tag}
      </h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {tagged.length} {tagged.length === 1 ? 'note' : 'notes'}
      </p>
      {cooccurring.length > 0 && (
        <div className="mt-5 flex flex-wrap items-center gap-2 text-sm">
          <span className="text-muted-foreground">Often appears with</span>
          {cooccurring.map(([t, n]) => (
            <TagPill key={t} tag={t} count={n} />
          ))}
        </div>
      )}
      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {tagged.map((n) => (
          <NoteCard key={n.slug} note={n} />
        ))}
      </div>
    </main>
  )
}
