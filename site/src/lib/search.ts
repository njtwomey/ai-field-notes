import MiniSearch from 'minisearch'
import { notes, type NoteMeta } from '@/lib/content'
import { plainMath } from '@/lib/math-text'

type Doc = { id: string; title: string; aliases: string; summary: string; tags: string; headings: string }

let index: MiniSearch<Doc> | undefined

/**
 * The search index covers each note's title, aliases, summary, tags and section headings, all already loaded as note
 * metadata. Note bodies are not indexed: a well-named note is found by these fields, and indexing the full text of
 * a thousand notes made the palette slow to open. Built on first use, in milliseconds.
 */
function searchIndex(): MiniSearch<Doc> {
  if (!index) {
    index = new MiniSearch<Doc>({
      fields: ['title', 'aliases', 'summary', 'tags', 'headings'],
      storeFields: [],
      searchOptions: {
        boost: { title: 4, aliases: 3, tags: 2, summary: 1.5, headings: 1 },
        prefix: true,
        fuzzy: 0.2,
        combineWith: 'AND',
      },
    })
    index.addAll(
      notes.map((n) => ({
        id: n.slug,
        title: n.title,
        aliases: n.aliases.join(' '),
        summary: plainMath(n.summary),
        tags: n.tags.join(' '),
        headings: n.headings.map((h) => h.text).join(' '),
      })),
    )
  }
  return index
}

/** Every tag with the number of notes that carry it, most used first. */
export const tagCounts: { tag: string; count: number }[] = (() => {
  const counts = new Map<string, number>()
  for (const n of notes) for (const t of n.tags) counts.set(t, (counts.get(t) ?? 0) + 1)
  return [...counts]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag))
})()

export type ParsedQuery = {
  /** Tags written as `#tag`; every one must be present. */
  tags: string[]
  /** The rest of the query, searched as text. */
  text: string
  /** The `#…` token still being typed at the end of the query, if any (for autocomplete). */
  partialTag?: string
}

/** Split a query into `#tag` filters and free text. A trailing `#tag` with no space after it is still being typed. */
export function parseQuery(query: string): ParsedQuery {
  const tokens = query.split(/\s+/).filter(Boolean)
  const typing =
    !/\s$/.test(query) && tokens.at(-1)?.startsWith('#') ? tokens.at(-1)!.slice(1).toLowerCase() : undefined
  const tags: string[] = []
  const words: string[] = []
  tokens.forEach((t, i) => {
    if (t.startsWith('#')) {
      // The tag being typed filters only once it names a real tag exactly.
      const tag = t.slice(1).toLowerCase()
      if (tag && (i < tokens.length - 1 || typing === undefined || tagCounts.some((c) => c.tag === tag))) tags.push(tag)
    } else words.push(t)
  })
  return { tags, text: words.join(' '), partialTag: typing }
}

/** Tags starting with (then containing) the fragment, most used first. */
export function suggestTags(fragment: string, limit = 12): { tag: string; count: number }[] {
  const f = fragment.toLowerCase()
  const starts = tagCounts.filter((c) => c.tag.startsWith(f))
  const contains = tagCounts.filter((c) => !c.tag.startsWith(f) && c.tag.includes(f))
  return [...starts, ...contains].slice(0, limit)
}

/** Notes matching every `#tag` in the query and, if there is text, the text search (ranked); tag-only lists by title. */
export function search(query: string, limit = 30): NoteMeta[] {
  const { tags, text } = parseQuery(query)
  const hasTags = (n: NoteMeta) => tags.every((t) => n.tags.includes(t))
  if (!text.trim()) {
    if (!tags.length) return []
    return notes
      .filter(hasTags)
      .sort((a, b) => a.title.localeCompare(b.title))
      .slice(0, limit)
  }
  const bySlug = new Map(notes.map((n) => [n.slug, n]))
  return searchIndex()
    .search(text)
    .map((r) => bySlug.get(r.id as string))
    .filter((n): n is NoteMeta => !!n && hasTags(n))
    .slice(0, limit)
}
