import MiniSearch from 'minisearch'
import { notes, type NoteMeta } from '@/lib/content'

type Doc = { id: string; title: string; aliases: string; summary: string; tags: string; headings: string; body: string }

let index: Promise<MiniSearch<Doc>> | undefined

/** Built on first use. Note bodies come from `virtual:search`, a separate chunk loaded only when search opens. */
export function searchIndex(): Promise<MiniSearch<Doc>> {
  index ??= import('virtual:search').then(({ default: bodies }) => {
    const ms = new MiniSearch<Doc>({
      fields: ['title', 'aliases', 'summary', 'tags', 'headings', 'body'],
      storeFields: [],
      searchOptions: {
        boost: { title: 4, aliases: 3, tags: 2, summary: 2, headings: 1.5 },
        prefix: true,
        fuzzy: 0.2,
        combineWith: 'AND',
      },
    })
    ms.addAll(
      notes.map((n) => ({
        id: n.slug,
        title: n.title,
        aliases: n.aliases.join(' '),
        summary: n.summary,
        tags: n.tags.join(' '),
        headings: n.headings.map((h) => h.text).join(' '),
        body: bodies[n.slug] ?? '',
      })),
    )
    return ms
  })
  return index
}

export async function search(query: string, limit = 20): Promise<NoteMeta[]> {
  const ms = await searchIndex()
  const bySlug = new Map(notes.map((n) => [n.slug, n]))
  return ms
    .search(query)
    .slice(0, limit)
    .map((r) => bySlug.get(r.id as string))
    .filter((n): n is NoteMeta => !!n)
}
