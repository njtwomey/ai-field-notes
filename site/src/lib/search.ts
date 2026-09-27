import MiniSearch from 'minisearch'
import {
  category,
  categoryTrail,
  notes,
  notesInCategory,
  taxonomy,
  type CategoryNode,
  type NoteMeta,
} from '@/lib/content'
import { plainMath } from '@/lib/math-text'

type Doc = { id: string; title: string; aliases: string; summary: string; tags: string; headings: string }

let index: MiniSearch<Doc> | undefined

/**
 * Two indexes. The metadata index covers title, aliases, summary, tags and headings, which are already loaded, so it
 * answers at once. The body index covers the full text; its data is a separate chunk fetched the first time the
 * palette opens and indexed in small asynchronous chunks, so the palette never waits for it. Body-only matches are
 * appended below the metadata matches once it is ready.
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

let bodyIndex: MiniSearch<{ id: string; body: string }> | undefined
let bodyLoading: Promise<void> | undefined

/** Fetch and index the note bodies in the background; resolves when full-text search is available. */
export function loadBodyIndex(): Promise<void> {
  bodyLoading ??= import('virtual:search').then(async ({ default: bodies }) => {
    const ms = new MiniSearch<{ id: string; body: string }>({
      fields: ['body'],
      storeFields: [],
      searchOptions: { prefix: true, fuzzy: 0.1, combineWith: 'AND' },
    })
    await ms.addAllAsync(
      Object.entries(bodies).map(([id, body]) => ({ id, body })),
      { chunkSize: 25 },
    )
    bodyIndex = ms
  })
  return bodyLoading
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
  /** A taxonomy branch written as `/path/to/branch`; results are restricted to it. */
  path?: string
  /** The `/…` token still being typed at the end of the query, if any (for autocomplete). */
  partialPath?: string
  /** The rest of the query, searched as text. */
  text: string
  /** The `#…` token still being typed at the end of the query, if any (for autocomplete). */
  partialTag?: string
}

/** Split a query into `#tag` filters and free text. A trailing `#tag` with no space after it is still being typed. */
export function parseQuery(query: string): ParsedQuery {
  const tokens = query.split(/\s+/).filter(Boolean)
  const last = tokens.at(-1)
  const open = !/\s$/.test(query)
  const typingPath = open && last?.startsWith('/') ? last.slice(1) : undefined
  const typing =
    !/\s$/.test(query) && tokens.at(-1)?.startsWith('#') ? tokens.at(-1)!.slice(1).toLowerCase() : undefined
  const tags: string[] = []
  const words: string[] = []
  let path: string | undefined
  tokens.forEach((t, i) => {
    if (t.startsWith('/')) {
      // A path filters once it names a real category; while being typed it is only a prefix.
      const p = t.slice(1).replace(/\/+$/, '')
      if (p && category(p) && (i < tokens.length - 1 || typingPath === undefined || category(p))) path = p
    } else if (t.startsWith('#')) {
      // The tag being typed filters only once it names a real tag exactly.
      const tag = t.slice(1).toLowerCase()
      if (tag && (i < tokens.length - 1 || typing === undefined || tagCounts.some((c) => c.tag === tag))) tags.push(tag)
    } else words.push(t)
  })
  return { tags, path, text: words.join(' '), partialTag: typing, partialPath: typingPath }
}

const ALL_CATEGORIES: CategoryNode[] = (() => {
  const out: CategoryNode[] = []
  const walk = (ns: CategoryNode[]) =>
    ns.forEach((n) => {
      out.push(n)
      walk(n.children)
    })
  walk(taxonomy)
  return out
})()

export type PathSuggestion = { path: string; trail: string; count: number; hasChildren: boolean }

/**
 * Taxonomy paths for a `/…` fragment. A fragment ending in `/` (or empty) lists that node's children; otherwise paths
 * whose id path starts with the fragment come first, then any whose title trail contains its last word.
 */
export function suggestPaths(fragment: string, limit = 14): PathSuggestion[] {
  const f = fragment.toLowerCase()
  const describe = (n: CategoryNode): PathSuggestion => ({
    path: n.path,
    trail: categoryTrail(n.path)
      .map((c) => c.title)
      .join(' › '),
    count: notesInCategory(n.path).length,
    hasChildren: n.children.length > 0,
  })
  if (f === '' || f.endsWith('/')) {
    const parent = f.replace(/\/+$/, '')
    const children = parent ? (category(parent)?.children ?? []) : taxonomy
    return children.map(describe).slice(0, limit)
  }
  const starts = ALL_CATEGORIES.filter((c) => c.path.startsWith(f))
  const word = f.split('/').pop() ?? f
  const titled = ALL_CATEGORIES.filter(
    (c) => !c.path.startsWith(f) && (c.title.toLowerCase().includes(word) || c.path.split('/').pop()!.includes(word)),
  )
  return [...starts, ...titled].slice(0, limit).map(describe)
}

/** Tags starting with (then containing) the fragment, most used first. */
export function suggestTags(fragment: string, limit = 12): { tag: string; count: number }[] {
  const f = fragment.toLowerCase()
  const starts = tagCounts.filter((c) => c.tag.startsWith(f))
  const contains = tagCounts.filter((c) => !c.tag.startsWith(f) && c.tag.includes(f))
  return [...starts, ...contains].slice(0, limit)
}

/**
 * Notes matching every `#tag` in the query and, if there is text, the text search (ranked); tag-only lists by title.
 * With `fullText`, body-only matches from the background index are appended once it is ready.
 */
export function search(
  query: string,
  { limit = 30, fullText = false }: { limit?: number; fullText?: boolean } = {},
): NoteMeta[] {
  const { tags, path, text } = parseQuery(query)
  const hasTags = (n: NoteMeta) =>
    tags.every((t) => n.tags.includes(t)) && (!path || n.category === path || n.category.startsWith(`${path}/`))
  if (!text.trim()) {
    if (!tags.length && !path) return []
    return notes
      .filter(hasTags)
      .sort((a, b) => a.title.localeCompare(b.title))
      .slice(0, limit)
  }
  const bySlug = new Map(notes.map((n) => [n.slug, n]))
  const found = searchIndex()
    .search(text)
    .map((r) => r.id as string)
  const seen = new Set(found)
  // Full-text matches the metadata missed, ranked after every metadata match.
  const extra =
    fullText && bodyIndex
      ? bodyIndex
          .search(text)
          .map((r) => r.id as string)
          .filter((id) => !seen.has(id))
      : []
  return [...found, ...extra]
    .map((id) => bySlug.get(id))
    .filter((n): n is NoteMeta => !!n && hasTags(n))
    .slice(0, limit)
}
