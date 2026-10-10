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

/**
 * How much a match in each field counts. MiniSearch scores every field by BM25 and adds the fields up, each times its
 * weight, so one word in a title counts five times as much as the same word in the body and the note's total decides
 * its rank. Aliases are other names for the title; `compact` holds the title and aliases run together (see below).
 */
export const SEARCH_WEIGHTS = {
  title: 5,
  aliases: 4,
  compact: 4,
  summary: 2,
  tags: 2,
  headings: 1.5,
  body: 1,
} as const

/** Search options shared by both indexes: prefix matching, and typos up to this fraction of a term's length. */
const SEARCH_OPTIONS = { boost: SEARCH_WEIGHTS, prefix: true, fuzzy: 0.2, combineWith: 'AND' } as const

type Field = keyof typeof SEARCH_WEIGHTS
type Doc = { id: string } & Partial<Record<Field, string>>

const META_FIELDS: Field[] = ['title', 'aliases', 'compact', 'summary', 'tags', 'headings']

/**
 * Titles and aliases that contain punctuation or spaces, each run together into one word ("VQ-VAE" → "vqvae",
 * "k-means" → "kmeans"), so that a query typed without the separators still finds them. The tokenizer splits on
 * punctuation, so "VQ-VAE" itself is indexed as "vq" and "vae", and "vqvae" matches neither.
 */
function compactNames(names: string[]): string {
  return names
    .filter((s) => /[^\p{L}\p{N}]/u.test(s.trim()))
    .map((s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ''))
    .join(' ')
}

const metaDocs = (): Doc[] =>
  notes.map((n) => ({
    id: n.slug,
    title: n.title,
    aliases: n.aliases.join(' '),
    compact: compactNames([n.title, ...n.aliases]),
    summary: plainMath(n.summary),
    tags: n.tags.join(' '),
    headings: n.headings.map((h) => h.text).join(' '),
  }))

let index: MiniSearch<Doc> | undefined
let fullIndex: MiniSearch<Doc> | undefined
let fullLoading: Promise<void> | undefined

/**
 * Two indexes with the same weights. The metadata index covers every field but the body; its data is already loaded,
 * so it answers at once. The full index adds the body; the bodies are a separate chunk fetched the first time the
 * palette opens and indexed in small asynchronous chunks, so the palette never waits for it. Once it is ready it
 * replaces the metadata index, and a body match counts towards the same total as a title or summary match.
 */
function searchIndex(): MiniSearch<Doc> {
  index ??= (() => {
    const ms = new MiniSearch<Doc>({ fields: META_FIELDS, storeFields: [], searchOptions: SEARCH_OPTIONS })
    ms.addAll(metaDocs())
    return ms
  })()
  return index
}

/** Fetch the note bodies and build the full index in the background; resolves when full-text search is available. */
export function loadBodyIndex(): Promise<void> {
  fullLoading ??= import('virtual:search').then(async ({ default: bodies }) => {
    const ms = new MiniSearch<Doc>({ fields: [...META_FIELDS, 'body'], storeFields: [], searchOptions: SEARCH_OPTIONS })
    await ms.addAllAsync(
      metaDocs().map((d) => ({ ...d, body: bodies[d.id] ?? '' })),
      { chunkSize: 25 },
    )
    fullIndex = ms
  })
  return fullLoading
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
 * With `fullText`, the note bodies are searched too once the background index is ready, weighted by `SEARCH_WEIGHTS`.
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
  const found = (fullText && fullIndex ? fullIndex : searchIndex()).search(text).map((r) => r.id as string)
  // A note whose title or an alias is exactly the query leads, whatever its score: "svm" opens with the SVM note.
  const key = exactKey(text)
  const exact = (n: NoteMeta) => exactKey(n.title) === key || n.aliases.some((a) => exactKey(a) === key)
  const ranked = found.map((id) => bySlug.get(id)).filter((n): n is NoteMeta => !!n && hasTags(n))
  return [...ranked.filter(exact), ...ranked.filter((n) => !exact(n))].slice(0, limit)
}

/** Lowercase, with punctuation and spacing folded, for exact title and alias comparison. */
function exactKey(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
}
