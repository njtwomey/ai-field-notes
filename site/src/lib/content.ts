import type { ComponentType } from 'react'
import { contentErrors, glossary, groups, notes, references, stats, taxonomy } from 'virtual:content'
import type { CategoryNode, GlossaryEntry, GlossaryKind, NoteKind, NoteMeta, Reference } from '@/lib/content-schema'

export { contentErrors, glossary, groups, notes, references, stats, taxonomy }
export type { CategoryNode, GlossaryEntry, GlossaryKind, NoteKind, NoteMeta, Reference }

export const notesBySlug: ReadonlyMap<string, NoteMeta> = new Map(notes.map((n) => [n.slug, n]))

type MdxModule = { default: ComponentType<{ components?: Record<string, unknown> }> }
const loaders = import.meta.glob<MdxModule>('@content/notes/**/index.mdx')

export function loadNote(slug: string): Promise<MdxModule> {
  // Notes live at any depth, so find the loader by the note's recorded path (content/<file>).
  const file = notesBySlug.get(slug)?.file
  const key = file && Object.keys(loaders).find((k) => k.endsWith(`/${file}`))
  if (!key) return Promise.reject(new Error(`no note "${slug}"`))
  return loaders[key]()
}

const prefetched = new Set<string>()

/**
 * Start downloading a note's code without rendering it, e.g. when its link is hovered, so opening it is instant.
 * Repeated calls are free; failures are ignored (the real navigation reports them).
 */
export function prefetchNote(slug: string): void {
  if (prefetched.has(slug) || !notesBySlug.has(slug)) return
  prefetched.add(slug)
  loadNote(slug).catch(() => prefetched.delete(slug))
}

/** Prefetch several notes when the browser is idle, a few at a time, so they never compete with the current page. */
export function prefetchNotesWhenIdle(slugs: string[]): () => void {
  const queue = slugs.filter((s) => !prefetched.has(s))
  let cancelled = false
  const idle = (cb: () => void) =>
    typeof window.requestIdleCallback === 'function'
      ? window.requestIdleCallback(cb, { timeout: 3000 })
      : globalThis.setTimeout(cb, 500)
  const step = () => {
    if (cancelled) return
    queue.splice(0, 3).forEach(prefetchNote)
    if (queue.length) idle(step)
  }
  idle(step)
  return () => {
    cancelled = true
  }
}

export const kindLabels: Record<NoteKind, string> = {
  concept: 'Concept',
  distribution: 'Distribution',
  technique: 'Technique',
  test: 'Statistical test',
  example: 'Worked example',
  'case-study': 'Case study',
  overview: 'Overview',
}

export function noteUrl(slug: string, tab?: 'code' | 'outputs', sub?: string): string {
  return `/n/${slug}${tab ? `/${tab}` : ''}${sub ? `/${sub}` : ''}`
}

/** Notes that name `slug` in their requires/partOf/related fields or link to it inline. */
export function backlinks(slug: string): NoteMeta[] {
  return notes.filter((n) => n.slug !== slug && [...n.requires, ...n.partOf, ...n.related, ...n.linked].includes(slug))
}

/** Notes that declare `partOf: [slug]`. Overview notes list these as their components. */
export function components(slug: string): NoteMeta[] {
  return notes.filter((n) => n.partOf.includes(slug))
}

const categoryIndex = new Map<string, CategoryNode>()
const categoryRank = new Map<string, number>()
const walk = (nodes: CategoryNode[]) =>
  nodes.forEach((n) => {
    categoryIndex.set(n.path, n)
    categoryRank.set(n.path, categoryRank.size)
    walk(n.children)
  })
walk(taxonomy)

/** Position of a category in taxonomy.yaml order (depth-first). Sort categories by this, never alphabetically. */
export function categoryOrder(path: string): number {
  return categoryRank.get(path) ?? Number.MAX_SAFE_INTEGER
}

export function category(path: string): CategoryNode | undefined {
  return categoryIndex.get(path)
}

/** Category nodes from root to `path`, for breadcrumbs. */
export function categoryTrail(path: string): CategoryNode[] {
  const parts = path.split('/')
  return parts.map((_, i) => categoryIndex.get(parts.slice(0, i + 1).join('/'))).filter((c) => !!c)
}

/** Category path each index note introduces (taxonomy.yaml `index`). */
const indexedCategory = new Map<string, string>()
for (const c of categoryIndex.values()) if (c.index) indexedCategory.set(c.index, c.path)

/** The note that introduces a category, if taxonomy.yaml names one. */
export function categoryIndexNote(path: string): NoteMeta | undefined {
  const slug = categoryIndex.get(path)?.index
  return slug ? notesBySlug.get(slug) : undefined
}

/** The category a note introduces, if it is that category's index note. */
export function indexOfCategory(slug: string): string | undefined {
  return indexedCategory.get(slug)
}

/** Notes in the order a list shows them: the category's index note first, the rest in their existing order. */
export function indexFirst(list: NoteMeta[], path: string): NoteMeta[] {
  const lead = category(path)?.index
  return lead ? [...list.filter((n) => n.slug === lead), ...list.filter((n) => n.slug !== lead)] : list
}

export function notesInCategory(path: string, deep = true): NoteMeta[] {
  return notes.filter((n) => n.category === path || (deep && n.category.startsWith(`${path}/`)))
}

export function allTags(): { tag: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const n of notes) for (const t of n.tags) counts.set(t, (counts.get(t) ?? 0) + 1)
  return [...counts].map(([tag, count]) => ({ tag, count })).sort((a, b) => a.tag.localeCompare(b.tag))
}

/** Reference keys for a note: inline citations first (numbered in order), then further reading. */
export function noteReferences(note: NoteMeta): { key: string; ref: Reference; number?: number }[] {
  const cited = note.cited.map((key, i) => ({ key, ref: references[key], number: i + 1 }))
  const extra = note.references.filter((k) => !note.cited.includes(k)).map((key) => ({ key, ref: references[key] }))
  return [...cited, ...extra]
}

export function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return ''
  if (authors.length <= 2) return authors.join(' & ')
  return `${authors[0]} et al.`
}

/** The top-level category (topic) a note belongs to. */
export function topicOf(note: NoteMeta): CategoryNode {
  return taxonomy.find((t) => note.category === t.path || note.category.startsWith(`${t.path}/`))!
}

export type BrowseParams = { c?: string; kind?: NoteKind; q?: string; view?: 'list' | 'map' }

/** Link to the browse page with filters. Empty values are dropped. */
export function browseUrl(params: BrowseParams = {}): string {
  const search = new URLSearchParams(Object.entries(params).filter(([, v]) => v) as [string, string][])
  const query = search.toString()
  return `/browse${query ? `?${query}` : ''}`
}
