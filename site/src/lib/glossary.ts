/** Lookups and display forms for content/glossary.yaml, shared by <Gloss>, the /glossary page and the search palette. */
import { glossary, notes, type GlossaryEntry, type GlossaryKind, type NoteMeta } from '@/lib/content'

export const glossaryEntries: GlossaryEntry[] = Object.values(glossary)

/** Every key and alias, mapped to its entry. Keys and aliases are unique; the content check enforces it. */
const byName = new Map<string, GlossaryEntry>()
for (const e of glossaryEntries) for (const n of [e.key, ...e.aliases]) byName.set(n, e)

export function glossaryEntry(name: string): GlossaryEntry | undefined {
  return byName.get(name)
}

export const glossaryKindLabels: Record<GlossaryKind, string> = {
  acronym: 'Acronym',
  concept: 'Concept',
  'proper-name': 'Proper name',
}

export function glossaryUrl(key: string): string {
  return `/glossary#${key}`
}

/**
 * How a use of an entry reads, as in the LaTeX acronym packages: `full` is "Long Form (SHORT)" (the first use in a
 * note), `short` the abbreviation, `long` the full name. An entry without a short form always reads as its long form.
 */
export type GlossForm = 'full' | 'short' | 'long'

export function glossText(e: GlossaryEntry, form: GlossForm, plural = false): string {
  const long = plural ? (e.longPlural ?? `${e.long}s`) : e.long
  const short = e.short && (plural ? `${e.short}s` : e.short)
  if (!short || form === 'long') return long
  return form === 'short' ? short : `${long} (${short})`
}

/** The heading an entry is listed under: its short form, or its long form when it has none. */
export function glossHeadword(e: GlossaryEntry): string {
  return e.short ?? e.long
}

const norm = (s: string) =>
  s
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
const names = glossaryEntries.map((e) => ({
  e,
  names: [e.short ?? '', e.key, ...e.aliases, e.long].filter(Boolean).map(norm),
}))

/**
 * Entries for the search palette: an exact match on a short form, key or alias first, then names that start with the
 * query, then names that contain it. Kept to a handful; the notes remain the main results.
 */
export function searchGlossary(query: string, limit = 4): GlossaryEntry[] {
  const q = norm(query)
  if (q.length < 2) return []
  const rank = (ns: string[]) =>
    ns.includes(q) ? 0 : ns.some((n) => n.startsWith(q)) ? 1 : ns.some((n) => n.includes(q)) ? 2 : 3
  return names
    .map((x) => ({ e: x.e, r: rank(x.names) }))
    .filter((x) => x.r < 3)
    .sort((a, b) => a.r - b.r || glossHeadword(a.e).localeCompare(glossHeadword(b.e)))
    .slice(0, limit)
    .map((x) => x.e)
}

const usage = new Map<string, NoteMeta[]>()
for (const n of notes) for (const key of n.glossed) usage.set(key, [...(usage.get(key) ?? []), n])

/** Notes that use an entry with <Gloss>. */
export function glossaryUsage(key: string): NoteMeta[] {
  return usage.get(key) ?? []
}
