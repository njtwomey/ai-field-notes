import { notes, type NoteKind, type NoteMeta } from '@/lib/content'

export type Filters = { c?: string; kind?: NoteKind; q?: string }

export const inCategory = (note: NoteMeta, path: string) =>
  note.category === path || note.category.startsWith(`${path}/`)

/** Notes matching every active filter. Text matches title, aliases, tags and summary, case-insensitively. */
export function filterNotes({ c, kind, q }: Filters): NoteMeta[] {
  const words = (q ?? '').toLowerCase().split(/\s+/).filter(Boolean)
  return notes.filter((n) => {
    if (c && !inCategory(n, c)) return false
    if (kind && n.kind !== kind) return false
    if (!words.length) return true
    const haystack = [n.title, ...n.aliases, ...n.tags, n.summary].join(' ').toLowerCase()
    return words.every((w) => haystack.includes(w))
  })
}
