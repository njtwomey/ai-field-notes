import { notes, taxonomy, type NoteKind, type NoteMeta } from '@/lib/content'
import { plainMath } from '@/lib/math-text'

export type Filters = { c?: string; part?: string; kind?: NoteKind; q?: string }

export const inCategory = (note: NoteMeta, path: string) =>
  note.category === path || note.category.startsWith(`${path}/`)

/** Notes matching every active filter. Text matches title, aliases, tags and summary, case-insensitively. */
export function filterNotes({ c, part, kind, q }: Filters): NoteMeta[] {
  const words = (q ?? '').toLowerCase().split(/\s+/).filter(Boolean)
  const targetPart = part
    ? taxonomy.find(
        (p) =>
          p.path === part ||
          String(p.num) === part ||
          `part-${p.num}` === part ||
          p.title.toLowerCase() === part.toLowerCase(),
      )
    : undefined
  return notes.filter((n) => {
    if (c && !inCategory(n, c)) return false
    if (targetPart && !inCategory(n, targetPart.path)) return false
    if (kind && n.kind !== kind) return false
    if (!words.length) return true
    const haystack = [n.title, ...n.aliases, ...n.tags, plainMath(n.summary)].join(' ').toLowerCase()
    return words.every((w) => haystack.includes(w))
  })
}
