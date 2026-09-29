/**
 * Loads content/glossary.yaml for the content index (plugins/content-index.ts) and the MDX first-use pass
 * (plugins/rehype-gloss.ts). The parse is cached by modification time, so the ~1200 note compiles share one read.
 *
 * Checks that need only the file itself live here: the schema, unique keys and aliases, a sense on every entry whose
 * short form is shared, and the long-form conventions. Checks against the taxonomy, notes and macros are in buildIndex.
 */
import fs from 'node:fs'
import YAML from 'yaml'
import { z } from 'zod'
import { glossaryEntrySchema, slug, type GlossaryEntry } from '../site/src/lib/content-schema.ts'

/** Runs of capitals allowed in a long form: Roman numerals, and names whose capitals are not an acronym. */
export const LONG_FORM_CAPITALS = new Set(['II', 'III', 'IV', 'NET', 'QR', 'F1', 'AI'])

export type Glossary = {
  entries: Record<string, GlossaryEntry>
  /** Every key and alias, mapped to its entry's key. */
  names: Map<string, string>
  errors: string[]
}

const FILE = 'content/glossary.yaml'
let cache: { file: string; mtime: number; glossary: Glossary } | undefined

/** The glossary, or an empty one when the file does not exist. Throws with the file named on a schema error. */
export function loadGlossary(file: string): Glossary {
  const mtime = fs.existsSync(file) ? fs.statSync(file).mtimeMs : -1
  if (cache?.file === file && cache.mtime === mtime) return cache.glossary
  // YAML rejects duplicate keys in a mapping, so every key is unique by construction.
  const raw: unknown = mtime < 0 ? {} : (YAML.parse(fs.readFileSync(file, 'utf8')) ?? {})
  const parsed = z.record(slug, glossaryEntrySchema).safeParse(raw)
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `  - ${i.path.join('.') || '(root)'}: ${i.message}`)
    throw new Error(`${FILE}:\n${issues.join('\n')}`)
  }
  const entries: Record<string, GlossaryEntry> = {}
  const names = new Map<string, string>()
  const errors: string[] = []
  for (const [key, value] of Object.entries(parsed.data)) entries[key] = { key, ...value }
  for (const key of Object.keys(entries)) names.set(key, key)
  for (const e of Object.values(entries)) {
    for (const alias of e.aliases) {
      const other = names.get(alias)
      if (other)
        errors.push(
          `${FILE}: ${e.key} alias "${alias}" is already ${other === alias ? 'a key' : `an alias of ${other}`}`,
        )
      else names.set(alias, e.key)
    }
  }

  const byShort = new Map<string, GlossaryEntry[]>()
  for (const e of Object.values(entries)) if (e.short) byShort.set(e.short, [...(byShort.get(e.short) ?? []), e])
  for (const [short, group] of byShort) {
    if (group.length < 2) continue
    const keys = group.map((e) => e.key).join(', ')
    for (const e of group.filter((e) => !e.sense))
      errors.push(`${FILE}: ${e.key} shares "${short}" with ${keys}; add a sense`)
    // A bare `<Gloss name="lda">` must not quietly pick one sense.
    const bare = short
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
    if (names.has(bare)) errors.push(`${FILE}: "${bare}" names ${names.get(bare)}, but "${short}" is shared by ${keys}`)
  }

  for (const e of Object.values(entries)) {
    for (const form of [e.long, e.longPlural].filter((f): f is string => !!f)) {
      for (const run of form.match(/\b[A-Z][A-Z0-9]+\b/g) ?? []) {
        if (!LONG_FORM_CAPITALS.has(run) && !(e.kind === 'proper-name' && e.short?.includes(run)))
          errors.push(`${FILE}: ${e.key} long form "${form}" contains the acronym ${run}; spell it out`)
      }
      // Title Case; a leading one-letter variable such as the k of "k-Nearest Neighbours" stays lowercase.
      if (/^[a-z]/.test(form) && !/^[a-z][-–]/.test(form))
        errors.push(`${FILE}: ${e.key} long form "${form}" should be in Title Case`)
    }
  }

  const glossary = { entries, names, errors }
  cache = { file, mtime, glossary }
  return glossary
}
