/**
 * The reading order of the notes, from content/reading-order.yaml: categories in taxonomy.yaml order (depth first, a
 * category's own notes before its subcategories), and inside each category the notes in the order the file lists them.
 * The previous / next arrows on a note and every list of notes follow it.
 */
import fs from 'node:fs'
import YAML from 'yaml'
import { z } from 'zod'
import type { CategoryNode } from '../site/src/lib/content-schema.ts'

const FILE = 'content/reading-order.yaml'

type OrderedNote = { slug: string; category: string }

export function readingOrder(
  file: string,
  taxonomy: CategoryNode[],
  notes: OrderedNote[],
): {
  /** Every note's slug, in reading order. */
  order: string[]
  /** Notes the file does not list; each is read at the end of its category. */
  unlisted: string[]
  errors: string[]
} {
  const errors: string[] = []
  const parsed = fs.existsSync(file)
    ? z.record(z.string(), z.array(z.string()).nullable()).safeParse(YAML.parse(fs.readFileSync(file, 'utf8')) ?? {})
    : { success: true as const, data: {} }
  if (!parsed.success) return { order: notes.map((n) => n.slug), unlisted: [], errors: [`${FILE}: ${parsed.error}`] }

  const byPath = new Map<string, CategoryNode>()
  const walk = (nodes: CategoryNode[]) =>
    nodes.forEach((c) => {
      byPath.set(c.path, c)
      walk(c.children)
    })
  walk(taxonomy)
  const bySlug = new Map(notes.map((n) => [n.slug, n]))
  // An index note is read first in the category it introduces, which may be an ancestor of its own folder.
  const introduces = new Map([...byPath.values()].filter((c) => c.index).map((c) => [c.index!, c.path]))
  const home = (n: OrderedNote) => introduces.get(n.slug) ?? n.category

  const listed = new Map<string, string[]>()
  const seen = new Set<string>()
  for (const [category, slugs] of Object.entries(parsed.data)) {
    if (!byPath.has(category)) {
      errors.push(`${FILE}: "${category}" is not a taxonomy category`)
      continue
    }
    const kept: string[] = []
    for (const slug of slugs ?? []) {
      const n = bySlug.get(slug)
      if (!n) errors.push(`${FILE}: ${category} lists unknown note "${slug}"`)
      else if (seen.has(slug)) errors.push(`${FILE}: "${slug}" is listed twice`)
      else if (home(n) !== category)
        errors.push(`${FILE}: "${slug}" is listed under ${category} but belongs to ${home(n)}`)
      else kept.push(slug)
      seen.add(slug)
    }
    listed.set(category, kept)
  }

  const unlisted = notes.filter((n) => !seen.has(n.slug) && byPath.has(home(n)))
  const order: string[] = []
  const visit = (nodes: CategoryNode[]) =>
    nodes.forEach((c) => {
      const extra = unlisted.filter((n) => home(n) === c.path).map((n) => n.slug)
      // An unlisted index note still leads its category.
      const lead = extra.filter((s) => s === c.index)
      order.push(...lead, ...(listed.get(c.path) ?? []), ...extra.filter((s) => s !== c.index).sort())
      visit(c.children)
    })
  visit(taxonomy)
  return { order, unlisted: unlisted.map((n) => n.slug), errors }
}
