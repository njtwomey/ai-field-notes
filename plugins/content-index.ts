/**
 * Builds the content index from content/ at dev and build time.
 *
 * - `virtual:search` exposes plain-text note bodies for full-text search. It is loaded in the background the first
 *   time the search palette opens; the palette answers from note metadata until it is ready.
 * - `virtual:content` exposes note metadata, references, the glossary and the taxonomy. It is loaded eagerly.
 *
 * Frontmatter, taxonomy and references are validated here with zod. Any broken slug, citation key or category fails
 * the build with a message that names the file.
 */
import fs from 'node:fs'
import path from 'node:path'
import GithubSlugger from 'github-slugger'
import type { Plugin, ViteDevServer } from 'vite'
import katex from 'katex'
import YAML from 'yaml'
import { z } from 'zod'
import {
  categorySchema,
  frontmatterSchema,
  referenceSchema,
  slug,
  type CategoryInput,
  type CategoryNode,
  type GlossaryEntry,
  type NoteMeta,
  type TopicGroup,
} from '../site/src/lib/content-schema.ts'
import { macroExample, macroGroups, macros } from '../content/macros.ts'
import { plainMath } from '../site/src/lib/math-text.ts'
import { loadGlossary, type Glossary } from './glossary.ts'

const CONTENT_ID = 'virtual:content'
const SEARCH_ID = 'virtual:search'

/** A note found on disk: its slug (the folder name), its folder relative to notes/, and the category its path implies. */
type NoteDir = { slug: string; dir: string; category: string }

/**
 * Every note under notes/, at any depth. A folder holding index.mdx is a note and is never searched further. Any other
 * folder is a category; folders starting with `_` hold shared code and are skipped.
 * Notes directly under notes/ are the legacy flat layout, whose category comes from frontmatter.
 */
function listNotes(notesDir: string, errors: string[]): { notes: NoteDir[]; folders: string[] } {
  const notes: NoteDir[] = []
  const folders: string[] = []
  const walk = (rel: string) => {
    for (const d of fs.readdirSync(path.join(notesDir, rel), { withFileTypes: true })) {
      if (!d.isDirectory() || d.name.startsWith('_') || d.name.startsWith('.')) continue
      const child = rel ? `${rel}/${d.name}` : d.name
      if (fs.existsSync(path.join(notesDir, child, 'index.mdx'))) {
        notes.push({ slug: d.name, dir: child, category: rel })
        const nested = fs.globSync('**/index.mdx', { cwd: path.join(notesDir, child) }).filter((f) => f !== 'index.mdx')
        for (const f of nested) errors.push(`content/notes/${child}/${f}: a note folder must not contain other notes`)
      } else {
        folders.push(child)
        walk(child)
      }
    }
  }
  if (fs.existsSync(notesDir)) walk('')
  notes.sort((a, b) => a.slug.localeCompare(b.slug))
  return { notes, folders }
}

function splitFrontmatter(source: string, file: string): { data: unknown; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?/.exec(source)
  if (!match) throw new Error(`${file}: missing YAML frontmatter`)
  return { data: YAML.parse(match[1]), body: source.slice(match[0].length) }
}

function extractHeadings(body: string): NoteMeta['headings'] {
  const slugger = new GithubSlugger()
  const headings: NoteMeta['headings'] = []
  let inFence = false
  for (const line of body.split('\n')) {
    if (line.startsWith('```')) inFence = !inFence
    const m = !inFence && /^(#{2,3})\s+(.+?)\s*$/.exec(line)
    if (m) headings.push({ depth: m[1].length, text: m[2], id: slugger.slug(m[2]) })
  }
  return headings
}

function attributeValues(body: string, component: string, attribute: string): string[] {
  const pattern = new RegExp(`<${component}\\b[^>]*?\\b${attribute}=["']([^"']+)["']`, 'g')
  return [...body.matchAll(pattern)].flatMap((m) => m[1].split(',').map((s) => s.trim()))
}

/**
 * Rough MDX → plain text for search. Keeps prose, drops code, JSX and import/export lines. A self-closing <Gloss> is
 * replaced by its entry's short and long forms, so the words it renders are searchable.
 */
function plainText(body: string, gloss: (name: string) => string | undefined): string {
  return body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/^(import|export)\s.*$/gm, ' ')
    .replace(/<Gloss\b[^>]*?\bname=["']([^"']+)["'][^>]*?\/>/g, (_, name: string) => ` ${gloss(name) ?? ''} `)
    .replace(/<[^>]+>/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/[#*_`>[\]()|{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Every $…$ and $$…$$ formula in an MDX body, outside code fences. */
function mathSnippets(body: string): string[] {
  const prose = body.replace(/```[\s\S]*?```/g, ' ')
  const display = [...prose.matchAll(/\$\$([\s\S]+?)\$\$/g)].map((m) => m[1])
  const inline = [...prose.replace(/\$\$[\s\S]+?\$\$/g, ' ').matchAll(/(?<![\\$])\$([^$\n]+?)\$/g)].map((m) => m[1])
  return [...display, ...inline]
}

/** KaTeX errors for a note's maths, rendered with the shared macros. Empty when everything parses. */
function mathErrors(body: string): string[] {
  return mathSnippets(body).flatMap((tex) => {
    try {
      // A fresh copy per formula: KaTeX may add definitions to the object it is given.
      katex.renderToString(tex, { macros: { ...macros }, throwOnError: true, strict: 'ignore' })
      return []
    } catch (err) {
      const message = err instanceof Error ? err.message.replace(/^KaTeX parse error: /, '') : String(err)
      return [`maths ${JSON.stringify(tex.trim().slice(0, 60))}: ${message}`]
    }
  })
}

function flattenCategories(nodes: CategoryInput[], prefix = ''): CategoryNode[] {
  return nodes.map((n) => {
    const p = prefix ? `${prefix}/${n.id}` : n.id
    return {
      path: p,
      title: n.title,
      description: n.description,
      icon: n.icon,
      index: n.index === true ? n.id : n.index || undefined,
      children: flattenCategories(n.children ?? [], p),
      num: n.num,
      roman: n.roman,
    }
  })
}

function collectPaths(nodes: CategoryNode[], into = new Set<string>()): Set<string> {
  for (const n of nodes) {
    into.add(n.path)
    collectPaths(n.children, into)
  }
  return into
}

function formatIssues(file: string, error: z.ZodError): string {
  return `${file}:\n${error.issues.map((i) => `  - ${i.path.join('.') || '(root)'}: ${i.message}`).join('\n')}`
}

/**
 * Build and validate the index. Strict (build, CI, `make check`) throws on any error. Lenient (the dev server) keeps
 * going: it drops whatever is broken, so the rest of the site renders, and returns the errors for the in-app banner.
 */
/** Site totals, computed once at build time for the search palette's empty state. */
function contentStats(
  notes: NoteMeta[],
  references: Record<string, unknown>,
  glossary: Record<string, unknown>,
  taxonomy: CategoryNode[],
) {
  const countCategories = (nodes: CategoryNode[]): number =>
    nodes.reduce((sum, n) => sum + 1 + countCategories(n.children), 0)
  return {
    notes: notes.length,
    topics: taxonomy.length,
    categories: countCategories(taxonomy),
    glossary: Object.keys(glossary).length,
    references: Object.keys(references).length,
    tags: new Set(notes.flatMap((n) => n.tags)).size,
    runnable: notes.filter((n) => n.code).length,
    workedExamples: notes.filter((n) => n.tags.includes('worked-example')).length,
  }
}

/** A name as a slug: lowercase, apostrophes dropped, runs of other characters as one hyphen. */
function nameSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

/**
 * Notes whose slug or title is one of the entry's names (key, long form, aliases, and its short form when no other entry
 * shares it), ignoring a trailing plural "s". Only exact names count: a note that merely lists the term in its aliases
 * is not about it.
 */
function glossNoteMatches(e: GlossaryEntry, notesByName: Map<string, string[]>, glossary: Glossary): string[] {
  const sharedShort = e.short && Object.values(glossary.entries).some((o) => o.key !== e.key && o.short === e.short)
  const names = [e.key, e.long, ...e.aliases, ...(e.short && !sharedShort ? [e.short] : [])].map(nameSlug)
  const forms = new Set(names.flatMap((n) => [n, n.replace(/s$/, ''), `${n}s`]))
  return [...new Set([...forms].flatMap((n) => notesByName.get(n) ?? []))]
}

export function buildIndex(
  contentDir: string,
  {
    strict = true,
    inScope = () => true,
  }: {
    strict?: boolean
    /** Notes outside the scope are indexed but their maths is not rendered (the expensive part); the default is all. */
    inScope?: (note: { slug: string; category: string }) => boolean
  } = {},
) {
  const errors: string[] = []

  const taxonomyFile = path.join(contentDir, 'taxonomy.yaml')
  const taxonomyParsed = z.array(categorySchema).safeParse(YAML.parse(fs.readFileSync(taxonomyFile, 'utf8')))
  if (!taxonomyParsed.success) throw new Error(formatIssues('content/taxonomy.yaml', taxonomyParsed.error))
  const taxonomy = flattenCategories(taxonomyParsed.data)
  const categoryPaths = collectPaths(taxonomy)

  // Home-page groups: every top-level topic in exactly one group, listed in taxonomy order.
  const groupsFile = path.join(contentDir, 'groups.yaml')
  const groupsParsed = z
    .array(
      z
        .object({
          id: z.string().optional(),
          num: z.number().optional(),
          roman: z.string().optional(),
          title: z.string().min(1),
          topics: z.array(slug).min(1),
        })
        .strict(),
    )
    .safeParse(YAML.parse(fs.readFileSync(groupsFile, 'utf8')))
  if (!groupsParsed.success) throw new Error(formatIssues('content/groups.yaml', groupsParsed.error))
  const groups: TopicGroup[] = groupsParsed.data
  const grouped = groups.flatMap((g) => g.topics)
  const topLevel = taxonomy.map((t) => t.path)
  if (grouped.join() !== topLevel.join()) {
    const missing = topLevel.filter((t) => !grouped.includes(t))
    const unknown = grouped.filter((t) => !topLevel.includes(t))
    const twice = grouped.filter((t, i) => grouped.indexOf(t) !== i)
    throw new Error(
      `content/groups.yaml: groups must list every top-level topic once, in taxonomy.yaml order.` +
        (missing.length ? ` Missing: ${missing.join(', ')}.` : '') +
        (unknown.length ? ` Not top-level topics: ${unknown.join(', ')}.` : '') +
        (twice.length ? ` Listed twice: ${twice.join(', ')}.` : '') +
        (!missing.length && !unknown.length && !twice.length ? ' The order differs.' : ''),
    )
  }

  const referencesFile = path.join(contentDir, 'references.yaml')
  const referencesParsed = z
    .record(z.string().regex(/^[a-z][a-z0-9-]*$/), referenceSchema)
    .safeParse(YAML.parse(fs.readFileSync(referencesFile, 'utf8')))
  if (!referencesParsed.success) throw new Error(formatIssues('content/references.yaml', referencesParsed.error))
  const references = referencesParsed.data

  // Parsed once and cached; the MDX first-use pass (plugins/rehype-gloss.ts) reads the same cache.
  const glossary = loadGlossary(path.join(contentDir, 'glossary.yaml'))
  errors.push(...glossary.errors)
  const glossWords = (name: string) => {
    const e = glossary.entries[glossary.names.get(name) ?? '']
    return e && [e.short, e.long].filter(Boolean).join(' ')
  }

  const notesDir = path.join(contentDir, 'notes')
  const notes: NoteMeta[] = []
  const bodies: Record<string, string> = {}
  const found = listNotes(notesDir, errors)
  // Folders that are neither notes nor categories are reported by `make doctor`; a note beneath one fails below with
  // an unknown category.
  const seen = new Map<string, string>()
  for (const { slug: noteSlug, dir, category } of found.notes) {
    const file = `notes/${dir}/index.mdx`
    const clash = seen.get(noteSlug)
    if (clash) {
      errors.push(
        `content/${file}: slug "${noteSlug}" is also used by content/${clash}; slugs are URLs and must be unique`,
      )
      continue
    }
    seen.set(noteSlug, file)
    const source = fs.readFileSync(path.join(contentDir, file), 'utf8')
    let data: unknown
    let body: string
    try {
      ;({ data, body } = splitFrontmatter(source, file))
    } catch (err) {
      // A YAML syntax error in one note must not take down the whole index.
      errors.push(`content/${file}: ${err instanceof Error ? err.message : String(err)}`)
      continue
    }
    const parsed = frontmatterSchema.safeParse(data)
    if (!parsed.success) {
      errors.push(formatIssues(`content/${file}`, parsed.error))
      continue
    }
    // Nested notes take their category from the folder path; flat (legacy) notes from frontmatter.
    if (category && parsed.data.category && parsed.data.category !== category) {
      errors.push(`content/${file}: frontmatter category "${parsed.data.category}" disagrees with folder "${category}"`)
    }
    const resolvedCategory = category || parsed.data.category
    if (!resolvedCategory) {
      errors.push(`content/${file}: no category; move the note into a category folder`)
      continue
    }
    if (inScope({ slug: noteSlug, category: resolvedCategory })) {
      errors.push(...mathErrors(body).map((e) => `content/${file}: ${e}`))
      // Summaries render $…$ maths too (abstract, cards, hover cards); check it, and limit the length as read.
      errors.push(...mathErrors(parsed.data.summary).map((e) => `content/${file}: summary ${e}`))
    }
    const readLength = plainMath(parsed.data.summary).trim().length
    if (readLength > 280) errors.push(`content/${file}: summary is ${readLength} characters as read; the limit is 280`)
    const text = plainText(body, glossWords)
    const glossNames = attributeValues(body, 'Gloss', 'name')
    for (const name of glossNames.filter((n) => !glossary.names.has(n)))
      errors.push(`content/${file}: unknown glossary name "${name}" in <Gloss> (see glossary.yaml)`)
    notes.push({
      ...parsed.data,
      category: resolvedCategory,
      slug: noteSlug,
      file,
      headings: extractHeadings(body),
      cited: [...new Set(attributeValues(body, 'Cite', 'id'))],
      linked: [...new Set(attributeValues(body, 'NoteLink', 'to'))],
      glossed: [...new Set(glossNames.flatMap((n) => glossary.names.get(n) ?? []))],
      wordCount: text.split(' ').length,
    })
    bodies[noteSlug] = text
  }

  for (const group of macroGroups) {
    for (const name of Object.keys(group.macros)) {
      const tex = macroExample(group, name)
      try {
        katex.renderToString(tex, { macros: { ...macros }, throwOnError: true, strict: 'ignore' })
      } catch (err) {
        errors.push(`content/macros.ts: ${name} does not render as ${JSON.stringify(tex)}; add an example. ${err}`)
      }
    }
  }

  const slugs = new Set(notes.map((n) => n.slug))
  for (const n of notes) {
    const where = `content/${n.file}`
    if (!categoryPaths.has(n.category)) errors.push(`${where}: unknown category "${n.category}" (see taxonomy.yaml)`)
    for (const field of ['requires', 'partOf', 'related', 'linked'] as const) {
      for (const target of n[field]) {
        if (!slugs.has(target)) errors.push(`${where}: ${field} → unknown note "${target}"`)
      }
    }
    for (const key of [...n.cited, ...n.references]) {
      if (!(key in references)) errors.push(`${where}: unknown reference "${key}" (see references.yaml)`)
    }
  }
  // A category's index note must exist and live inside that category.
  const byNoteSlug = new Map(notes.map((n) => [n.slug, n]))
  const indexed = new Map<string, string>()
  const checkIndex = (nodes: CategoryNode[]) =>
    nodes.forEach((c) => {
      if (c.index) {
        const other = indexed.get(c.index)
        if (other) errors.push(`content/taxonomy.yaml: "${c.index}" is the index of both ${other} and ${c.path}`)
        indexed.set(c.index, c.path)
        const n = byNoteSlug.get(c.index)
        if (!n) errors.push(`content/taxonomy.yaml: ${c.path} index → unknown note "${c.index}"`)
        else if (n.category !== c.path && !n.category.startsWith(`${c.path}/`))
          errors.push(`content/taxonomy.yaml: ${c.path} index "${c.index}" is in ${n.category}, outside the category`)
      }
      checkIndex(c.children)
    })
  checkIndex(taxonomy)

  // Glossary entries point at real notes and categories, and every string that can hold maths renders.
  const notesByName = new Map<string, string[]>()
  for (const n of notes)
    for (const name of new Set([n.slug, nameSlug(n.title)]))
      notesByName.set(name, [...(notesByName.get(name) ?? []), n.slug])
  for (const e of Object.values(glossary.entries)) {
    const where = `content/glossary.yaml: ${e.key}`
    if (e.note && !slugs.has(e.note)) errors.push(`${where}: note → unknown note "${e.note}"`)
    for (const s of e.see.filter((s) => !slugs.has(s))) errors.push(`${where}: see → unknown note "${s}"`)
    if (e.note && e.see.includes(e.note)) errors.push(`${where}: "${e.note}" is both note and see`)
    // A note whose slug or title names the entry is the entry's note: the mapping is declared, never left implicit.
    const direct = glossNoteMatches(e, notesByName, glossary)
    for (const s of direct.filter((s) => s !== e.note))
      errors.push(`${where}: note "${s}" is titled by this entry; set note: ${s}${e.note ? ` (now ${e.note})` : ''}`)
    for (const c of e.category) if (!categoryPaths.has(c)) errors.push(`${where}: unknown category "${c}"`)
    for (const text of [e.definition, e.short, e.long, e.longPlural]) {
      if (text) errors.push(...mathErrors(text).map((m) => `${where}: ${m}`))
    }
    const readLength = plainMath(e.definition).trim().length
    if (readLength > 400) errors.push(`${where}: definition is ${readLength} characters as read; the limit is 400`)
  }

  if (errors.length && strict) throw new Error(`Content validation failed:\n${errors.join('\n')}`)

  // Lenient: remove dangling references so components never look up something that does not exist.
  const valid = notes.filter((n) => categoryPaths.has(n.category))
  const validSlugs = new Set(valid.map((n) => n.slug))
  const safe = valid.map((n) => ({
    ...n,
    requires: n.requires.filter((s) => validSlugs.has(s)),
    partOf: n.partOf.filter((s) => validSlugs.has(s)),
    related: n.related.filter((s) => validSlugs.has(s)),
    linked: n.linked.filter((s) => validSlugs.has(s)),
    cited: n.cited.filter((k) => k in references),
    references: n.references.filter((k) => k in references),
  }))
  const glossaryEntries = Object.fromEntries(
    Object.entries(glossary.entries).map(([key, e]) => [
      key,
      {
        ...e,
        note: e.note && validSlugs.has(e.note) ? e.note : undefined,
        see: e.see.filter((s) => validSlugs.has(s)),
      },
    ]),
  )
  return {
    notes: safe,
    references,
    glossary: glossaryEntries,
    taxonomy,
    groups,
    bodies,
    folders: found.folders,
    errors,
  }
}

export function contentIndex({ contentDir }: { contentDir: string }): Plugin {
  let server: ViteDevServer | undefined
  const resolved = (id: string) => `\0${id}`

  return {
    name: 'mlc-content-index',
    configureServer(s) {
      server = s
      s.watcher.add(contentDir)
      const refresh = (file: string) => {
        if (!file.startsWith(contentDir) || !/\.(mdx|ya?ml)$/.test(file)) return
        for (const id of [CONTENT_ID, SEARCH_ID]) {
          const mod = server?.moduleGraph.getModuleById(resolved(id))
          if (mod) server?.moduleGraph.invalidateModule(mod)
        }
        server?.ws.send({ type: 'full-reload' })
      }
      s.watcher.on('change', refresh)
      s.watcher.on('add', refresh)
      s.watcher.on('unlink', refresh)
    },
    resolveId(id) {
      if (id === CONTENT_ID || id === SEARCH_ID) return resolved(id)
    },
    load(id) {
      if (id !== resolved(CONTENT_ID) && id !== resolved(SEARCH_ID)) return
      // The dev server stays up on content errors and lists them in the page; builds stay strict.
      const { notes, references, glossary, taxonomy, groups, bodies, errors } = buildIndex(contentDir, {
        strict: !server,
      })
      if (id === resolved(SEARCH_ID)) return `export default ${JSON.stringify(bodies)}`
      return [
        `export const notes = ${JSON.stringify(notes)}`,
        `export const references = ${JSON.stringify(references)}`,
        `export const glossary = ${JSON.stringify(glossary)}`,
        `export const taxonomy = ${JSON.stringify(taxonomy)}`,
        `export const groups = ${JSON.stringify(groups)}`,
        `export const contentErrors = ${JSON.stringify(errors)}`,
        `export const stats = ${JSON.stringify(contentStats(notes, references, glossary, taxonomy))}`,
      ].join('\n')
    },
  }
}
