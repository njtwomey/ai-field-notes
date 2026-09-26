/**
 * Builds the content index from content/ at dev and build time.
 *
 * - `virtual:content` exposes note metadata, references and the taxonomy. It is small and loaded eagerly.
 * - `virtual:search` exposes plain-text note bodies for the search index. It is loaded when search first opens.
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
  type CategoryInput,
  type CategoryNode,
  type NoteMeta,
} from '../site/src/lib/content-schema.ts'
import { macroExample, macroGroups, macros } from '../content/macros.ts'

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

/** Rough MDX → plain text for search. Keeps prose, drops code, JSX and import/export lines. */
function plainText(body: string): string {
  return body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/^(import|export)\s.*$/gm, ' ')
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
      children: flattenCategories(n.children ?? [], p),
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

export function buildIndex(contentDir: string) {
  const errors: string[] = []

  const taxonomyFile = path.join(contentDir, 'taxonomy.yaml')
  const taxonomyParsed = z.array(categorySchema).safeParse(YAML.parse(fs.readFileSync(taxonomyFile, 'utf8')))
  if (!taxonomyParsed.success) throw new Error(formatIssues('content/taxonomy.yaml', taxonomyParsed.error))
  const taxonomy = flattenCategories(taxonomyParsed.data)
  const categoryPaths = collectPaths(taxonomy)

  const referencesFile = path.join(contentDir, 'references.yaml')
  const referencesParsed = z
    .record(z.string().regex(/^[a-z][a-z0-9-]*$/), referenceSchema)
    .safeParse(YAML.parse(fs.readFileSync(referencesFile, 'utf8')))
  if (!referencesParsed.success) throw new Error(formatIssues('content/references.yaml', referencesParsed.error))
  const references = referencesParsed.data

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
    const { data, body } = splitFrontmatter(source, file)
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
    errors.push(...mathErrors(body).map((e) => `content/${file}: ${e}`))
    const text = plainText(body)
    notes.push({
      ...parsed.data,
      category: resolvedCategory,
      slug: noteSlug,
      file,
      headings: extractHeadings(body),
      cited: [...new Set(attributeValues(body, 'Cite', 'id'))],
      linked: [...new Set(attributeValues(body, 'NoteLink', 'to'))],
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
  if (errors.length) throw new Error(`Content validation failed:\n${errors.join('\n')}`)

  return { notes, references, taxonomy, bodies, folders: found.folders }
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
      const { notes, references, taxonomy, bodies } = buildIndex(contentDir)
      if (id === resolved(SEARCH_ID)) return `export default ${JSON.stringify(bodies)}`
      return [
        `export const notes = ${JSON.stringify(notes)}`,
        `export const references = ${JSON.stringify(references)}`,
        `export const taxonomy = ${JSON.stringify(taxonomy)}`,
      ].join('\n')
    },
  }
}
