/**
 * `make doctor`: checks the content tree beyond what the build validates. Errors fail; warnings only report.
 *
 * The build (plugins/content-index.ts) already enforces frontmatter, unique slugs, the leaf rule, known categories,
 * relations and citations. This adds the checks that span content/, python/ and the generated assets, and structural
 * warnings that help keep the taxonomy tidy as notes move.
 */
import fs from 'node:fs'
import path from 'node:path'
import { compile } from '@mdx-js/mdx'
import { buildIndex } from '../plugins/content-index.ts'
import { mdxOptions } from '../plugins/mdx-options.ts'
import type { CategoryNode } from '../site/src/lib/content-schema.ts'

const root = path.resolve(import.meta.dirname, '..')
const contentDir = path.join(root, 'content')
const errors: string[] = []
const warnings: string[] = []

let index: ReturnType<typeof buildIndex>
try {
  index = buildIndex(contentDir)
} catch (err) {
  console.error(err instanceof Error ? err.message : err)
  process.exit(1)
}
const { notes, taxonomy, folders } = index
const slugs = new Set(notes.map((n) => n.slug))

// Every note must compile exactly as the site compiles it; the index check reads frontmatter and maths only.
await Promise.all(
  notes.map(async (n) => {
    const file = path.join(contentDir, n.file)
    try {
      await compile({ path: file, value: fs.readFileSync(file, 'utf8') }, mdxOptions)
    } catch (err) {
      const e = err as { reason?: string; message: string; line?: number; column?: number }
      const where = e.line ? `:${e.line}:${e.column}` : ''
      errors.push(`content/${n.file}${where}: MDX does not compile: ${e.reason ?? e.message}`)
    }
  }),
)

// Block components must open and close on lines of their own. Joined to prose (e.g. by a reflow) they either fail
// to compile or silently render inline.
const blockTag = /<\/?(Definition|Derivation|Callout)\b[^>]*>/g
for (const n of notes) {
  fs.readFileSync(path.join(contentDir, n.file), 'utf8')
    .split('\n')
    .forEach((line, i) => {
      for (const m of line.matchAll(blockTag)) {
        if (line.trim() !== m[0]) errors.push(`content/${n.file}:${i + 1}: put ${m[0]} on a line of its own`)
      }
    })
}

// Sibling files whose names differ only in case (widget.ts next to Widget.tsx) resolve to the wrong file on
// case-insensitive file systems, such as macOS by default.
const byFolder = new Map<string, Map<string, string>>()
for (const rel of fs.globSync('notes/**/*.{ts,tsx,mdx}', { cwd: contentDir })) {
  const dir = path.dirname(rel)
  const stem = path
    .basename(rel)
    .replace(/\.(tsx?|mdx)$/, '')
    .toLowerCase()
  const seen = byFolder.get(dir) ?? new Map<string, string>()
  const other = seen.get(stem)
  if (other && other !== path.basename(rel))
    errors.push(`content/${rel}: clashes with ${other} on a case-insensitive file system; rename one`)
  seen.set(stem, path.basename(rel))
  byFolder.set(dir, seen)
}

// Slug-keyed links into python/ and the generated assets must still resolve after a note moves.
const exampleIds = new Set(
  [
    ...fs.readFileSync(path.join(root, 'python/runs.toml'), 'utf8').matchAll(/^\[\[examples\]\]\s*\nid = "([^"]+)"/gm),
  ].map((m) => m[1]),
)
for (const n of notes) {
  if (n.code && !exampleIds.has(n.code))
    errors.push(`content/${n.file}: code "${n.code}" is not an example in runs.toml`)
}

const sourceFiles = fs
  .globSync('notes/**/*.{ts,tsx,mdx}', { cwd: contentDir })
  .concat(fs.globSync('src/**/*.{ts,tsx}', { cwd: path.join(root, 'site') }).map((f) => `../site/${f}`))
for (const rel of sourceFiles) {
  const file = path.join(contentDir, rel)
  const source = fs.readFileSync(file, 'utf8')
  for (const [, id] of source.matchAll(/useFigure<[^>]*>\(\s*'([^']+)'/g)) {
    const owner = id.split('/')[0]
    if (!slugs.has(owner)) errors.push(`${path.relative(root, file)}: figure "${id}" is not under a note slug`)
    if (!fs.existsSync(path.join(root, 'site/public/generated/figures', `${id}.json`))) {
      errors.push(`${path.relative(root, file)}: figure "${id}" has no generated data (run make assets)`)
    }
  }
  // A module binding named after a browser global shadows it in that file, which breaks code (including the dev
  // server's hot-reload preamble) that relies on the global. The DSP window function is called makeWindow for this.
  for (const [, names] of source.matchAll(/import\s*\{([^}]*)\}\s*from/g)) {
    for (const raw of names.split(',')) {
      const local = raw
        .trim()
        .split(/\s+as\s+/)
        .pop()
        ?.replace(/^type\s+/, '')
      if (local && ['window', 'document', 'self', 'globalThis'].includes(local)) {
        errors.push(`${path.relative(root, file)}: import binding "${local}" shadows the browser global; rename it`)
      }
    }
  }
  // Notes never import from another note: shared code goes in site/src or a `_shared` folder.
  if (rel.startsWith('notes/')) {
    const noteDir = path.dirname(
      path.join(
        contentDir,
        notes.find((n) => file.startsWith(path.join(contentDir, path.dirname(n.file)) + path.sep))?.file ?? rel,
      ),
    )
    for (const [, spec] of source.matchAll(/from '(\.\.?\/[^']+)'/g)) {
      const target = path.resolve(path.dirname(file), spec)
      if (!target.startsWith(noteDir + path.sep) && !target.split(path.sep).some((p) => p.startsWith('_'))) {
        errors.push(`content/${rel}: imports "${spec}" from outside its note; move shared code to site/src or _shared/`)
      }
    }
  }
}

// Layout: legacy flat notes, redundant frontmatter, stray folders.
for (const n of notes) {
  const nested = n.file.split('/').length > 3
  if (!nested) warnings.push(`content/${n.file}: flat layout; move into its category folder (${n.category})`)
}
const categoryPaths = new Set<string>()
const walk = (nodes: CategoryNode[]) => nodes.forEach((c) => (categoryPaths.add(c.path), walk(c.children)))
walk(taxonomy)
for (const folder of folders) {
  if (!categoryPaths.has(folder)) warnings.push(`content/notes/${folder}/: not a taxonomy category and holds no note`)
}

// Structure: empty categories and branches that hold a single note.
const deepCount = (p: string) => notes.filter((n) => n.category === p || n.category.startsWith(`${p}/`)).length
// Empty categories are the planned parts of the encyclopedia, so they are counted, not warned about one by one.
const empty: string[] = []
const visit = (nodes: CategoryNode[]) => {
  for (const c of nodes) {
    const count = deepCount(c.path)
    if (count === 0) empty.push(c.path)
    else if (count === 1 && c.children.length === 0 && c.path.includes('/')) {
      warnings.push(`taxonomy: "${c.path}" holds a single note; consider merging it into its parent`)
    }
    visit(c.children)
  }
}
visit(taxonomy)

// Near-duplicate names: slugs, titles and aliases that normalise to the same key across different notes.
const normalise = (s: string) =>
  s
    .toLowerCase()
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\b(the|a|an|of)\b/g, ' ')
    .replace(/s\b/g, '')
    .replace(/\s+/g, '')
const owners = new Map<string, Set<string>>()
for (const n of notes) {
  for (const name of [n.slug, n.title, ...n.aliases]) {
    const key = normalise(name)
    // Symbols such as Γ or α normalise to nothing; they are not names worth comparing.
    if (key.length < 3) continue
    if (!owners.has(key)) owners.set(key, new Set())
    owners.get(key)!.add(n.slug)
  }
}
for (const [key, set] of owners) {
  if (set.size > 1) warnings.push(`names: "${key}" is shared by ${[...set].join(', ')}`)
}

for (const w of warnings) console.log(`warning  ${w}`)
if (empty.length) console.log(`info     ${empty.length} taxonomy categories have no notes yet (planned)`)
for (const e of errors) console.error(`error    ${e}`)
console.log(`\n${notes.length} notes · ${errors.length} errors · ${warnings.length} warnings`)
process.exit(errors.length ? 1 : 0)
