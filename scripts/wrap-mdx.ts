/**
 * `make wrap`: rewrap the prose of notes to 120 characters (plugins/wrap-prose.ts). Only paragraphs with a longer line
 * are reflowed, and only at whitespace inside text, so maths, code, JSX tags, tables, headings and frontmatter are
 * untouched. A rewrapped note is written only if it parses to the same tree and renders to the same HTML as before,
 * up to whitespace; otherwise it is left as it was and reported.
 *
 *   node scripts/wrap-mdx.ts [--check] <taxonomy path | slug ...>
 *
 * --check writes nothing and fails when any note has a paragraph that would be rewrapped. Long lines that cannot be
 * broken (a table row, a heading, display maths, a long tag or URL) are not counted.
 */
import fs from 'node:fs'
import path from 'node:path'
import { buildIndex } from '../plugins/content-index.ts'
import { sameRender, WIDTH, wrapProse } from '../plugins/wrap-prose.ts'

const root = path.resolve(import.meta.dirname, '..')
const contentDir = path.join(root, 'content')

const argv = process.argv.slice(2)
const check = argv.includes('--check')
const scopeArgs = argv.filter((a) => !a.startsWith('--')).map((a) => a.replace(/^\/+|\/+$/g, ''))
const inScope = (n: { slug: string; category: string }) =>
  scopeArgs.length === 0 || scopeArgs.some((a) => n.slug === a || n.category === a || n.category.startsWith(`${a}/`))

const { notes: allNotes } = buildIndex(contentDir, { inScope: () => false })
for (const a of scopeArgs)
  if (!allNotes.some((n) => n.slug === a || n.category === a || n.category.startsWith(`${a}/`)))
    throw new Error(`scope "${a}" matches no note slug or taxonomy path`)
const notes = allNotes.filter(inScope)

let changed = 0
let paragraphs = 0
const failures: string[] = []
for (const note of notes) {
  const file = path.join(contentDir, note.file)
  const before = fs.readFileSync(file, 'utf8')
  const result = wrapProse(before)
  if (result.failure) {
    failures.push(`${note.file}: ${result.failure}`)
    continue
  }
  if (result.paragraphs === 0) continue
  if (check) {
    changed++
    paragraphs += result.paragraphs
    console.error(`${note.file}: ${result.paragraphs} paragraph(s) with lines over ${WIDTH} characters`)
    continue
  }
  const renderFailure = await sameRender(before, result.text)
  if (renderFailure) {
    failures.push(`${note.file}: ${renderFailure}`)
    continue
  }
  fs.writeFileSync(file, result.text)
  changed++
  paragraphs += result.paragraphs
}

for (const f of failures) console.error(`not rewrapped: ${f}`)
const verb = check ? 'need rewrapping' : 'rewrapped'
console.error(
  `${notes.length} notes · ${changed} ${verb} (${paragraphs} paragraphs) · ${failures.length} left unchanged`,
)
if (check && (changed > 0 || failures.length > 0)) process.exit(1)
