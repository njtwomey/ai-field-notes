/**
 * One-off: move flat notes (content/notes/<slug>/) into their category folders (content/notes/<category>/<slug>/) and
 * drop the now-redundant `category:` frontmatter line. Optionally renames category prefixes on the way.
 *
 *   node scripts/migrate-notes.ts                          # dry run: print the moves
 *   node scripts/migrate-notes.ts --rename foundations=maths --apply
 *
 * Renaming a category in taxonomy.yaml is done by hand, before --apply. URLs are unaffected: they use the slug only.
 */
import fs from 'node:fs'
import path from 'node:path'

const notesDir = path.resolve(import.meta.dirname, '..', 'content', 'notes')
const apply = process.argv.includes('--apply')
const renames = process.argv
  .flatMap((a, i, all) => (a === '--rename' ? [all[i + 1]] : []))
  .map((r) => r.split('=') as [string, string])

const rename = (category: string) => {
  for (const [from, to] of renames) {
    if (category === from || category.startsWith(`${from}/`)) return to + category.slice(from.length)
  }
  return category
}

let moves = 0
for (const d of fs.readdirSync(notesDir, { withFileTypes: true })) {
  const file = path.join(notesDir, d.name, 'index.mdx')
  if (!d.isDirectory() || !fs.existsSync(file)) continue
  const source = fs.readFileSync(file, 'utf8')
  const match = /^category:\s*(\S+)\s*$/m.exec(source.split('\n---')[0])
  if (!match) throw new Error(`${file}: flat note without a category`)
  const target = path.join(notesDir, rename(match[1]), d.name)
  console.log(`${d.name}  →  notes/${path.relative(notesDir, target)}/`)
  moves++
  if (!apply) continue
  if (fs.existsSync(target)) throw new Error(`${target} already exists`)
  fs.mkdirSync(path.dirname(target), { recursive: true })
  fs.renameSync(path.join(notesDir, d.name), target)
  const index = path.join(target, 'index.mdx')
  fs.writeFileSync(index, fs.readFileSync(index, 'utf8').replace(/^category:.*\n/m, ''))
}
console.log(`\n${moves} notes ${apply ? 'moved' : 'to move (dry run; pass --apply)'}`)
