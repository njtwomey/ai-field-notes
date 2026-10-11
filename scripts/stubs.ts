/**
 * `make stubs`: list notes by status, to find the placeholders still to write.
 *
 * ```bash
 * node scripts/stubs.ts                         # every stub note, grouped by category
 * node scripts/stubs.ts part-3-data/encoding    # only under these category paths (prefix match)
 * node scripts/stubs.ts --status draft          # notes with another status
 * node scripts/stubs.ts --count                 # counts per top-level category only
 * ```
 */
import fs from 'node:fs'
import path from 'node:path'
import { parse } from 'yaml'

const root = path.resolve(import.meta.dirname, '..')
const notesDir = path.join(root, 'content', 'notes')
const args = process.argv.slice(2)
let status = 'stub'
let countOnly = false
const scopes: string[] = []
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--status') status = args[++i]
  else if (args[i] === '--count') countOnly = true
  else scopes.push(args[i].replace(/^content\/notes\//, '').replace(/\/$/, ''))
}

const byCategory = new Map<string, string[]>()
const counts = new Map<string, number>()
let total = 0
for (const f of fs.globSync('**/index.mdx', { cwd: notesDir }).sort()) {
  const dir = path.dirname(f)
  const category = path.dirname(dir)
  if (scopes.length && !scopes.some((s) => dir === s || dir.startsWith(`${s}/`) || dir.split('/').includes(s))) continue
  const fm = parse(fs.readFileSync(path.join(notesDir, f), 'utf8').split('---')[1]) as { status?: string }
  if ((fm.status ?? 'draft') !== status) continue
  total++
  const top = dir.split('/')[0]
  counts.set(top, (counts.get(top) ?? 0) + 1)
  byCategory.set(category, [...(byCategory.get(category) ?? []), path.basename(dir)])
}
if (countOnly) for (const [top, n] of counts) console.log(`${String(n).padStart(5)}  ${top}`)
else
  for (const [category, slugs] of byCategory) {
    console.log(category)
    for (const s of slugs) console.log(`  ${s}`)
  }
console.log(`${total} ${status} note${total === 1 ? '' : 's'}`)
