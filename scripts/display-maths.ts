/**
 * `make maths`: put the `$$` fences of display maths on lines of their own. remark-math renders `$$…$$` as display
 * maths only when each fence stands alone on its line; `$$f(x) = …$$` on one line, or `$$\begin{aligned}` with the
 * maths after the fence, renders as inline maths (limits beside the sum, small fractions).
 *
 *   node scripts/display-maths.ts [--check] <taxonomy path | slug ...>
 *
 * Each fence is moved to its own line with the indentation and blockquote prefix of the line it was on, so display
 * maths inside a list item or a quote stays there. Code fences are skipped. --check writes nothing and fails when any
 * note would change.
 */
import fs from 'node:fs'
import path from 'node:path'
import { buildIndex } from '../plugins/content-index.ts'

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

/** Split each line holding a `$$` fence next to other text so that every fence stands alone. */
export function fenceDisplayMaths(source: string): string {
  const out: string[] = []
  let code = false
  let maths = false
  for (const line of source.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) code = !code
    const [, prefix, rest] = /^((?:\s*>)*\s*)(.*)$/.exec(line)!
    const fences = rest.split('$$').length - 1
    if (code || fences === 0 || rest.trim() === '$$') {
      if (!code && rest.trim() === '$$') maths = !maths
      out.push(line)
    } else if (!maths && rest.startsWith('$$') && fences === 2 && rest.trimEnd().endsWith('$$')) {
      // `$$maths$$` on one line.
      out.push(prefix + '$$', prefix + rest.trim().slice(2, -2).trim(), prefix + '$$')
    } else if (!maths && rest.startsWith('$$') && fences === 1) {
      // `$$\begin{aligned}`: an opening fence with maths after it.
      out.push(prefix + '$$', prefix + rest.slice(2).trim())
      maths = true
    } else if (maths && rest.trimEnd().endsWith('$$') && fences === 1) {
      // `\end{aligned}$$`: maths before the closing fence.
      out.push(prefix + rest.trimEnd().slice(0, -2).trimEnd(), prefix + '$$')
      maths = false
    } else {
      out.push(line) // a fence mid-sentence: left for a person
    }
  }
  return out.join('\n')
}

let changed = 0
for (const n of allNotes.filter(inScope)) {
  const file = path.join(contentDir, n.file)
  const before = fs.readFileSync(file, 'utf8')
  const after = fenceDisplayMaths(before)
  if (after === before) continue
  changed++
  console.log(`${check ? 'needs fixing' : 'fixed'}  ${path.relative(root, file)}`)
  if (!check) fs.writeFileSync(file, after)
}
console.log(`${changed} note(s) ${check ? 'with display maths on a line with other text' : 'changed'}`)
if (check && changed) process.exit(1)
