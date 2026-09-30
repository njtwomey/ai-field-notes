/**
 * `make aifn-layers` (part of `make lint`): checks the layer order of `aifn-js/src`. Every module sits in one tier of
 * the table between the `aifn-layers` markers in `aifn-js/README.md`, and may import (values or types) only from
 * modules in strictly lower tiers, through `aifn/<module>`. Also fails on a relative import that reaches into another
 * module's folder, on a module folder missing from the table (or a table entry with no folder), and on a copy of the
 * table in `docs/aifn-plan.md` that differs from the README's.
 */
import fs from 'node:fs'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const srcDir = path.join(root, 'aifn-js', 'src')
const errors: string[] = []

/** The tier table between `<!-- aifn-layers:start -->` and `<!-- aifn-layers:end -->`: rows `| tier | a, b, c |`. */
function readTiers(file: string): Map<string, number> | null {
  const text = fs.readFileSync(path.join(root, file), 'utf8')
  const block = /<!-- aifn-layers:start -->([\s\S]*?)<!-- aifn-layers:end -->/.exec(text)
  if (!block) return null
  const tiers = new Map<string, number>()
  for (const line of block[1].split('\n')) {
    const row = /^\|\s*(\d+)\s*\|([^|]*)\|/.exec(line)
    if (!row) continue
    for (const name of row[2].split(',').map((m) => m.replace(/`/g, '').trim())) {
      if (!name) continue
      if (tiers.has(name)) errors.push(`${file}: module ${name} is listed in two tiers`)
      tiers.set(name, Number(row[1]))
    }
  }
  return tiers
}

const tiers = readTiers('aifn-js/README.md')
if (!tiers) {
  console.error('aifn-layers: no tier table between <!-- aifn-layers:start/end --> markers in aifn-js/README.md')
  process.exit(1)
}

const plan = readTiers('docs/aifn-plan.md')
if (!plan) errors.push('docs/aifn-plan.md: no tier table between <!-- aifn-layers:start/end --> markers')
else {
  for (const [m, t] of tiers)
    if (plan.get(m) !== t) errors.push(`docs/aifn-plan.md: ${m} is in tier ${plan.get(m)}, README says ${t}`)
  for (const m of plan.keys()) if (!tiers.has(m)) errors.push(`docs/aifn-plan.md: ${m} is not in the README's table`)
}

const modules = fs
  .readdirSync(srcDir, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => d.name)
for (const m of modules) if (!tiers.has(m)) errors.push(`aifn-js/src/${m}: module missing from the tier table`)
for (const m of tiers.keys()) if (!modules.includes(m)) errors.push(`tier table: ${m} has no folder in aifn-js/src`)

function* files(dir: string): Generator<string> {
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, d.name)
    if (d.isDirectory()) yield* files(p)
    else if (/\.tsx?$/.test(d.name)) yield p
  }
}

// Static imports and re-exports (`from '…'`), side-effect imports and `import('…')` type queries.
const specifiers = /(?:\bfrom\s*|\bimport\s*\(\s*|^\s*import\s+)['"]([^'"]+)['"]/gm

for (const m of modules) {
  const tier = tiers.get(m)
  for (const file of files(path.join(srcDir, m))) {
    const rel = path.relative(root, file)
    const text = fs.readFileSync(file, 'utf8')
    for (const match of text.matchAll(specifiers)) {
      const spec = match[1]
      const line = text.slice(0, match.index).split('\n').length
      const where = `${rel}:${line}`
      if (spec.startsWith('.')) {
        const target = path.relative(srcDir, path.resolve(path.dirname(file), spec)).split(path.sep)[0]
        if (target !== m)
          errors.push(`${where}: relative import '${spec}' reaches into ${target}; use 'aifn/${target}'`)
        continue
      }
      const dep = /^aifn\/([^/]+)$/.exec(spec)?.[1]
      if (!dep || dep === m) continue
      const depTier = tiers.get(dep)
      if (depTier === undefined) errors.push(`${where}: imports unknown module aifn/${dep}`)
      else if (tier !== undefined && depTier >= tier)
        errors.push(
          `${where}: ${m} (tier ${tier}) imports aifn/${dep} (tier ${depTier}); only lower tiers may be imported`,
        )
    }
  }
}

if (errors.length) {
  for (const e of errors) console.error(`✗ ${e}`)
  console.error(`aifn-layers: ${errors.length} error${errors.length === 1 ? '' : 's'}`)
  process.exit(1)
}
console.log(`aifn-layers: ${modules.length} modules, every import goes down a tier`)
