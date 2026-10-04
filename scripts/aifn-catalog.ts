/**
 * `make catalog`: collect every registry entry of aifn-js (both packages) into `aifn-js/generated/catalog.json`, the
 * one list that the site's content check, the lab's reference pages and pickers, aifn-py's fixture coverage and the
 * docs read (design S §3.2). `make check` runs it with `--check`.
 *
 * - **Collection.** Every module of `aifn` and `aifn-methods` (each directory with an `index.ts`) is loaded through
 *   Vite, so its registries are built exactly as an importer builds them. An entry is any exported value carrying
 *   `info` with a `kind` (`define(info, value)`), found directly among a module's exports or inside an exported table
 *   of entries (`windowRegistry`, `firstOrderAlgorithms`, …). Entries are deduplicated by identity, so a value
 *   re-exported by a family index counts once. Primitives come from the primitive table of `aifn/foundation/tensor`.
 * - **Address.** `<info.module>/<info.key>` within a kind (`optim/first-order/adam`); two different entries with one
 *   address fail.
 * - **Checks** (always): every `info.module` is a module of its package; every `notes` slug is a note in
 *   `content/notes`; every `glossary` key is in `content/glossary.yaml`, and when that glossary entry names a note,
 *   the registry entry lists the same note; every `cite` key is in `content/references.yaml`. With `--check`, the
 *   committed catalog must equal the one generated.
 * - **Fixture coverage** (report only): stable entries without a reference case in the Python-generated fixtures
 *   (`aifn-js/*\/test/fixtures/**\/*.json`), matched by key.
 *
 * ```bash
 * node scripts/aifn-catalog.ts           # write the catalog
 * node scripts/aifn-catalog.ts --check   # fail when it is stale or a link is broken
 * ```
 */
import fs from 'node:fs'
import path from 'node:path'
import { createServer } from 'vite'
import { parse } from 'yaml'

const root = path.resolve(import.meta.dirname, '..')
const out = path.join(root, 'aifn-js', 'generated', 'catalog.json')
const check = process.argv.includes('--check')

type Info = {
  key: string
  kind: string
  module: string
  name: string
  stability: string
  notes?: readonly string[]
  glossary?: string
  cite?: readonly string[]
  [field: string]: unknown
}
type Entry = { info: Info }
type Pkg = { name: 'aifn' | 'aifn-methods'; src: string }

const packages: Pkg[] = [
  { name: 'aifn', src: path.join(root, 'aifn-js', 'core', 'src') },
  { name: 'aifn-methods', src: path.join(root, 'aifn-js', 'methods', 'src') },
]

/** Module paths of a package: every directory under src holding an index.ts, relative to src, parents first. */
function modulesOf(src: string): string[] {
  const found: string[] = []
  const walk = (dir: string) => {
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!d.isDirectory() || d.name.startsWith('_') || d.name === 'node_modules') continue
      const full = path.join(dir, d.name)
      if (fs.existsSync(path.join(full, 'index.ts'))) found.push(path.relative(src, full).split(path.sep).join('/'))
      walk(full)
    }
  }
  walk(src)
  return found.sort()
}

function isEntry(x: unknown): x is Entry {
  if ((typeof x !== 'function' && typeof x !== 'object') || x === null || !('info' in x)) return false
  const info = (x as { info?: unknown }).info
  return typeof info === 'object' && info !== null && typeof (info as { kind?: unknown }).kind === 'string'
}

/** Info as plain JSON: functions dropped, non-finite numbers as strings (the TensorWire convention). */
function plain(x: unknown): unknown {
  if (typeof x === 'function' || x === undefined) return undefined
  if (typeof x === 'number') return Number.isFinite(x) ? x : Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf'
  if (x === null || typeof x !== 'object') return x
  if (Array.isArray(x)) return x.map(plain)
  const o: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(x)) {
    const p = plain(v)
    if (p !== undefined) o[k] = p
  }
  return o
}

const errors: string[] = []

// ── Collect ──────────────────────────────────────────────────────────────────────────────────────────────────────────

const server = await createServer({
  configFile: false,
  root,
  server: { middlewareMode: true, hmr: false },
  appType: 'custom',
  logLevel: 'error',
  optimizeDeps: { noDiscovery: true, include: [] },
})

type Found = { entry: Entry; pkg: Pkg['name']; from: string }
const seen = new Map<Entry, Found>()
const moduleSets = new Map<Pkg['name'], Set<string>>()
let primitives: { id: string; module: string; name: string; [k: string]: unknown }[] = []
try {
  for (const pkg of packages) {
    const mods = modulesOf(pkg.src)
    moduleSets.set(pkg.name, new Set(mods))
    for (const mod of mods) {
      const ns = (await server.ssrLoadModule(`${pkg.name}/${mod}`)) as Record<string, unknown>
      const add = (v: unknown) => {
        if (isEntry(v) && !seen.has(v)) seen.set(v, { entry: v, pkg: pkg.name, from: mod })
      }
      for (const v of Object.values(ns)) {
        add(v)
        // A registry table: a plain object whose values are entries.
        if (v !== null && typeof v === 'object' && !isEntry(v) && Object.getPrototypeOf(v) === Object.prototype) {
          const values = Object.values(v)
          if (values.length > 0 && values.every(isEntry)) values.forEach(add)
        }
      }
    }
  }
  const tensor = (await server.ssrLoadModule('aifn/foundation/tensor')) as {
    registry: { list(): typeof primitives }
  }
  primitives = tensor.registry.list()
} finally {
  await server.close()
}

// ── Build ────────────────────────────────────────────────────────────────────────────────────────────────────────────

type Row = { address: string; package: string } & Info
const byKind = new Map<string, Row[]>()
const addresses = new Map<string, Found>()
// Addresses are unique within a kind across both packages.
for (const found of seen.values()) {
  const info = found.entry.info
  for (const field of ['key', 'kind', 'module', 'name', 'stability'] as const) {
    if (typeof info[field] !== 'string' || info[field] === '')
      errors.push(`${found.pkg}/${found.from}: an entry has no ${field} (${JSON.stringify(info.key ?? info.name)})`)
  }
  // Applications may prefix their module with `applied/` (the package's address space); the module is the rest.
  const module = found.pkg === 'aifn-methods' ? info.module.replace(/^applied\//, '') : info.module
  const address = `${module}/${info.key}`
  const clash = addresses.get(`${info.kind} ${address}`)
  if (clash) errors.push(`two ${info.kind} entries have the address '${address}' (${clash.from}, ${found.from})`)
  addresses.set(`${info.kind} ${address}`, found)
  const mods = moduleSets.get(found.pkg)
  if (mods && !mods.has(module))
    errors.push(`${info.kind} '${address}': module '${info.module}' is not a module of ${found.pkg}`)
  const head = { address, package: found.pkg, key: info.key, kind: info.kind, module, name: info.name }
  const row = { ...head, ...(plain(info) as Info), module }
  const rows = byKind.get(info.kind) ?? []
  rows.push(row)
  byKind.set(info.kind, rows)
}
const primitiveRows: Row[] = primitives.map((p) => {
  const doc = (p.doc ?? {}) as { summary?: string; formula?: string; note?: string; references?: string[] }
  return {
    address: p.id,
    package: 'aifn',
    key: p.name,
    kind: 'primitive',
    module: p.module,
    name: p.name,
    stability: 'experimental',
    ...(doc.summary ? { summary: doc.summary } : {}),
    ...(doc.formula ? { formula: doc.formula } : {}),
    ...(doc.note ? { notes: [doc.note] } : {}),
    ...(doc.references?.length ? { cite: doc.references } : {}),
    primitive: p.kind,
    arity: p.arity,
    rules: p.rules,
    ...(p.dtype ? { dtype: p.dtype } : {}),
  } as Row
})
byKind.set('primitive', primitiveRows)

const kinds = [...byKind.keys()].sort()
const catalog = {
  $comment:
    'Generated by scripts/aifn-catalog.ts (make catalog) from the registries of aifn and aifn-methods; do not edit. ' +
    'Each entry is its registry info with Spaces inlined, addressed <module>/<key> within its kind.',
  counts: Object.fromEntries(kinds.map((k) => [k, byKind.get(k)!.length])),
  entries: Object.fromEntries(
    kinds.map((k) => [k, byKind.get(k)!.sort((a, b) => (a.address < b.address ? -1 : a.address > b.address ? 1 : 0))]),
  ),
}
const text = JSON.stringify(catalog, null, 1) + '\n'

// ── Links: notes, glossary, references ──────────────────────────────────────────────────────────────────────────────

const slugs = new Set<string>()
const walkNotes = (dir: string) => {
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!d.isDirectory()) continue
    const full = path.join(dir, d.name)
    if (fs.existsSync(path.join(full, 'index.mdx'))) slugs.add(d.name)
    else walkNotes(full)
  }
}
walkNotes(path.join(root, 'content', 'notes'))
const glossary = parse(fs.readFileSync(path.join(root, 'content', 'glossary.yaml'), 'utf8')) as Record<
  string,
  { note?: string; aliases?: string[] }
>
const glossaryKey = new Map<string, string>()
for (const [key, g] of Object.entries(glossary)) {
  glossaryKey.set(key, key)
  for (const a of g.aliases ?? []) if (!glossaryKey.has(a)) glossaryKey.set(a, key)
}
const references = new Set(
  Object.keys(parse(fs.readFileSync(path.join(root, 'content', 'references.yaml'), 'utf8')) as object),
)

let linked = 0
const notesLinked = new Set<string>()
for (const kind of kinds)
  for (const row of byKind.get(kind)!) {
    const where = `${kind} '${row.address}'`
    for (const slug of row.notes ?? []) {
      if (!slugs.has(slug)) errors.push(`${where}: note '${slug}' is not in content/notes`)
      else notesLinked.add(slug)
    }
    if (row.notes?.length) linked++
    if (row.glossary !== undefined) {
      const key = glossaryKey.get(row.glossary)
      if (!key) errors.push(`${where}: glossary key '${row.glossary}' is not in content/glossary.yaml`)
      else {
        const note = glossary[key].note
        if (note && row.notes?.length && !row.notes.includes(note))
          errors.push(`${where}: glossary '${key}' names note '${note}', which the entry's notes do not list`)
      }
    }
    for (const c of row.cite ?? [])
      if (!references.has(c)) errors.push(`${where}: reference '${c}' is not in content/references.yaml`)
  }

// ── Fixture coverage (report only) ──────────────────────────────────────────────────────────────────────────────────

const fixtureKeys = new Set<string>()
const collectKeys = (x: unknown, depth: number) => {
  if (depth > 3 || x === null || typeof x !== 'object' || Array.isArray(x)) return
  for (const [k, v] of Object.entries(x)) {
    fixtureKeys.add(k.toLowerCase())
    collectKeys(v, depth + 1)
  }
}
for (const pkg of ['core', 'methods']) {
  const dir = path.join(root, 'aifn-js', pkg, 'test', 'fixtures')
  if (!fs.existsSync(dir)) continue
  for (const f of fs.readdirSync(dir, { recursive: true }) as string[])
    if (f.endsWith('.json')) collectKeys(JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')), 0)
}
/** A fixture case exists when the key, or the key without a kind suffix, names a fixture group. */
const hasFixture = (row: Row) => {
  const k = row.key.toLowerCase()
  const bare = k.replace(/(family|bijector|kernel|steps|sampler)$/, '')
  return fixtureKeys.has(k) || fixtureKeys.has(bare)
}
const rowsAll = kinds.filter((k) => k !== 'primitive').flatMap((k) => byKind.get(k)!)
const stable = rowsAll.filter((r) => r.stability === 'stable')
const uncovered = stable.filter((r) => !hasFixture(r))
const covered = rowsAll.filter(hasFixture).length

// ── Report ───────────────────────────────────────────────────────────────────────────────────────────────────────────

const current = fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : ''
if (check) {
  if (current !== text) errors.push('aifn-js/generated/catalog.json is stale: run `make catalog`')
} else if (current !== text) {
  fs.mkdirSync(path.dirname(out), { recursive: true })
  fs.writeFileSync(out, text)
}

const total = rowsAll.length + primitiveRows.length
console.log(
  `aifn-catalog: ${total} entries (${kinds.map((k) => `${k} ${byKind.get(k)!.length}`).join(', ')}); ` +
    `${linked} link ${notesLinked.size} notes`,
)
console.log(
  `fixture coverage (report only): ${stable.length} stable entries, ${uncovered.length} without a reference case` +
    (uncovered.length ? `: ${uncovered.map((r) => `${r.kind} ${r.address}`).join(', ')}` : '') +
    `; ${covered} of ${rowsAll.length} entries match a fixture group`,
)
for (const e of errors) console.error(`✗ ${e}`)
if (errors.length) {
  console.error(`aifn-catalog: ${errors.length} error${errors.length === 1 ? '' : 's'}`)
  process.exit(1)
}
if (!check && current !== text) console.log(`wrote ${path.relative(root, out)}`)
