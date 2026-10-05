/**
 * `make catalog-check` (part of `make check`): collect every registry entry of the installed engine packages
 * (`aifn-compute` and `aifn-methods`) and check that each one's links into this repository's content exist. The engine
 * cannot check them: it does not have the notes, the glossary or the references.
 *
 * - **Collection.** Every module named in the `exports` of `node_modules/aifn-compute/package.json` and
 *   `node_modules/aifn-methods/package.json` is imported by name (`aifn-compute/optim/first-order`, …), so its
 *   registries are built exactly as an importer builds them. An entry is any exported value carrying `info` with a
 *   `kind` (`define(info, value)`), found directly among a module's exports or inside an exported table of entries
 *   (`windowRegistry`, `firstOrderAlgorithms`, …). Entries are deduplicated by identity, so a value re-exported by a
 *   family index counts once. Primitives come from the primitive table of `aifn-compute/foundation/tensor`.
 * - **Address.** `<info.module>/<info.key>` within a kind (`optim/first-order/adam`); two different entries with one
 *   address fail, and every `info.module` must be a module of its package.
 * - **Links.** Every `notes` slug is a note in `content/notes`; every `glossary` key is in `content/glossary.yaml`, and
 *   when that glossary entry names a note, the registry entry lists the same note; every `cite` key is in
 *   `content/references.yaml`.
 *
 * ```bash
 * node scripts/aifn-catalog.ts   # fail when a link is broken
 * ```
 */
import fs from 'node:fs'
import path from 'node:path'
import { parse } from 'yaml'

const root = path.resolve(import.meta.dirname, '..')

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
type PkgName = 'aifn-compute' | 'aifn-methods'
const packages: PkgName[] = ['aifn-compute', 'aifn-methods']

/** Module paths of an installed package: its `exports` keys without the leading `./` (the root `.` excluded). */
function modulesOf(pkg: PkgName): string[] {
  const manifest = path.join(root, 'node_modules', pkg, 'package.json')
  const exports = (JSON.parse(fs.readFileSync(manifest, 'utf8')) as { exports: Record<string, unknown> }).exports
  return Object.keys(exports)
    .filter((k) => k.startsWith('./') && !k.includes('*'))
    .map((k) => k.slice(2))
    .sort()
}

function isEntry(x: unknown): x is Entry {
  if ((typeof x !== 'function' && typeof x !== 'object') || x === null || !('info' in x)) return false
  const info = (x as { info?: unknown }).info
  return typeof info === 'object' && info !== null && typeof (info as { kind?: unknown }).kind === 'string'
}

const errors: string[] = []

// ── Collect ──────────────────────────────────────────────────────────────────────────────────────────────────────────

// Imported by name: this script sits in the repository, so the specifiers resolve to its node_modules.
const load = async (specifier: string) => (await import(specifier)) as Record<string, unknown>

type Found = { entry: Entry; pkg: PkgName; from: string }
const seen = new Map<Entry, Found>()
const moduleSets = new Map<PkgName, Set<string>>()
for (const pkg of packages) {
  const mods = modulesOf(pkg)
  moduleSets.set(pkg, new Set(mods))
  for (const mod of mods) {
    const ns = await load(`${pkg}/${mod}`)
    const add = (v: unknown) => {
      if (isEntry(v) && !seen.has(v)) seen.set(v, { entry: v, pkg, from: mod })
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
type Primitive = { id: string; module: string; name: string; doc?: { note?: string; references?: string[] } }
const tensor = (await load('aifn-compute/foundation/tensor')) as { registry: { list(): Primitive[] } }
const primitives = tensor.registry.list()

// ── Rows ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

type Row = { kind: string; address: string; notes?: readonly string[]; glossary?: string; cite?: readonly string[] }
const rows: Row[] = []
const addresses = new Map<string, Found>()
// Addresses are unique within a kind across both packages.
for (const found of seen.values()) {
  const info = found.entry.info
  for (const field of ['key', 'kind', 'module', 'name', 'stability'] as const) {
    if (typeof info[field] !== 'string' || info[field] === '')
      errors.push(`${found.pkg}/${found.from}: an entry has no ${field} (${JSON.stringify(info.key ?? info.name)})`)
  }
  // Methods may prefix their module with `applied/` (the package's address space); the module is the rest.
  const module = found.pkg === 'aifn-methods' ? info.module.replace(/^applied\//, '') : info.module
  const address = `${module}/${info.key}`
  const clash = addresses.get(`${info.kind} ${address}`)
  if (clash) errors.push(`two ${info.kind} entries have the address '${address}' (${clash.from}, ${found.from})`)
  addresses.set(`${info.kind} ${address}`, found)
  if (!moduleSets.get(found.pkg)?.has(module))
    errors.push(`${info.kind} '${address}': module '${info.module}' is not a module of ${found.pkg}`)
  rows.push({ kind: info.kind, address, notes: info.notes, glossary: info.glossary, cite: info.cite })
}
for (const p of primitives)
  rows.push({
    kind: 'primitive',
    address: p.id,
    ...(p.doc?.note ? { notes: [p.doc.note] } : {}),
    ...(p.doc?.references?.length ? { cite: p.doc.references } : {}),
  })

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
for (const row of rows) {
  const where = `${row.kind} '${row.address}'`
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

// ── Report ───────────────────────────────────────────────────────────────────────────────────────────────────────────

const counts = new Map<string, number>()
for (const r of rows) counts.set(r.kind, (counts.get(r.kind) ?? 0) + 1)
const version = (pkg: PkgName) =>
  (JSON.parse(fs.readFileSync(path.join(root, 'node_modules', pkg, 'package.json'), 'utf8')) as { version: string })
    .version
console.log(
  `aifn-catalog (${packages.map((p) => `${p} ${version(p)}`).join(', ')}): ${rows.length} entries ` +
    `(${[...counts.keys()]
      .sort()
      .map((k) => `${k} ${counts.get(k)}`)
      .join(', ')}); ${linked} link ${notesLinked.size} notes`,
)
for (const e of errors) console.error(`✗ ${e}`)
if (errors.length) {
  console.error(`aifn-catalog: ${errors.length} error${errors.length === 1 ? '' : 's'}`)
  process.exit(1)
}
