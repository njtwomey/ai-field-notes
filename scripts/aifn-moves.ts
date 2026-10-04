/**
 * The export-map codemod for the aifn module tree (`.scratch/aifn/module-tree.md` §5). It reads `aifn-js/moves.json`
 * (`{ from, to, export? }` rows: whole files, or single exports cut out of a file) and a snapshot of the public
 * surface taken before any move, and rewrites import specifiers so that each imported name comes from the node that
 * now defines it.
 *
 *   node scripts/aifn-moves.ts snapshot            record every public path's names and their defining files
 *   node scripts/aifn-moves.ts rewrite <dir> ...   rewrite imports in the given directories (or files)
 *
 * Names are resolved through the snapshot (old public path → defining file, taken with the TypeScript checker), then
 * through the moves (defining file → its new file), then to the new file's public path: the node directory holding
 * it, `aifn/<family>/<module>` or `aifn-methods/<area>/…`. Relative imports are resolved from the importer's old
 * location and rewritten the same way: `./x` when the target sits in the importer's module, `../x` for an ancestor's
 * shared file, else the target's public path.
 */
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

const root = path.resolve(import.meta.dirname, '..')
const js = path.join(root, 'aifn-js')
const snapshotFile = path.join(root, '.scratch', 'aifn', 'moves-snapshot.json')
const rel = (p: string) => path.relative(js, p).split(path.sep).join('/')

type Named = { file: string; name: string; type: boolean }
type Snapshot = {
  /** public path (e.g. `aifn/linalg`) → exported name → where it is defined */
  paths: Record<string, Record<string, Named>>
  /** file (relative to aifn-js) → its own exported names → where each is defined (re-exports resolved) */
  files: Record<string, Record<string, Named>>
}

function* walk(dir: string): Generator<string> {
  if (!fs.existsSync(dir)) return
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, d.name)
    if (d.isDirectory()) yield* walk(p)
    else if (/\.tsx?$/.test(d.name)) yield p
  }
}

function snapshot() {
  const sources = [...walk(path.join(js, 'core', 'src')), ...walk(path.join(js, 'applications', 'src'))]
  const program = ts.createProgram(sources, {
    target: ts.ScriptTarget.ES2023,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    allowImportingTsExtensions: true,
    noEmit: true,
    strict: true,
    skipLibCheck: true,
  })
  const checker = program.getTypeChecker()
  const out: Snapshot = { paths: {}, files: {} }
  // Every file's exported names, each marked type-only or not (the checker resolves `export *` and aliases).
  const exported = new Map<string, Map<string, boolean>>()
  for (const file of sources) {
    const sf = program.getSourceFile(file)
    const sym = sf && checker.getSymbolAtLocation(sf)
    const names = new Map<string, boolean>()
    if (sym)
      for (const e of checker.getExportsOfModule(sym)) {
        const target = e.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(e) : e
        names.set(e.name, (target.flags & ts.SymbolFlags.Value) === 0)
      }
    exported.set(path.resolve(file), names)
  }
  // Owners, one hop: an index's name belongs to the file its `export … from './x'` names (recursively through
  // sub-indexes); a name declared in the index itself, or in any other file, belongs to that file.
  const owners = new Map<string, Record<string, Named>>()
  const ownersOf = (file: string): Record<string, Named> => {
    const known = owners.get(file)
    if (known) return known
    const names: Record<string, Named> = {}
    owners.set(file, names)
    const own = exported.get(file) ?? new Map<string, boolean>()
    const sf = program.getSourceFile(file)
    const isIndex = path.basename(file) === 'index.ts'
    const resolveRel = (spec: string) => {
      const base = path.resolve(path.dirname(file), spec)
      for (const c of [base, `${base}.ts`, `${base}.tsx`, path.join(base, 'index.ts')]) if (exported.has(c)) return c
      throw new Error(`${rel(file)}: cannot resolve ${spec}`)
    }
    if (isIndex && sf)
      for (const st of sf.statements) {
        if (!ts.isExportDeclaration(st) || !st.moduleSpecifier) continue
        const spec = (st.moduleSpecifier as ts.StringLiteral).text
        if (!spec.startsWith('.')) throw new Error(`${rel(file)}: re-export from ${spec}`)
        const target = resolveRel(spec)
        const sub = ownersOf(target)
        if (!st.exportClause) {
          for (const [n, v] of Object.entries(sub)) names[n] = v
        } else if (ts.isNamedExports(st.exportClause)) {
          for (const el of st.exportClause.elements) {
            const local = (el.propertyName ?? el.name).text
            const hit = sub[local]
            if (!hit) throw new Error(`${rel(file)}: ${local} not exported by ${spec}`)
            names[el.name.text] = hit
          }
        } else names[st.exportClause.name.text] = { file: rel(target), name: '*', type: false }
      }
    for (const [n, type] of own) if (!names[n]) names[n] = { file: rel(file), name: n, type }
    return names
  }
  for (const file of sources) {
    const r = rel(file)
    out.files[r] = ownersOf(path.resolve(file))
    const m = /^(core|applications)\/src\/(.+)\/index\.ts$/.exec(r)
    if (m) out.paths[`${m[1] === 'core' ? 'aifn' : 'aifn-methods'}/${m[2]}`] = out.files[r]
  }
  fs.mkdirSync(path.dirname(snapshotFile), { recursive: true })
  fs.writeFileSync(snapshotFile, JSON.stringify(out, null, 1))
  console.log(`snapshot: ${Object.keys(out.paths).length} public paths, ${Object.keys(out.files).length} files`)
}

// ── Moves ────────────────────────────────────────────────────────────────────────────────────────────────────────────

type Row = { step: string; from: string; to: string | null; export?: string; via?: string; note?: string }
const movesFile = path.join(js, 'moves.json')
const loadRows = (): Row[] => (JSON.parse(fs.readFileSync(movesFile, 'utf8')) as { moves: Row[] }).moves
const abs = (r: string) => path.join(js, r)
const exists = (r: string) => fs.existsSync(abs(r))

/** Does file `r` (relative to aifn-js) itself declare or re-export `name`? */
function declares(r: string, name: string): boolean {
  if (!exists(r)) return false
  const text = fs.readFileSync(abs(r), 'utf8')
  const n = name.replace(/\$/g, '\\$')
  if (
    new RegExp(
      `^export\\s+(declare\\s+)?(default\\s+)?(async\\s+)?(function\\*?|const|let|class|abstract\\s+class|interface|type|enum)\\s+${n}\\b`,
      'm',
    ).test(text)
  )
    return true
  for (const m of text.matchAll(/^export\s+(?:type\s+)?\{([^}]*)\}/gm))
    for (const el of m[1].split(','))
      if (
        el
          .trim()
          .replace(/^type\s+/, '')
          .split(/\s+as\s+/)
          .pop()!
          .trim() === name
      )
        return true
  return false
}

class Moves {
  rows = loadRows()
  snap = JSON.parse(fs.readFileSync(snapshotFile, 'utf8')) as Snapshot
  fileTo = new Map<string, string | null>()
  fileFrom = new Map<string, string>()
  exportRows = new Map<string, Row>()
  via = new Map<string, string>()
  constructor() {
    for (const r of this.rows) {
      if (r.export) {
        this.exportRows.set(`${r.from}#${r.export}`, r)
        continue
      }
      if (r.to === null) {
        if (!exists(r.from)) {
          this.fileTo.set(r.from, null)
          if (r.via) this.via.set(r.from, r.via)
        }
        continue
      }
      const barrel = (f: string) => fs.readFileSync(abs(f), 'utf8').includes('TODO(tree 1d): remove')
      if (r.from === r.to || (exists(r.to) && (!exists(r.from) || barrel(r.from)))) {
        this.fileTo.set(r.from, r.to)
        this.fileFrom.set(r.to, r.from)
      }
    }
  }
  /** Where the old file `r` is now. */
  now(r: string): string | null {
    return this.fileTo.has(r) ? this.fileTo.get(r)! : r
  }
  /** The original path of the current file `r`. */
  original(r: string): string {
    return this.fileFrom.get(r) ?? r
  }
  /** Where name `name` of old file `file` is defined now: [current file, name there]. */
  locate(file: string, name: string): [string, string] {
    const ex = this.exportRows.get(`${file}#${name}`)
    if (ex && ex.to && declares(ex.to, name) && !declares(this.now(file) ?? '', name)) return [ex.to, name]
    const now = this.now(file)
    if (now === null) {
      const via = this.via.get(file)
      const hit = via && this.snap.paths[via]?.[name]
      if (!hit) throw new Error(`${file} was deleted and ${name} has no via`)
      return this.locate(hit.file, hit.name)
    }
    return [now, name]
  }
}

/** The public path of the node holding current file `r`. */
function publicPath(r: string): string {
  const dir = path.posix.dirname(r)
  if (dir.startsWith('core/src/')) return `aifn/${dir.slice('core/src/'.length)}`
  if (dir.startsWith('applications/src/')) return `aifn-methods/${dir.slice('applications/src/'.length)}`
  throw new Error(`no public path for ${r}`)
}

/** The name under which a node's index exports `name` of its file `file` (both current), or null. */
function indexName(file: string, name: string): string | null {
  const index = path.posix.join(path.posix.dirname(file), 'index.ts')
  if (file === index) return name
  if (!exists(index)) return null
  const text = fs.readFileSync(abs(index), 'utf8')
  const sf = ts.createSourceFile(index, text, ts.ScriptTarget.Latest, true)
  const base = path.posix.basename(file).replace(/\.tsx?$/, '')
  for (const st of sf.statements) {
    if (!ts.isExportDeclaration(st) || !st.moduleSpecifier) continue
    const spec = (st.moduleSpecifier as ts.StringLiteral).text
    if (spec !== `./${base}`) continue
    if (!st.exportClause) return name
    if (ts.isNamespaceExport(st.exportClause)) {
      if (name === '*') return st.exportClause.name.text
      continue
    }
    for (const el of st.exportClause.elements) if ((el.propertyName ?? el.name).text === name) return el.name.text
  }
  return null
}

const problems: string[] = []

/** Old flat folders that are not nodes of the tree (their files wait for 1c). */
const legacyDirs = (() => {
  const spec = JSON.parse(fs.readFileSync(path.join(js, 'modules.json'), 'utf8')) as {
    core: { families: { family: string }[] }
    aliases: { path: string }[]
  }
  const families = new Set(spec.core.families.map((f) => f.family))
  const out = new Set<string>()
  for (const a of spec.aliases) {
    const m = /^(aifn|aifn-methods)\/(.+)$/.exec(a.path)!
    if (m[1] === 'aifn' && families.has(m[2])) continue
    out.add(`${m[1] === 'aifn' ? 'core' : 'applications'}/src/${m[2]}`)
  }
  return out
})()

/** The specifier from current file `from` to current file `to` (relative when `to` is in `from`'s dir or an ancestor's). */
function specifier(from: string, to: string): { spec: string; public: boolean } {
  const fd = path.posix.dirname(from)
  const td = path.posix.dirname(to)
  const pkg = (p: string) => p.split('/')[0]
  const base = path.posix.basename(to).replace(/\.tsx?$/, '')
  const isIndex = base === 'index'
  // A file left in an old flat folder (until 1c) reaches the helpers that moved away by relative path, so that no
  // private helper becomes public for it.
  if (path.posix.basename(from) === 'index.ts' && pkg(from) === pkg(to) && `${td}/`.startsWith(`${fd}/`) && td !== fd) {
    // An index re-exports its descendants by relative path.
    return { spec: `./${path.posix.relative(fd, td)}`, public: false }
  }
  if (!isIndex && pkg(from) === pkg(to) && (fd === td || (fd.startsWith(`${td}/`) && td.split('/').length > 2))) {
    const r = path.posix.relative(fd, path.posix.join(td, base))
    return { spec: r.startsWith('.') ? r : `./${r}`, public: false }
  }
  return { spec: publicPath(to), public: true }
}

function resolveOld(fromOld: string, spec: string, snap: Snapshot): string | null {
  const base = path.posix.join(path.posix.dirname(fromOld), spec)
  for (const c of [base, `${base}.ts`, `${base}.tsx`, `${base}/index.ts`]) if (snap.files[c]) return c
  return null
}

type El = { name: string; local: string; type: boolean }

/** Rewrite the aifn imports of current file `r`; returns true when it changed. */
function rewriteFile(mv: Moves, r: string): boolean {
  const file = abs(r)
  const text = fs.readFileSync(file, 'utf8')
  if (text.includes('TODO(tree 1d): remove')) return false // an alias barrel: generated, never rewritten
  const sf = ts.createSourceFile(
    file,
    text,
    ts.ScriptTarget.Latest,
    true,
    r.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  )
  const orig = mv.original(r)
  const edits: { start: number; end: number; text: string }[] = []
  for (const st of sf.statements) {
    const isImport = ts.isImportDeclaration(st)
    if (!isImport && !ts.isExportDeclaration(st)) continue
    if (!st.moduleSpecifier) continue
    const spec = (st.moduleSpecifier as ts.StringLiteral).text
    // Owners of the imported names in the snapshot.
    let owners: Record<string, Named> | undefined
    let wholeFile: string | undefined
    if (spec.startsWith('.')) {
      // A specifier that resolves in the current tree names the file it resolves to (by that file's original path);
      // one that does not is resolved from the importer's original place.
      const cur = path.posix.join(path.posix.dirname(r), spec)
      const hit = [`${cur}.ts`, `${cur}.tsx`, `${cur}/index.ts`].find((c) => exists(c))
      // A file left in an old flat folder keeps a relative import that resolves (set by hand to avoid init cycles).
      if (hit && legacyDirs.has(path.posix.dirname(r)) && path.posix.dirname(hit) !== path.posix.dirname(r)) continue
      const old = hit ? (mv.snap.files[mv.original(hit)] ? mv.original(hit) : null) : resolveOld(orig, spec, mv.snap)
      if (!old) continue
      owners = mv.snap.files[old]
      wholeFile = old
    } else if (/^aifn(-applied)?\//.test(spec)) {
      owners = mv.snap.paths[spec]
      if (!owners) {
        // A path of the new tree: names cut out of its files since are found through the export rows.
        const dir = `${spec.startsWith('aifn-methods/') ? 'applications/src/' + spec.slice(13) : 'core/src/' + spec.slice(5)}`
        const found: Record<string, Named> = {}
        const nb = (isImport ? st.importClause?.namedBindings : st.exportClause) as
          ts.NamedImportBindings | ts.NamedExportBindings | undefined
        if (!nb || !('elements' in nb)) continue
        for (const e of nb.elements) {
          const n = (e.propertyName ?? e.name).text
          const row = mv.rows.find((x) => x.export === n && x.to && path.posix.dirname(mv.now(x.from) ?? '') === dir)
          if (row) found[n] = { file: row.from, name: n, type: false }
        }
        if (!Object.keys(found).length) continue
        owners = new Proxy(found, {
          get: (t, k: string) => t[k] ?? { file: `${dir}/index.ts`, name: k, type: false, keep: true },
        }) as Record<string, Named>
      }
    } else continue
    const declType = isImport ? !!st.importClause?.isTypeOnly : st.isTypeOnly
    const els: El[] = []
    let namespace: string | undefined
    if (isImport) {
      const clause = st.importClause
      if (!clause) continue
      if (clause.name) throw new Error(`${r}: default import from ${spec}`)
      const nb = clause.namedBindings
      if (nb && ts.isNamespaceImport(nb)) namespace = nb.name.text
      else if (nb)
        for (const e of nb.elements)
          els.push({ name: (e.propertyName ?? e.name).text, local: e.name.text, type: e.isTypeOnly })
    } else {
      const ec = st.exportClause
      if (!ec) {
        // `export * from` in a module file: keep, pointing at the file's new place.
        if (!wholeFile) continue
        const now = mv.now(wholeFile)
        if (!now) continue
        const s2 = specifier(r, now).spec
        if (s2 !== spec)
          edits.push({ start: st.moduleSpecifier.getStart(sf), end: st.moduleSpecifier.getEnd(), text: `'${s2}'` })
        continue
      }
      if (ts.isNamespaceExport(ec)) continue
      for (const e of ec.elements)
        els.push({ name: (e.propertyName ?? e.name).text, local: e.name.text, type: e.isTypeOnly })
    }
    if (namespace) {
      if (!wholeFile) {
        problems.push(`${r}: namespace import of ${spec} (rewrite by hand)`)
        continue
      }
      const now = mv.now(wholeFile)
      if (!now) continue
      let { spec: s2 } = specifier(r, now)
      if (legacyDirs.has(path.posix.dirname(now)) && path.posix.dirname(now) !== path.posix.dirname(r)) {
        // A namespace of a file still in an old flat folder: by relative path (its alias is the whole old module).
        const rp = path.posix.relative(path.posix.dirname(r), now.replace(/\.tsx?$/, ''))
        s2 = rp.startsWith('.') ? rp : `./${rp}`
      }
      if (s2 !== spec)
        edits.push({ start: st.moduleSpecifier.getStart(sf), end: st.moduleSpecifier.getEnd(), text: `'${s2}'` })
      continue
    }
    const groups = new Map<string, string[]>()
    let changed = false
    for (const e of els) {
      const owner = owners[e.name]
      if (!owner) {
        problems.push(`${r}: ${e.name} not in snapshot of ${spec}`)
        groups.set(spec, [...(groups.get(spec) ?? []), elText(e.type, e.name, e.local)])
        continue
      }
      if ((owner as Named & { keep?: boolean }).keep) {
        groups.set(spec, [...(groups.get(spec) ?? []), elText(e.type, e.name, e.local)])
        continue
      }
      const [f2, n2] = mv.locate(owner.file, owner.name)
      const target = { ...specifier(r, f2) }
      let imported = n2
      if (target.public) {
        let ix = indexName(f2, n2)
        if (ix === null && legacyDirs.has(path.posix.dirname(r)) && r.split('/')[0] === f2.split('/')[0]) {
          // A file left in an old flat folder (until 1c) reaches a private helper that moved away by relative path,
          // so that the helper does not become public for it.
          const rp = path.posix.relative(path.posix.dirname(r), f2.replace(/\.tsx?$/, ''))
          target.spec = rp.startsWith('.') ? rp : `./${rp}`
          target.public = false
          ix = n2
        }
        if (ix === null) {
          addInternal(f2, n2, owner.type || e.type || declType, r)
          ix = n2
        }
        imported = ix
      } else if (n2 === '*') {
        problems.push(`${r}: namespace ${e.name} of ${f2} imported relatively (rewrite by hand)`)
        imported = e.name
      }
      if (target.spec !== spec || imported !== e.name) changed = true
      groups.set(target.spec, [...(groups.get(target.spec) ?? []), elText(e.type, imported, e.local)])
    }
    if (!changed) continue
    const kw = isImport ? 'import' : 'export'
    const out = [...groups].map(([s2, list]) => `${kw} ${declType ? 'type ' : ''}{ ${list.join(', ')} } from '${s2}'`)
    edits.push({ start: st.getStart(sf), end: st.getEnd(), text: out.join('\n') })
  }
  if (!edits.length) return false
  let next = text
  for (const e of edits.sort((a, b) => b.start - a.start)) next = next.slice(0, e.start) + e.text + next.slice(e.end)
  if (next === text) return false
  fs.writeFileSync(file, next)
  return true
}

/**
 * Export `name` of `file` from its node's index as internal: a helper that was private to an old module and is now
 * shared across nodes (module-tree.md §4.3). Marked `@internal` so the API report flags it for review in phase 1.
 */
function addInternal(file: string, name: string, type: boolean, importer: string) {
  const index = abs(path.posix.join(path.posix.dirname(file), 'index.ts'))
  const base = path.posix.basename(file).replace(/\.tsx?$/, '')
  const marker =
    '// @internal: helpers shared across nodes since the tree move (module-tree.md §4.3); review in phase 1.'
  let text = fs.existsSync(index) ? fs.readFileSync(index, 'utf8') : ''
  if (!text.includes(marker)) text = `${text.trimEnd()}\n\n${marker}\n`
  text = `${text.trimEnd()}\n/** @internal */\nexport { ${type ? 'type ' : ''}${name} } from './${base}'\n`
  fs.writeFileSync(index, text)
  internals.push(`${name} (${file}) for ${importer}`)
}
const internals: string[] = []

function elText(type: boolean, imported: string, local: string): string {
  return `${type ? 'type ' : ''}${imported}${local !== imported ? ` as ${local}` : ''}`
}

// ── Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Move the whole files of `step` (plain rename; directories created). */
function move(step: string) {
  let n = 0
  for (const r of loadRows()) {
    if (r.step !== step || r.export || r.from === r.to) continue
    if (!exists(r.from)) continue
    if (r.to === null) {
      fs.rmSync(abs(r.from))
      n++
      continue
    }
    if (exists(r.to) && !fs.readFileSync(abs(r.to), 'utf8').includes('@skeleton')) throw new Error(`${r.to} exists`)
    fs.mkdirSync(path.dirname(abs(r.to)), { recursive: true })
    fs.renameSync(abs(r.from), abs(r.to))
    n++
  }
  console.log(`move ${step}: ${n} files`)
}

/** Rewrite imports in every .ts/.tsx file under the given paths (relative to the repo root). */
function rewrite(targets: string[]) {
  const mv = new Moves()
  let n = 0
  for (const t of targets) {
    const p = path.resolve(root, t)
    const list = fs.statSync(p).isDirectory() ? [...walk(p)] : [p]
    for (const f of list) {
      const r = path.relative(js, f).split(path.sep).join('/')
      if (f.includes(`${path.sep}node_modules${path.sep}`)) continue
      if (rewriteFile(mv, r.startsWith('..') ? path.relative(js, f) : r)) n++
    }
  }
  console.log(`rewrite: ${n} files changed; ${internals.length} internal exports added`)
  for (const i of internals) console.log(`  internal: ${i}`)
}

/**
 * Write the index of every new node that receives files of `step`, from the old indexes' statements (each statement
 * goes to the node its file moved to; statements naming exports cut out by an export row go with the export).
 */
function indexes(step: string) {
  const mv = new Moves()
  const nodes = new Map<string, string[]>() // node dir → statements
  const docs = new Map<string, string>()
  const olds = new Set<string>()
  for (const r of mv.rows) if (r.step === step && r.to && !r.export) olds.add(path.posix.dirname(r.from))
  const movedIndexes = new Set(
    mv.rows.filter((r) => r.to && r.from.endsWith('/index.ts')).map((r) => path.posix.dirname(r.to!)),
  )
  for (const oldDir of olds) {
    const oldIndex = `${oldDir}/index.ts`
    // The old index text: the barrel may already be written, so read the statements from a saved copy.
    const saved = path.join(root, '.scratch', 'aifn', 'old-indexes', oldIndex)
    if (!fs.existsSync(saved)) {
      if (!exists(oldIndex)) continue
      fs.mkdirSync(path.dirname(saved), { recursive: true })
      fs.copyFileSync(abs(oldIndex), saved)
    }
    const text = fs.readFileSync(saved, 'utf8')
    const sf = ts.createSourceFile(saved, text, ts.ScriptTarget.Latest, true)
    const doc = /^\/\*\*[\s\S]*?\*\//.exec(text)?.[0] ?? ''
    for (const st of sf.statements) {
      if (!ts.isExportDeclaration(st) || !st.moduleSpecifier) continue
      const spec = (st.moduleSpecifier as ts.StringLiteral).text
      const old = resolveOld(oldIndex, spec, mv.snap)
      if (!old) continue
      const now = mv.now(old)
      if (now === null) continue
      const ec = st.exportClause
      const put = (file: string, stmt: string) => {
        const node = path.posix.dirname(file)
        if (movedIndexes.has(node) || node === oldDir) return
        const base = path.posix.basename(file).replace(/\.tsx?$/, '')
        nodes.set(node, [...(nodes.get(node) ?? []), stmt.replace('%', `./${base}`)])
        if (!docs.has(node)) docs.set(node, doc)
      }
      if (!ec || ts.isNamespaceExport(ec)) {
        put(now, st.getText(sf).replace(/(['"])[^'"]+\1\s*$/, "'%'"))
        continue
      }
      const byFile = new Map<string, string[]>()
      for (const e of ec.elements) {
        const [f2] = mv.locate(old, (e.propertyName ?? e.name).text)
        byFile.set(f2, [...(byFile.get(f2) ?? []), e.getText(sf)])
      }
      for (const [f2, list] of byFile) put(f2, `export ${st.isTypeOnly ? 'type ' : ''}{ ${list.join(', ')} } from '%'`)
    }
  }
  for (const [node, stmts] of nodes) {
    const index = abs(`${node}/index.ts`)
    const pub = publicPath(`${node}/index.ts`)
    const doc = (docs.get(node) ?? '').replace(/`aifn\/[a-z-]+`/, `\`${pub}\``)
    let prev = ''
    if (fs.existsSync(index)) prev = fs.readFileSync(index, 'utf8')
    if (prev.includes('@skeleton')) prev = ''
    const body = stmts.filter((s) => !prev.includes(s)).join('\n')
    fs.writeFileSync(index, prev ? `${prev.trimEnd()}\n${body}\n` : `${doc}\n\n${body}\n`)
    console.log(`index: ${node}`)
  }
}

/** Regenerate the alias barrel of each old folder listed: the old surface, re-exported from where each name lives. */
function barrels(oldDirs: string[]) {
  const mv = new Moves()
  for (const oldDir of oldDirs) {
    const pub = publicPath(`${oldDir}/index.ts`)
    const names = mv.snap.paths[pub]
    if (!names) throw new Error(`no snapshot for ${pub}`)
    const saved = path.join(root, '.scratch', 'aifn', 'old-indexes', `${oldDir}/index.ts`)
    if (!fs.existsSync(saved) && exists(`${oldDir}/index.ts`)) {
      fs.mkdirSync(path.dirname(saved), { recursive: true })
      fs.copyFileSync(abs(`${oldDir}/index.ts`), saved)
    }
    const groups = new Map<string, { values: string[]; types: string[] }>()
    for (const [name, owner] of Object.entries(names)) {
      const [f2, n2] = mv.locate(owner.file, owner.name)
      let spec: string
      let exported: string
      if (path.posix.dirname(f2) === oldDir) {
        spec = `./${path.posix.basename(f2).replace(/\.tsx?$/, '')}`
        exported = n2
      } else {
        spec = publicPath(f2)
        const inDir = spec.startsWith(`${publicPath(`${oldDir}/index.ts`)}/`)
        const ix = indexName(f2, n2)
        if (ix === null) {
          problems.push(`barrel ${pub}: ${n2} (${f2}) not exported by ${spec}`)
          continue
        }
        exported = ix
        if (inDir) spec = `./${spec.slice(pub.length + 1)}`
      }
      const g = groups.get(spec) ?? { values: [], types: [] }
      ;(owner.type ? g.types : g.values).push(exported === name ? name : `${exported} as ${name}`)
      groups.set(spec, g)
    }
    const lines = [...groups]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([spec, g]) => {
        const list = [...g.values.sort(), ...g.types.sort().map((t) => `type ${t}`)]
        return `export { ${list.join(', ')} } from '${spec}'`
      })
    const doc = `/**\n * @deprecated \`${pub}\` moved into the module tree (.scratch/aifn/module-tree.md); this barrel re-exports its old\n * surface from the new paths. TODO(tree 1d): remove.\n */`
    fs.writeFileSync(abs(`${oldDir}/index.ts`), `${doc}\n\n${lines.join('\n')}\n`)
    console.log(`barrel: ${pub} (${Object.keys(names).length} names, ${groups.size} sources)`)
  }
}

// ── Export-level cuts (step 1c) ──────────────────────────────────────────────────────────────────────────────────────

/** The names a top-level statement declares. */
function declared(st: ts.Statement): string[] {
  if (
    ts.isFunctionDeclaration(st) ||
    ts.isClassDeclaration(st) ||
    ts.isInterfaceDeclaration(st) ||
    ts.isTypeAliasDeclaration(st) ||
    ts.isEnumDeclaration(st)
  )
    return st.name ? [st.name.text] : []
  if (ts.isVariableStatement(st))
    return st.declarationList.declarations.flatMap((d) => (ts.isIdentifier(d.name) ? [d.name.text] : []))
  return []
}
const isExported = (st: ts.Statement) =>
  !!(ts.canHaveModifiers(st) && ts.getModifiers(st)?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword))

/** Identifiers referenced anywhere inside a node. */
function referenced(node: ts.Node, out = new Set<string>()): Set<string> {
  if (ts.isIdentifier(node)) out.add(node.text)
  node.forEachChild((c) => void referenced(c, out))
  return out
}

/** Drop import specifiers whose local name no longer appears in the rest of the text. */
function pruneImports(text: string, file: string): string {
  const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true)
  const edits: { start: number; end: number; text: string }[] = []
  const body = new Set<string>()
  for (const st of sf.statements) if (!ts.isImportDeclaration(st)) referenced(st, body)
  for (const st of sf.statements) {
    if (!ts.isImportDeclaration(st) || !st.importClause) continue
    const nb = st.importClause.namedBindings
    if (!nb) continue
    if (ts.isNamespaceImport(nb)) {
      if (!body.has(nb.name.text)) edits.push({ start: st.getFullStart(), end: st.getEnd(), text: '' })
      continue
    }
    const keep = nb.elements.filter((e) => body.has(e.name.text))
    if (keep.length === nb.elements.length) continue
    if (!keep.length) {
      edits.push({ start: st.getFullStart(), end: st.getEnd(), text: '' })
      continue
    }
    edits.push({ start: nb.getStart(sf), end: nb.getEnd(), text: `{ ${keep.map((e) => e.getText(sf)).join(', ')} }` })
  }
  let out = text
  for (const e of edits.sort((a, b) => b.start - a.start)) out = out.slice(0, e.start) + e.text + out.slice(e.end)
  return out
}

/**
 * Cut the exports `names` (and the private helpers only they use) out of `src` into `dest` (both current paths
 * relative to aifn-js), bodies unchanged; copy the imports and prune the unused ones; move the names between the two
 * nodes' indexes; record export rows. Reports what the moved code still needs from `src`, and what `src` needs back.
 */
function cutExports(src: string, dest: string, names: string[], step: string, doc?: string) {
  const text = fs.readFileSync(abs(src), 'utf8')
  const sf = ts.createSourceFile(src, text, ts.ScriptTarget.Latest, true)
  const top = new Map<string, ts.Statement>()
  for (const st of sf.statements) for (const n of declared(st)) top.set(n, st)
  const selected = new Set<ts.Statement>()
  for (const n of names) {
    const st = top.get(n)
    if (!st) throw new Error(`${src}: no top-level ${n}`)
    selected.add(st)
  }
  // Private helpers used only by the moved statements go with them.
  for (let changed = true; changed;) {
    changed = false
    const rest = sf.statements.filter((st) => !selected.has(st) && !ts.isImportDeclaration(st))
    const usedByRest = new Set<string>()
    for (const st of rest) referenced(st, usedByRest)
    const usedBySel = new Set<string>()
    for (const st of selected) referenced(st, usedBySel)
    for (const n of usedBySel) {
      const st = top.get(n)
      if (!st || selected.has(st) || isExported(st)) continue
      // Used by any other remaining statement (not itself)?
      const others = new Set<string>()
      for (const r of rest) if (r !== st) referenced(r, others)
      if (declared(st).some((d) => others.has(d))) continue
      selected.add(st)
      changed = true
    }
  }
  const usedBySel = new Set<string>()
  for (const st of selected) referenced(st, usedBySel)
  const rest = sf.statements.filter((st) => !selected.has(st) && !ts.isImportDeclaration(st))
  const usedByRest = new Set<string>()
  for (const st of rest) referenced(st, usedByRest)
  const moved = new Set([...selected].flatMap(declared))
  const needFromSrc = [...usedBySel].filter((n) => top.has(n) && !moved.has(n))
  const needBack = [...moved].filter((n) => usedByRest.has(n))
  const imports = sf.statements.filter(ts.isImportDeclaration).map((st) => st.getText(sf))
  const chunks = sf.statements
    .filter((st) => selected.has(st))
    .map((st) => text.slice(st.getFullStart(), st.getEnd()).replace(/^\s*\n/, ''))
  // The source without the moved statements.
  let srcOut = text
  for (const st of [...selected].sort((a, b) => b.getFullStart() - a.getFullStart()))
    srcOut = srcOut.slice(0, st.getFullStart()) + srcOut.slice(st.getEnd())
  srcOut = pruneImports(srcOut, src).replace(/\n{3,}/g, '\n\n')
  // The destination: its doc, the source's imports (re-pointed from the destination), then the statements.
  const header = exists(dest)
    ? fs.readFileSync(abs(dest), 'utf8').trimEnd() + '\n\n'
    : `${doc ?? '/** Cut from ' + src + '. */'}\n\n`
  const reImports = imports.map((line) => {
    const m = /from '(\.[^']*)'/.exec(line)
    if (!m) return line
    const target = path.posix.join(path.posix.dirname(src), m[1])
    const hit = [`${target}.ts`, `${target}/index.ts`].find((c) => exists(c))
    if (!hit) return line
    return line.replace(m[0], `from '${specifier(dest, hit).spec}'`)
  })
  if (needFromSrc.length) reImports.push(`import { ${needFromSrc.join(', ')} } from '${specifier(dest, src).spec}'`)
  let destOut = `${header}${exists(dest) ? '' : reImports.join('\n') + '\n\n'}${chunks.join('\n\n')}\n`
  if (exists(dest)) {
    // Merge the imports in front of the existing body.
    const d = fs.readFileSync(abs(dest), 'utf8')
    const dsf = ts.createSourceFile(dest, d, ts.ScriptTarget.Latest, true)
    const lastImport = [...dsf.statements].filter(ts.isImportDeclaration).pop()
    const at = lastImport ? lastImport.getEnd() : (/^\/\*\*[\s\S]*?\*\/\n/.exec(d)?.[0].length ?? 0)
    const fresh = reImports.filter((line) => !d.includes(line))
    destOut = `${d.slice(0, at)}\n${fresh.join('\n')}\n${d.slice(at).trimEnd()}\n\n${chunks.join('\n\n')}\n`
  }
  destOut = pruneImports(destOut, dest).replace(/\n{3,}/g, '\n\n')
  fs.mkdirSync(path.dirname(abs(dest)), { recursive: true })
  fs.writeFileSync(abs(src), srcOut)
  fs.writeFileSync(abs(dest), destOut)
  // Indexes: take the moved names out of the source node's statement, add them to the destination node's index.
  const exportedMoved = [...selected].filter(isExported).flatMap(declared)
  const srcIndex = abs(path.posix.join(path.posix.dirname(src), 'index.ts'))
  const srcBase = path.posix.basename(src).replace(/\.tsx?$/, '')
  const typeNames = new Set(
    [...selected].filter((st) => ts.isInterfaceDeclaration(st) || ts.isTypeAliasDeclaration(st)).flatMap(declared),
  )
  if (fs.existsSync(srcIndex) && path.basename(src) !== 'index.ts') {
    let t = fs.readFileSync(srcIndex, 'utf8')
    t = t.replace(new RegExp(`export (type )?\\{([^}]*)\\} from '\\./${srcBase}'`, 'g'), (_all, ty, list: string) => {
      const keep = list
        .split(',')
        .map((x) => x.trim())
        .filter((x) => x && !exportedMoved.includes(x.replace(/^type\s+/, '').split(/\s+as\s+/)[0]))
      return keep.length ? `export ${ty ?? ''}{ ${keep.join(', ')} } from './${srcBase}'` : ''
    })
    fs.writeFileSync(srcIndex, t.replace(/\n{3,}/g, '\n\n'))
  }
  const destIndex = abs(path.posix.join(path.posix.dirname(dest), 'index.ts'))
  const destBase = path.posix.basename(dest).replace(/\.tsx?$/, '')
  if (path.basename(dest) !== 'index.ts' && exportedMoved.length) {
    let t = fs.existsSync(destIndex) ? fs.readFileSync(destIndex, 'utf8') : ''
    if (t.includes('@skeleton')) t = `/** \`${publicPath(dest)}\`. */\n`
    const list = exportedMoved.map((n) => (typeNames.has(n) ? `type ${n}` : n)).join(', ')
    t = `${t.trimEnd()}\nexport { ${list} } from './${destBase}'\n`
    fs.writeFileSync(destIndex, t)
  }
  const rows = loadRows()
  const orig = new Moves().original(src)
  for (const n of exportedMoved) rows.push({ step, from: orig, export: n, to: dest })
  fs.writeFileSync(
    movesFile,
    JSON.stringify(
      { $comment: (JSON.parse(fs.readFileSync(movesFile, 'utf8')) as { $comment: string }).$comment, moves: rows },
      null,
      1,
    ),
  )
  console.log(`cut ${src} → ${dest}: ${[...moved].join(', ')}`)
  if (needFromSrc.length) console.log(`  dest imports from src: ${needFromSrc.join(', ')} (export them if private)`)
  if (needBack.length) console.log(`  ! src still uses: ${needBack.join(', ')} (import them back)`)
}

/** Drop names an index exports twice (the later statement loses them); drop statements left empty. */
function dedupe(targets: string[]) {
  for (const t of targets)
    for (const f of walk(path.resolve(root, t))) {
      if (path.basename(f) !== 'index.ts') continue
      const text = fs.readFileSync(f, 'utf8')
      const sf = ts.createSourceFile(f, text, ts.ScriptTarget.Latest, true)
      const seen = new Set<string>()
      const edits: { start: number; end: number; text: string }[] = []
      for (const st of sf.statements) {
        if (!ts.isExportDeclaration(st) || !st.exportClause || !ts.isNamedExports(st.exportClause)) continue
        const keep = st.exportClause.elements.filter((e) => !seen.has(e.name.text))
        for (const e of st.exportClause.elements) seen.add(e.name.text)
        if (keep.length === st.exportClause.elements.length) continue
        const spec = (st.moduleSpecifier as ts.StringLiteral).text
        edits.push({
          start: st.getStart(sf),
          end: st.getEnd(),
          text: keep.length
            ? `export ${st.isTypeOnly ? 'type ' : ''}{ ${keep.map((e) => e.getText(sf)).join(', ')} } from '${spec}'`
            : '',
        })
      }
      if (!edits.length) continue
      let out = text
      for (const e of edits.sort((a, b) => b.start - a.start)) out = out.slice(0, e.start) + e.text + out.slice(e.end)
      fs.writeFileSync(f, out.replace(/\n{3,}/g, '\n\n'))
      console.log(`dedupe: ${path.relative(root, f)}`)
    }
}

/** Where each name of an old public path lives now: `where aifn/distributions Normal kl …`. */
function where(old: string, names: string[]) {
  const mv = new Moves()
  const owners = mv.snap.paths[old]
  for (const n of names) {
    const o = owners?.[n]
    if (!o) {
      console.log(`${n}\t?`)
      continue
    }
    const [f2, n2] = mv.locate(o.file, o.name)
    const pub = publicPath(f2)
    console.log(`${n}\t${pub}\t${indexName(f2, n2) ?? '(not exported)'}`)
  }
}

const [cmd, ...args] = process.argv.slice(2)
if (cmd === 'snapshot') {
  if (fs.existsSync(snapshotFile) && !args.includes('--force')) throw new Error('snapshot exists; never retake it')
  snapshot()
} else if (cmd === 'move') move(args[0])
else if (cmd === 'rewrite') rewrite(args)
else if (cmd === 'indexes') indexes(args[0])
else if (cmd === 'barrels') barrels(args)
else if (cmd === 'dedupe') dedupe(args)
else if (cmd === 'where') where(args[0], args.slice(1))
else if (cmd === 'cut') {
  const doc = process.env.DOC
  cutExports(args[0], args[1], args.slice(2), process.env.STEP ?? 'split', doc)
} else throw new Error('usage: snapshot | move <step> | rewrite <path…> | indexes <step> | barrels <old dir…>')
for (const p of problems) console.error(`✗ ${p}`)
if (problems.length) process.exitCode = 1
