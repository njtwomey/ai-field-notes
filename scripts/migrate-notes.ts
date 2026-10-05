/**
 * Codemod: move note widgets off `aifn-render/compat` and the site's maths (`@/lib/math…`, `@/lib/dsp`) onto
 * Plot v2 (`Figure`, `useFigureState`, `Plot` + layers) and `aifn`. The mapping is in the engine docs (https://njtwomey.github.io/aifn-engine/).
 *
 *   node scripts/migrate-notes.ts --dry-run [paths…]   # report per file what it would change and what is left
 *   node scripts/migrate-notes.ts [paths…]             # rewrite in place and format with prettier
 *   --json <file>   also write the report as JSON       --quiet   totals only     --print   print each result
 *
 * Paths are files or folders (default content/notes). The mechanical part only: imports, `Interactive` → `Figure`
 * shells, `useParam`/`useState` + `Param*` controls → `useFigureState` fields, `XYChart`/`Heatmap` with literal or
 * memoised series → `Plot` + layers, and site maths → `aifn` where the mapping is 1:1. Anything else is left as it was
 * and marked `// MIGRATE: <reason>` above its statement. Idempotent: a second run changes nothing.
 *
 * Each phase reads the current text with ts-morph, collects text edits, and applies them at once; the next phase
 * re-parses. Edits contained in a larger edit of the same phase are dropped (a removed control takes its references).
 */
import fs from 'node:fs'
import path from 'node:path'
import { Node, Project, SyntaxKind, type SourceFile } from 'ts-morph'

// ── Tables ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Names that live in `aifn-render/compat` (or are legacy there): a reference to one is a compat usage. */
const COMPAT = new Set([
  'Interactive',
  'XYChart',
  'Heatmap',
  'ImagePlot',
  'GlyphPlot',
  'MarginalPanels',
  'ParamSlider',
  'ParamChoice',
  'ParamSwitch',
  'ParamButton',
  'ParamNumberField',
  'useParam',
  'StepControls',
  'XYSeries',
  'Series',
  'HeatmapOverlay',
  'HeatmapProps',
  'XYChartProps',
  'XYRect',
  'GradientEstimators',
  'useDebouncedCallback',
])
const RENDER_MODULES = new Set(['aifn-render', 'aifn-render/compat'])
const SITE_MATHS = /^@\/lib\/(math|dsp|distributions)(\/|$)/

type MathRule =
  | { kind: 'fn'; module: string; name: string; arity: number; reorder?: number[]; pack?: (a: string[]) => string }
  | { kind: 'type'; module: string; name: string }
  | { kind: 'rng' }
  | { kind: 'linspace' }

const special = (name: string, arity = 1, to = name, reorder?: number[]): [string, MathRule] => [
  name,
  { kind: 'fn', module: 'aifn-compute/numerics/special', name: to, arity, reorder },
]
/** Site maths → aifn, where the mapping is 1:1 (same values; at most a rename, an argument order or a wrapper). */
const MATHS: Record<string, Record<string, MathRule>> = {
  '@/lib/math': {
    rng: { kind: 'rng' },
    linspace: { kind: 'linspace' },
    ...Object.fromEntries([special('sigmoid')]),
  },
  '@/lib/math/special': Object.fromEntries([
    special('normalCdf'),
    special('normalPdf'),
    special('normalQuantile'),
    special('logGamma'),
    special('logFactorial'),
    special('erf'),
    special('logChoose', 2),
    special('studentTCdf', 2),
    special('incompleteBeta', 3, 'regularisedBeta', [1, 2, 0]),
    special('incompleteGamma', 2, 'regularisedGammaP'),
  ]),
  '@/lib/math/tests': Object.fromEntries([special('studentTQuantile', 2)]),
  '@/lib/math/mat2': {
    Vec2: { kind: 'type', module: 'aifn-compute/numerics/linalg', name: 'Vec2' },
    Mat2: { kind: 'type', module: 'aifn-compute/numerics/linalg', name: 'Mat2' },
    Eig2: { kind: 'type', module: 'aifn-compute/numerics/linalg', name: 'Eig2' },
    apply: { kind: 'fn', module: 'aifn-compute/numerics/linalg', name: 'apply2', arity: 2 },
    det: { kind: 'fn', module: 'aifn-compute/numerics/linalg', name: 'det2', arity: 1 },
    eig2: { kind: 'fn', module: 'aifn-compute/numerics/linalg', name: 'eig2', arity: 1 },
    // The site's symmetric (p, q, r) forms take the matrix [[p, q], [q, r]] in aifn (same values; eigh2 signs vectors).
    eigSym: {
      kind: 'fn',
      module: 'aifn-compute/numerics/linalg',
      name: 'eigh2',
      arity: 3,
      pack: ([p, q, r]) => `[[${p}, ${q}], [${q}, ${r}]]`,
    },
    cholesky2: {
      kind: 'fn',
      module: 'aifn-compute/numerics/linalg',
      name: 'cholesky2',
      arity: 3,
      pack: ([p, q, r]) => `[[${p}, ${q}], [${q}, ${r}]]`,
    },
  },
  '@/lib/dsp': {
    hzToMel: { kind: 'fn', module: 'aifn-compute/signal/audio', name: 'hzToMel', arity: 1 },
    melToHz: { kind: 'fn', module: 'aifn-compute/signal/audio', name: 'melToHz', arity: 1 },
    isPowerOfTwo: { kind: 'fn', module: 'aifn-compute/foundation/fourier', name: 'isPowerOfTwo', arity: 1 },
    nextPowerOfTwo: { kind: 'fn', module: 'aifn-compute/foundation/fourier', name: 'nextPowerOfTwo', arity: 1 },
  },
}

/** Compat's own rules (compat/controls.tsx): a label naming one of these gets a slider, otherwise a number field. */
const SLIDER_WORDS =
  /\b(time|t\s*\(s\)|frame|playback|scrub|progress|phase|angle|direction|degrees|radians|rotation|orientation|azimuth|threshold|cutoff|probability|fraction|ratio|proportion|prevalence|leakage|split ratio|quantile|percentile|coverage|confidence level|correlation|rho|blend|mix|interpolation)\b/i
/** …and a number field whose label names one of these is an integer. */
const INT_WORDS =
  /\b(seed|draws|steps|iterations|epochs|sample|samples|count|points|terms|layers|depth|clusters|components|neighbours|degree|order|horizon|folds|trees|sweeps|chains|paths|trials|rounds|batches|subsequence)\b/i
const RESERVED = new Set([
  'values',
  'schema',
  'set',
  'handle',
  'bind',
  'reset',
  'isDefault',
  'json',
  'setJson',
  'dropped',
  'attach',
])

// ── Edits ───────────────────────────────────────────────────────────────────────────────────────────────────────────

type Edit = { start: number; end: number; text: string; mark?: boolean }

/** Applies edits; one inside a larger edit is dropped (the larger one already rewrote that text). */
function applyEdits(text: string, edits: Edit[]): string {
  const sorted = edits
    .map((e, i) => ({ ...e, i }))
    .sort(
      (a, b) =>
        a.start - b.start ||
        (a.end - a.start === 0 ? -1 : 0) - (b.end - b.start === 0 ? -1 : 0) ||
        b.end - a.end ||
        // MIGRATE lines go last among insertions at one place: next to the statement they describe.
        Number(!!a.mark) - Number(!!b.mark) ||
        a.i - b.i,
    )
  const kept: Edit[] = []
  let cover: Edit | undefined
  for (const e of sorted) {
    if (cover && e.start >= cover.start && e.end <= cover.end) {
      const touching = e.start === e.end && (e.start === cover.start || e.start === cover.end)
      if (!touching) continue
    }
    if (cover && e.start < cover.end && e.end > cover.end) throw new Error(`overlapping edits at ${e.start}`)
    kept.push(e)
    if (e.end > e.start && (!cover || e.end > cover.end)) cover = e
  }
  let out = text
  for (const e of kept.reverse()) out = out.slice(0, e.start) + e.text + out.slice(e.end)
  return out
}

/** What one file's run did: changes made, items left for a person, and compat/site-maths usages before and after. */
type Report = { file: string; changes: string[]; left: string[]; before: number; after: number; error?: string }

/** State carried across phases: imports to add, and the edits and notes of the phase in hand. */
class Ctx {
  edits: Edit[] = []
  adds = new Map<string, Set<string>>()
  changes: string[] = []
  left: string[] = []
  marks = new Set<string>()
  sf!: SourceFile
  add(module: string, ...names: string[]) {
    const set = this.adds.get(module) ?? new Set()
    for (const n of names) set.add(n)
    this.adds.set(module, set)
  }
  edit(start: number, end: number, text: string) {
    this.edits.push({ start, end, text })
  }
  replace(node: Node, text: string) {
    this.edit(node.getStart(), node.getEnd(), text)
  }
  /** Marks the statement holding `node` with `// MIGRATE: reason` (once). */
  mark(node: Node, reason: string) {
    const stmt = statementOf(node)
    const key = `${stmt.getStart()}:${reason}`
    if (this.marks.has(key)) return
    this.marks.add(key)
    this.left.push(reason)
    const leading = stmt.getFullText().slice(0, stmt.getStart() - stmt.getPos())
    if (leading.includes(`MIGRATE: ${reason}`)) return
    this.edits.push({
      start: stmt.getStart(),
      end: stmt.getStart(),
      text: `// MIGRATE: ${reason}\n${indentOf(stmt)}`,
      mark: true,
    })
  }
}

// ── AST helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────

const isFunction = (n: Node) => Node.isArrowFunction(n) || Node.isFunctionDeclaration(n) || Node.isFunctionExpression(n)
const nearestFunction = (n: Node): Node | undefined => n.getFirstAncestor((a) => isFunction(a))

function statementOf(node: Node): Node {
  let n: Node = node
  while (n.getParent() && !Node.isBlock(n.getParent()!) && !Node.isSourceFile(n.getParent()!)) n = n.getParent()!
  return n
}

function indentOf(node: Node): string {
  const text = node.getSourceFile().getFullText()
  const lineStart = text.lastIndexOf('\n', node.getStart() - 1) + 1
  return /^[ \t]*/.exec(text.slice(lineStart))![0]
}

/** The top-level statement of `body` that holds `node`. */
function topStatement(body: Node, node: Node): Node {
  let n = node
  while (n.getParent() && n.getParent() !== body) n = n.getParent()!
  return n
}

/** React components: PascalCase function declarations and const arrow/function expressions with a block body. */
function components(sf: SourceFile): { fn: Node; body: Node; name: string }[] {
  const out: { fn: Node; body: Node; name: string }[] = []
  for (const f of sf.getFunctions()) {
    const body = f.getBody()
    if (body && /^[A-Z]/.test(f.getName() ?? '')) out.push({ fn: f, body, name: f.getName()! })
  }
  for (const v of sf.getVariableDeclarations()) {
    const init = v.getInitializer()
    if (!init || !/^[A-Z]/.test(v.getName())) continue
    const fn = Node.isCallExpression(init) ? init.getArguments()[0] : init // memo(…), forwardRef(…)
    if (fn && (Node.isArrowFunction(fn) || Node.isFunctionExpression(fn))) {
      const body = fn.getBody()
      if (Node.isBlock(body)) out.push({ fn, body, name: v.getName() })
    }
  }
  return out
}

const tagName = (n: Node): string | undefined =>
  Node.isJsxOpeningElement(n) || Node.isJsxSelfClosingElement(n) ? n.getTagNameNode().getText() : undefined

/** JSX opening/self-closing elements with one of these tag names. */
const elements = (root: Node, names: Set<string> | string) =>
  root.getDescendants().filter((n) => {
    const t = tagName(n)
    return t !== undefined && (typeof names === 'string' ? t === names : names.has(t))
  })

type Attrs = Map<string, { node: Node; init: Node | undefined; text: string }>

/** An element's attributes by name; `undefined` if it has a spread attribute. */
function attrsOf(el: Node): Attrs | undefined {
  const out: Attrs = new Map()
  const attrs = Node.isJsxOpeningElement(el) || Node.isJsxSelfClosingElement(el) ? el.getAttributes() : []
  for (const a of attrs) {
    if (!Node.isJsxAttribute(a)) return undefined
    const init = a.getInitializer()
    let text = 'true'
    if (init && Node.isJsxExpression(init)) text = init.getExpression()?.getText() ?? 'undefined'
    else if (init && Node.isStringLiteral(init)) text = jsString(init.getLiteralValue())
    out.set(a.getNameNode().getText(), {
      node: a,
      init: init && Node.isJsxExpression(init) ? init.getExpression() : init,
      text,
    })
  }
  return out
}

const jsString = (s: string) => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n')}'`

/** A label's static text if it is a string literal (as compat's heuristics saw it), else undefined. */
function staticString(node: Node | undefined): string | undefined {
  if (!node) return undefined
  if (Node.isStringLiteral(node) || Node.isNoSubstitutionTemplateLiteral(node)) return node.getLiteralValue()
  return undefined
}

/** Identifiers in `root` that read a binding called `name` (not property names, keys or JSX attribute names). */
function readsOf(root: Node, name: string): Node[] {
  return root.getDescendantsOfKind(SyntaxKind.Identifier).filter((id) => {
    if (id.getText() !== name) return false
    const p = id.getParent()!
    if (Node.isPropertyAccessExpression(p) && p.getNameNode() === id) return false
    if (Node.isPropertyAssignment(p) && p.getNameNode() === id) return false
    if (Node.isJsxAttribute(p)) return false
    if (Node.isImportSpecifier(p) || Node.isImportClause(p) || Node.isNamespaceImport(p)) return false
    if (Node.isQualifiedName(p) && p.getRight() === id) return false
    if (
      (Node.isVariableDeclaration(p) || Node.isParameterDeclaration(p) || Node.isBindingElement(p)) &&
      p.getNameNode() === id
    )
      return false
    if (
      (Node.isFunctionDeclaration(p) || Node.isTypeAliasDeclaration(p) || Node.isPropertySignature(p)) &&
      p.getNameNode() === id
    )
      return false
    if (Node.isMethodDeclaration(p) || Node.isPropertyDeclaration(p)) return false
    return true
  })
}

/** Declarations of `name` inside `root` (variables, parameters, binding elements, functions). */
function declarationsOf(root: Node, name: string): Node[] {
  return root.getDescendants().filter((n) => {
    if (
      Node.isVariableDeclaration(n) ||
      Node.isParameterDeclaration(n) ||
      Node.isBindingElement(n) ||
      Node.isFunctionDeclaration(n)
    ) {
      const nameNode = n.getNameNode()
      return nameNode !== undefined && Node.isIdentifier(nameNode) && nameNode.getText() === name
    }
    return false
  })
}

/** Is `name` declared anywhere in the file other than by an import? */
const declaredInFile = (sf: SourceFile, name: string) => declarationsOf(sf, name).length > 0

/** Counts compat and site-maths references (not the import specifiers themselves). */
function countLegacy(sf: SourceFile): number {
  const names = new Set<string>()
  for (const imp of sf.getImportDeclarations()) {
    const mod = imp.getModuleSpecifierValue()
    for (const s of imp.getNamedImports()) {
      const imported = s.getName()
      if ((RENDER_MODULES.has(mod) && COMPAT.has(imported)) || SITE_MATHS.test(mod))
        names.add(s.getAliasNode()?.getText() ?? imported)
    }
  }
  let n = 0
  for (const name of names) n += readsOf(sf, name).length
  return n
}

// ── Phase 1: site maths → aifn ──────────────────────────────────────────────────────────────────────────────────────

function phaseMaths(c: Ctx) {
  const sf = c.sf
  for (const imp of sf.getImportDeclarations()) {
    const mod = imp.getModuleSpecifierValue()
    if (!SITE_MATHS.test(mod)) continue
    const rules = MATHS[mod] ?? {}
    const keep: string[] = []
    const removed: Node[] = []
    for (const spec of imp.getNamedImports()) {
      const name = spec.getName()
      const local = spec.getAliasNode()?.getText() ?? name
      const rule = rules[name]
      const ok = rule && !declaredInFile(sf, local) && migrateMathsName(c, rule, local)
      if (ok) removed.push(spec)
      else keep.push(name)
    }
    if (removed.length === 0) {
      if (keep.length)
        c.mark(imp, `no aifn equivalent applied for ${keep.join(', ')} (MIGRATING-NOTES.md, maths table)`)
      continue
    }
    if (keep.length === 0) removeLine(c, imp)
    else {
      const kept = imp.getNamedImports().filter((s) => !removed.includes(s))
      c.replace(
        imp,
        `import ${imp.isTypeOnly() ? 'type ' : ''}{ ${kept.map((s) => s.getText()).join(', ')} } from '${mod}'`,
      )
      c.mark(imp, `no aifn equivalent applied for ${keep.join(', ')} (MIGRATING-NOTES.md, maths table)`)
    }
    c.changes.push(
      `maths: ${imp
        .getNamedImports()
        .filter((s) => removed.includes(s))
        .map((s) => s.getName())
        .join(', ')} → aifn`,
    )
  }
}

/** Rewrites every reference to one site-maths name; false (and nothing written) if any reference is not mechanical. */
function migrateMathsName(c: Ctx, rule: MathRule, local: string): boolean {
  const sf = c.sf
  const refs = readsOf(sf, local)
  const edits: Edit[] = []
  const at = (n: Node, text: string) => edits.push({ start: n.getStart(), end: n.getEnd(), text })
  const free = (name: string) => name === local || !declaredInFile(sf, name)

  if (rule.kind === 'type') {
    if (!free(rule.name)) return false
    for (const r of refs) at(r, rule.name)
    c.add(rule.module, `type ${rule.name}`)
  } else if (rule.kind === 'fn') {
    if (!free(rule.name)) return false
    for (const r of refs) {
      const p = r.getParent()!
      const isCall = Node.isCallExpression(p) && p.getExpression() === r
      if (isCall && (rule.reorder || rule.pack)) {
        const args = p.getArguments().map((a) => a.getText())
        if (args.length !== rule.arity) return false
        at(p, `${rule.name}(${rule.pack ? rule.pack(args) : rule.reorder!.map((i) => args[i]).join(', ')})`)
      } else if (isCall) {
        if (rule.name !== local) at(r, rule.name)
      } else if (rule.arity === 1 && !rule.pack && !Node.isTypeQuery(p)) {
        // An overloaded aifn op passed as a callback infers its last overload (Value): pin it to numbers.
        const call = `(v: number) => ${rule.name}(v)`
        at(r, Node.isShorthandPropertyAssignment(p) ? `${local}: ${call}` : call)
      } else return false
    }
    c.add(rule.module, rule.name)
  } else if (rule.kind === 'linspace') {
    if (!free('toFlat')) return false
    for (const r of refs) {
      const p = r.getParent()!
      if (!(Node.isCallExpression(p) && p.getExpression() === r)) return false
      at(
        p,
        `toFlat(linspace(${p
          .getArguments()
          .map((a) => a.getText())
          .join(', ')}))`,
      )
    }
    if (local !== 'linspace') return false
    c.add('aifn-compute/foundation/tensor', 'linspace', 'toFlat')
  } else if (!migrateRng(c, local, refs, at)) return false
  c.edits.push(...edits)
  return true
}

/**
 * `const g = rng(seed)` → `const g = stream(seed)`, `g.uniform()` → `uniform(g)`, `g.normal()` → `normal(g)`,
 * `ReturnType<typeof rng>` → `Stream`. Bails out unless every binding that holds an rng is one of those forms and no
 * other binding in the file shares its name.
 */
/** The binding an identifier reads (by the binder's scopes), or undefined. */
const bindingOf = (id: Node): Node | undefined => {
  // In `{ x }` the identifier's own symbol is the property; the value it reads is the shorthand's value symbol.
  const p = id.getParent()
  if (p && Node.isShorthandPropertyAssignment(p) && p.getNameNode() === id)
    return p.getValueSymbol()?.getDeclarations()[0]
  return id.getSymbol()?.getDeclarations()[0]
}

/** Is `name` taken in the file by a declaration or by an import from a module other than `except`? */
function taken(sf: SourceFile, name: string, except?: string): boolean {
  if (declaredInFile(sf, name)) return true
  return sf
    .getImportDeclarations()
    .some(
      (i) =>
        i.getModuleSpecifierValue() !== except &&
        i.getNamedImports().some((s) => (s.getAliasNode()?.getText() ?? s.getName()) === name),
    )
}

const RNG_ALIAS: Record<string, string> = {
  stream: 'randomStream',
  uniform: 'drawUniform',
  normal: 'drawNormal',
  Stream: 'RandomStream',
}

/**
 * `const g = rng(seed)` → `const g = stream(seed)`, `g.uniform()` → `uniform(g)`, `g.normal()` → `normal(g)`,
 * `ReturnType<typeof rng>` → `Stream`. Bindings are resolved by scope, so another `g` elsewhere is left alone. Bails
 * out (nothing written) if a binding holding an rng is used in any other way.
 */
function migrateRng(c: Ctx, local: string, refs: Node[], write: (n: Node, t: string) => void): boolean {
  const starts = new Set<number>()
  const at = (n: Node, t: string) => {
    starts.add(n.getStart())
    write(n, t)
  }
  const sf = c.sf
  const RANDOM = 'aifn-compute/foundation/random'
  const name = Object.fromEntries(
    ['stream', 'uniform', 'normal', 'Stream'].map((n) => [n, taken(sf, n, RANDOM) ? RNG_ALIAS[n] : n]),
  )
  if (
    Object.values(name).some(
      (n) => n !== 'stream' && n !== 'uniform' && n !== 'normal' && n !== 'Stream' && taken(sf, n, RANDOM),
    )
  )
    return false
  const typed = new Set([`ReturnType<typeof ${local}>`])
  for (const t of sf.getTypeAliases())
    if (t.getTypeNode()?.getText() === `ReturnType<typeof ${local}>`) typed.add(t.getName())
  const holders = new Set<Node>()
  for (const r of refs) {
    const p = r.getParent()!
    if (Node.isTypeQuery(p)) {
      const ret = p.getParent()
      if (!ret || !Node.isTypeReference(ret) || ret.getTypeName().getText() !== 'ReturnType') return false
      at(ret, name.Stream)
      continue
    }
    if (!(Node.isCallExpression(p) && p.getExpression() === r)) return false
    const holder = p.getParent()!
    if (Node.isVariableDeclaration(holder) && Node.isIdentifier(holder.getNameNode())) {
      holders.add(holder)
      at(r, name.stream)
    } else if (Node.isPropertyAccessExpression(holder) && ['uniform', 'normal'].includes(holder.getName())) {
      const call = holder.getParent()
      if (!Node.isCallExpression(call) || call.getArguments().length) return false
      at(
        call,
        `${name[holder.getName()]}(${name.stream}(${p
          .getArguments()
          .map((a) => a.getText())
          .join(', ')}))`,
      )
    } else if (Node.isCallExpression(holder) && holder.getArguments().includes(p as never)) {
      // Passed straight to a helper: only one declared in this file, whose parameter is typed below.
      const callee = holder.getExpression()
      const decl = Node.isIdentifier(callee) ? bindingOf(callee) : undefined
      if (!decl || Node.isImportSpecifier(decl) || decl.getSourceFile() !== sf) return false
      at(r, name.stream)
    } else return false
  }
  for (const d of sf.getDescendants()) {
    if (!(Node.isParameterDeclaration(d) || Node.isVariableDeclaration(d)) || !Node.isIdentifier(d.getNameNode()))
      continue
    if (typed.has(d.getTypeNode()?.getText() ?? '')) holders.add(d)
  }
  const used = new Set<string>(['stream'])
  for (const h of holders) {
    const hn = (h as never as { getName(): string }).getName()
    for (const id of readsOf(sf, hn)) {
      if (bindingOf(id) !== h) continue
      const p = id.getParent()!
      if (Node.isPropertyAccessExpression(p) && p.getExpression() === id) {
        const call = p.getParent()
        if (!['uniform', 'normal'].includes(p.getName()) || !Node.isCallExpression(call) || call.getArguments().length)
          return false
        at(call, `${name[p.getName()]}(${hn})`)
        used.add(p.getName())
      } else if (Node.isCallExpression(p) && p.getArguments().includes(id as never)) {
        const callee = p.getExpression()
        const decl = Node.isIdentifier(callee) ? bindingOf(callee) : undefined
        if (!decl || Node.isImportSpecifier(decl) || decl.getSourceFile() !== sf) return false
      } else return false
    }
  }
  // Any other zero-argument .uniform()/.normal() is an rng the binder could not follow (a field, an inferred
  // parameter, a structural type): leave the whole file's rng alone.
  const stray = sf.getDescendantsOfKind(SyntaxKind.CallExpression).some((call) => {
    const e = call.getExpression()
    return (
      Node.isPropertyAccessExpression(e) &&
      ['uniform', 'normal'].includes(e.getName()) &&
      !call.getArguments().length &&
      !starts.has(call.getStart())
    )
  })
  if (stray) return false
  if (refs.some((r) => Node.isTypeQuery(r.getParent()!))) used.add('Stream')
  for (const u of refs.map((r) => r.getParent()!.getParent()!).filter((h) => Node.isPropertyAccessExpression(h)))
    used.add((u as never as { getName(): string }).getName())
  for (const u of used) {
    const spec = name[u] === u ? u : `${u} as ${name[u]}`
    c.add(RANDOM, u === 'Stream' ? `type ${spec}` : spec)
  }
  return true
}

// ── Phase 2: useParam / useState + Param* controls → useFigureState ─────────────────────────────────────────────────

type Hook = {
  name: string
  kind: 'param' | 'state'
  stmt: Node
  init: string
  opts?: Record<string, string> // useParam's min/max/step
  setter?: string
  /** `useState<T>`'s T, kept on a choice field so the value keeps its union type. */
  typeArg?: string
  /** The bindings of the value and the setter, for scope-exact references. */
  binding: Node
  setterBinding?: Node
}

function findHooks(body: Node): Map<string, Hook> {
  const hooks = new Map<string, Hook>()
  for (const stmt of body.getChildSyntaxList()?.getChildren() ?? []) {
    if (!Node.isVariableStatement(stmt)) continue
    const decls = stmt.getDeclarations()
    if (decls.length !== 1) continue
    const d = decls[0]
    const init = d.getInitializer()
    if (!init || !Node.isCallExpression(init)) continue
    const callee = init.getExpression().getText()
    const args = init.getArguments()
    const nameNode = d.getNameNode()
    if (
      callee === 'useParam' &&
      Node.isIdentifier(nameNode) &&
      args.length === 2 &&
      Node.isObjectLiteralExpression(args[1])
    ) {
      const opts: Record<string, string> = {}
      let simple = true
      for (const p of args[1].getProperties()) {
        if (Node.isPropertyAssignment(p) && ['min', 'max', 'step'].includes(p.getName()))
          opts[p.getName()] = p.getInitializer()!.getText()
        else if (Node.isShorthandPropertyAssignment(p) && ['min', 'max', 'step'].includes(p.getName()))
          opts[p.getName()] = p.getName()
        else simple = false
      }
      if (simple && opts.min !== undefined && opts.max !== undefined)
        hooks.set(nameNode.getText(), {
          name: nameNode.getText(),
          kind: 'param',
          stmt,
          init: args[0].getText(),
          opts,
          binding: d,
        })
    } else if (callee === 'useState' && Node.isArrayBindingPattern(nameNode) && args.length === 1) {
      const [v, s] = nameNode.getElements()
      if (
        v &&
        s &&
        Node.isBindingElement(v) &&
        Node.isBindingElement(s) &&
        Node.isIdentifier(v.getNameNode()) &&
        Node.isIdentifier(s.getNameNode()) &&
        nameNode.getElements().length === 2
      )
        hooks.set(v.getName(), {
          name: v.getName(),
          kind: 'state',
          stmt,
          init: args[0].getText(),
          setter: s.getName(),
          typeArg: init.getTypeArguments()[0]?.getText(),
          binding: v,
          setterBinding: s,
        })
    }
  }
  return hooks
}

type Field = { key: string; def: string; hook: Hook; control?: Node; valueType: string; when?: Node }

function phaseState(c: Ctx) {
  const CONTROLS = new Set(['ParamSlider', 'ParamChoice', 'ParamSwitch', 'ParamButton'])
  for (const { fn, body, name } of components(c.sf)) {
    const controls = elements(body, CONTROLS).filter((e) => nearestFunction(e) === fn)
    const hooks = findHooks(body)
    const params = [...hooks.values()].filter((h) => h.kind === 'param')
    // Plain buttons become Buttons whatever the figure; only value controls and useParams need a state.
    if (!controls.some((e) => tagName(e) !== 'ParamButton') && !params.length) continue
    const figures = elements(body, new Set(['Interactive', 'Figure'])).filter((e) => nearestFunction(e) === fn)
    if (figures.length !== 1) {
      c.mark(
        controls[0] ?? params[0].stmt,
        `${name}: ${figures.length} figures in one component; give each Figure its own useFigureState`,
      )
      continue
    }
    const fig = figures[0]
    const figAttrs = attrsOf(fig)
    if (!figAttrs) continue
    if (figAttrs.has('state')) {
      // Converted on an earlier run: give the controls left behind the same reasons that run gave.
      const slotNode = figAttrs.get('controls')?.init
      for (const el of controls) {
        const inside = slotNode && el.getStart() >= slotNode.getStart() && el.getEnd() <= slotNode.getEnd()
        if (!inside) {
          if (tagName(el) !== 'ParamButton') c.mark(el, `${tagName(el)} outside the controls slot`)
          continue
        }
        const f = fieldOf(el, hooks)
        if (typeof f === 'string') {
          if (f) c.mark(el, f)
        } else if (gateOf(f.control!, slotNode) === undefined)
          c.mark(el, `${tagName(el)} inside an expression other than {cond && …}`)
      }
      continue
    }
    const controlsAttr = figAttrs.get('controls')
    // `state` and the builders must not shadow (or be shadowed by) anything in the file.
    const shadowed = ['state', 'useFigureState', 'slider', 'int', 'float', 'choice', 'setting'].find(
      (n) =>
        declarationsOf(c.sf, n).some(
          (d) => !(Node.isVariableDeclaration(d) && /^useFigureState\(/.test(d.getInitializer()?.getText() ?? '')),
        ) ||
        c.sf
          .getImportDeclarations()
          .some(
            (i) =>
              i.getModuleSpecifierValue() !== 'aifn-render' &&
              i.getNamedImports().some((x) => (x.getAliasNode()?.getText() ?? x.getName()) === n),
          ),
    )
    if (shadowed) {
      c.mark(fig, `${name}: ${shadowed} is a name in this file already; rename it, then rerun`)
      continue
    }
    if (declarationsOf(fn, 'state').length) {
      c.mark(fig, `${name}: a binding named state already exists; convert the controls by hand`)
      continue
    }
    const fields: Field[] = []
    const used = new Map<string, number>()
    const slot = controlsAttr?.init
    const inSlot = slot ? controls.filter((e) => e.getStart() >= slot.getStart() && e.getEnd() <= slot.getEnd()) : []
    for (const el of controls)
      if (!inSlot.includes(el) && tagName(el) !== 'ParamButton') c.mark(el, `${tagName(el)} outside the controls slot`)
    for (const el of inSlot) {
      const f = fieldOf(el, hooks)
      if (typeof f === 'string') {
        if (f) c.mark(el, f)
        continue
      }
      // `{cond && <ParamSlider …/>}`: the field shows only when cond holds; the whole `{…}` goes.
      const gate = gateOf(f.control!, slot!)
      if (gate === undefined) {
        c.mark(el, `${tagName(el)} inside an expression other than {cond && …}`)
        continue
      }
      if (gate) {
        f.when = gate.cond
        f.control = gate.node
      }
      used.set(f.key, (used.get(f.key) ?? 0) + 1)
      fields.push(f)
    }
    // A useParam no control shows is moved by a handle or a click: an on-chart field.
    for (const h of params) {
      if (used.has(h.name) || controls.some((el) => readsOf(el, h.name).length)) continue
      const extra = [...(h.opts!.step ? [`step: ${h.opts!.step}`] : []), 'onChart: true']
      fields.push({
        key: h.name,
        def: `slider(${h.opts!.min}, ${h.opts!.max}, ${h.init}, { ${extra.join(', ')} })`,
        hook: h,
        valueType: 'number',
      })
      used.set(h.name, 1)
    }
    const ok = fields.filter((f) => used.get(f.key) === 1)
    for (const f of fields)
      if (used.get(f.key)! > 1) c.mark(f.control ?? f.hook.stmt, `${f.key} is bound to several controls`)
    if (!ok.length) continue

    // Field expressions may only read module values and locals declared before the state.
    const first = Math.min(...ok.map((f) => f.hook.stmt.getStart()))
    const converted = new Set(ok.flatMap((f) => [f.key, f.hook.setter ?? '']))
    const locals = new Map<string, number>()
    for (const s of body.getChildSyntaxList()!.getChildren()) {
      if (!Node.isVariableStatement(s)) continue
      for (const d of s.getDeclarations()) {
        const n = d.getNameNode()
        for (const id of Node.isIdentifier(n) ? [n] : n.getDescendantsOfKind(SyntaxKind.Identifier))
          locals.set(id.getText(), s.getStart())
      }
    }
    const idsIn = (def: string) =>
      def
        .replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"/g, '')
        .replace(/\b[A-Za-z_$][\w$]*\s*:(?!:)/g, '')
        .replace(/\.\s*[A-Za-z_$][\w$]*/g, '')
        .match(/[A-Za-z_$][\w$]*/g) ?? []
    const later = (id: string) => locals.has(id) && locals.get(id)! >= first && !converted.has(id)
    const badField = ok.find(
      (f) =>
        idsIn(f.def).some((id) => converted.has(id) || later(id)) || (f.when && idsIn(f.when.getText()).some(later)),
    )
    if (badField) {
      c.mark(
        badField.control ?? badField.hook.stmt,
        `${name}: field ${badField.key} depends on component values; declare the schema by hand`,
      )
      continue
    }
    // A gate reading converted values becomes a closure over the set's values: `(v) => v.mode === 'x'`.
    for (const f of ok) {
      if (!f.when) continue
      const cond = f.when
      const edits: Edit[] = []
      for (const g of ok)
        for (const id of [...readsOf(cond, g.key)].filter((r) => bindingOf(r) === g.hook.binding))
          edits.push({ start: id.getStart() - cond.getStart(), end: id.getEnd() - cond.getStart(), text: `v.${g.key}` })
      const text = applyEdits(cond.getText(), edits)
      f.def = withOption(f.def, edits.length ? `when: (v) => ${text}` : `when: () => ${text}`)
    }
    const removed = new Set(ok.flatMap((f) => (f.control ? [f.control] : [])))
    const refEdits = rewriteRefs(fn, ok, removed)
    if (typeof refEdits === 'string') {
      c.mark(fig, `${name}: ${refEdits}`)
      continue
    }
    c.edits.push(...refEdits)
    // The schema replaces the first converted hook; the others go.
    const ind = indentOf(ok[0].hook.stmt)
    const schema = `const state = useFigureState({\n${ok.map((f) => `${ind}  ${f.key}: ${f.def},`).join('\n')}\n${ind}})`
    const stmts = [...new Set(ok.map((f) => f.hook.stmt))].sort((a, b) => a.getStart() - b.getStart())
    stmts.forEach((s, i) => (i === 0 ? c.replace(s, schema) : removeLine(c, s)))
    if (controlsAttr && removed.size) removeControls(c, controlsAttr, removed)
    const title = figAttrs.get('title')
    const after = (title ?? [...figAttrs.values()][0]).node
    c.edit(after.getEnd(), after.getEnd(), ' state={state}')
    const builder = (f: Field) => /^\w+/.exec(f.def)![0]
    const builders = new Set(ok.map(builder))
    c.add('aifn-render', 'useFigureState', ...builders)
    c.changes.push(`state: ${ok.map((f) => `${f.key}:${builder(f)}`).join(', ')}`)
  }
}

/**
 * How a control sits in the controls slot: null when it is a plain child, `{ cond, node }` for `{cond && <X/>}` (node
 * is the `{…}` to remove), undefined for anything else (a ternary, a call).
 */
function gateOf(control: Node, slot: Node): { cond: Node; node: Node } | null | undefined {
  let n = control
  while (Node.isParenthesizedExpression(n.getParent()!)) n = n.getParent()!
  const p = n.getParent()!
  let gate: { cond: Node; node: Node } | null = null
  let top = n
  if (Node.isBinaryExpression(p) && p.getOperatorToken().getText() === '&&' && p.getRight() === n) {
    const jx = p.getParent()
    if (!jx || !Node.isJsxExpression(jx)) return undefined
    gate = { cond: p.getLeft(), node: jx }
    top = jx
  }
  // From there up to the slot, only JSX nesting: a branch of a ternary, a call or a map is not a place a row can go.
  for (let a = top; a !== slot; a = a.getParent()!) {
    const up = a.getParent()
    if (!up) return undefined
    const plain =
      Node.isJsxElement(up) ||
      Node.isJsxFragment(up) ||
      Node.isParenthesizedExpression(up) ||
      (Node.isJsxExpression(up) && up === slot.getParent())
    if (!plain && up !== slot) return undefined
    if (up === slot.getParent()) break
  }
  return gate
}

/** Adds `opt` to a field builder call's options (`slider(…, { … })`, `setting(x, 'label')`, `choice(o, x)`). */
function withOption(def: string, opt: string): string {
  if (def.endsWith(' })')) return `${def.slice(0, -3)}, ${opt} })`
  const m = /^(setting|toggle)\((.*?), ((?:'(?:[^'\\]|\\.)*')|[^,]+)\)$/.exec(def)
  if (m) return `${m[1]}(${m[2]}, { label: ${m[3]}, ${opt} })`
  return `${def.slice(0, -1)}, { ${opt} })`
}

/** The field for one control, '' to leave it silently (a plain button), or the reason it is left. */
function fieldOf(el: Node, hooks: Map<string, Hook>): Field | string {
  const tag = tagName(el)!
  const a = attrsOf(el)
  if (!a) return `${tag} with spread props`
  const label = a.get('label')
  const labelText = label?.text
  if (tag === 'ParamButton') {
    // `<ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>`: the seed becomes an int field.
    const click = a.get('onClick')?.init
    const m =
      click && /^\(\)\s*=>\s*(\w+)\(\(?(\w+)\)?\s*=>\s*\2\s*\+\s*1\)$/.exec(click.getText().replace(/\s+/g, ' '))
    const hook = m && [...hooks.values()].find((h) => h.setter === m[1])
    if (!hook || !/^\d+$/.test(hook.init)) return ''
    const setterUses = readsOf(nearestFunction(el)!, hook.setter!).length
    if (setterUses !== 1) return ''
    const node = Node.isJsxOpeningElement(el) ? el.getParent()! : el
    return {
      key: hook.name,
      def: `int(${hook.init}, { ge: 0, label: 'seed' })`,
      hook,
      control: node,
      valueType: 'number',
    }
  }
  const unsupported = [...a.keys()].find((k) => ['disabled', 'key'].includes(k))
  if (unsupported) return `${tag} with ${unsupported}: no field equivalent`
  const node = Node.isJsxOpeningElement(el) ? el.getParent()! : el
  const hookFor = (valueAttr: string) => {
    const v = a.get(valueAttr)?.init
    const on = a.get('onChange')?.init
    if (!v || !on || !Node.isIdentifier(v) || !Node.isIdentifier(on)) return undefined
    const h = hooks.get(v.getText())
    return h?.kind === 'state' && h.setter === on.getText() ? h : undefined
  }
  if (tag === 'ParamChoice') {
    const h = hookFor('value')
    const options = a.get('options')
    if (!h || !options) return 'ParamChoice not bound to a local useState'
    const t = h.typeArg ? `<${h.typeArg}>` : ''
    return {
      key: h.name,
      def: `choice${t}(${options.text}, ${h.init}${labelText ? `, { label: ${labelText} }` : ''})`,
      hook: h,
      control: node,
      valueType: `typeof state.${h.name}`,
    }
  }
  if (tag === 'ParamSwitch') {
    const h = hookFor('checked')
    if (!h) return 'ParamSwitch not bound to a local useState'
    return {
      key: h.name,
      def: `setting(${h.init}${labelText ? `, ${labelText}` : ''})`,
      hook: h,
      control: node,
      valueType: 'boolean',
    }
  }
  // ParamSlider
  let h: Hook | undefined
  let min: string | undefined, max: string | undefined, step: string | undefined
  const param = a.get('param')?.init
  if (param && Node.isIdentifier(param) && hooks.get(param.getText())?.kind === 'param') {
    h = hooks.get(param.getText())!
    ;({ min, max, step } = h.opts!)
  } else {
    h = hookFor('value')
    min = a.get('min')?.text
    max = a.get('max')?.text
    step = a.get('step')?.text
  }
  if (!h) return 'ParamSlider not bound to a local useParam/useState'
  if (RESERVED.has(h.name)) return `${h.name} is a reserved field name`
  const known = new Set([
    'label',
    'param',
    'value',
    'onChange',
    'min',
    'max',
    'step',
    'format',
    'withArrows',
    'debounceMs',
    'className',
    'slider',
    'variant',
    'type',
    'scale',
    'spacing',
    'increment',
    'points_per_decade',
    'points',
    'logTransform',
    'headerValue',
    'suggestions',
  ])
  const odd = [...a.keys()].find((k) => !known.has(k))
  if (odd) return `ParamSlider prop ${odd} has no field equivalent`
  const lit = staticString(label?.init)
  const sliderProp = a.get('slider')?.text
  const variant = a.get('variant')?.text
  const asSlider =
    sliderProp === 'true' || variant === "'slider'"
      ? true
      : sliderProp === 'false' || variant === "'field'"
        ? false
        : lit !== undefined && SLIDER_WORDS.test(lit)
  const opt = (k: string, v: string | undefined) => (v === undefined ? [] : [`${k}: ${v}`])
  if (asSlider) {
    const extra = [
      ...opt('step', step),
      ...opt('label', labelText),
      ...opt('format', a.get('format')?.text),
      ...(a.get('withArrows')?.text === 'false' ? ['steppable: false'] : []),
    ]
    return {
      key: h.name,
      def: `slider(${min ?? '0'}, ${max ?? '100'}, ${h.init}${extra.length ? `, { ${extra.join(', ')} }` : ''})`,
      hook: h,
      control: node,
      valueType: 'number',
    }
  }
  const typeAttr = a.get('type')?.text
  // Compat's label words, or a field stepping by whole numbers from a whole number (a count, a size, a length).
  const whole = (v: string | undefined) => v !== undefined && /^\d+$/.test(v)
  const bound = (v: string | undefined) => v === undefined || /^-?\d+$/.test(v) || /^[A-Z][A-Z0-9_]*$/.test(v)
  const isInt =
    typeAttr === "'int'" ||
    (typeAttr === undefined &&
      ((lit !== undefined && INT_WORDS.test(lit)) || (whole(h.init) && whole(step) && bound(min) && bound(max))))
  let logTransform = a.get('logTransform')?.text
  let ppd = a.get('points_per_decade')?.text
  const format = a.get('format')?.text
  if (logTransform === undefined && format && /10\s*\*\*/.test(format)) {
    logTransform = "'value-is-log'"
    ppd ??= '2'
  }
  const extra = [
    ...opt('min', min),
    ...opt('max', max),
    ...opt('step', step),
    ...opt('label', labelText),
    ...['scale', 'spacing', 'increment', 'points', 'suggestions', 'headerValue'].flatMap((k) => opt(k, a.get(k)?.text)),
    ...opt('points_per_decade', ppd),
    ...opt('logTransform', logTransform),
    ...opt('format', format),
  ]
  return {
    key: h.name,
    def: `${isInt ? 'int' : 'float'}(${h.init}${extra.length ? `, { ${extra.join(', ')} }` : ''})`,
    hook: h,
    control: node,
    valueType: 'number',
  }
}

/** Rewrites reads of converted hooks to the state; a string says why it cannot. */
function rewriteRefs(fn: Node, fields: Field[], removed: Set<Node>): Edit[] | string {
  const edits: Edit[] = []
  const at = (n: Node, text: string) => edits.push({ start: n.getStart(), end: n.getEnd(), text })
  const inRemoved = (n: Node) => [...removed].some((r) => n.getStart() >= r.getStart() && n.getEnd() <= r.getEnd())
  for (const f of fields) {
    const k = f.key
    const reads = readsOf(fn, k).filter((r) => !inRemoved(r) && bindingOf(r) === f.hook.binding)
    for (const r of reads) {
      const p = r.getParent()!
      if (f.hook.kind === 'param') {
        if (Node.isPropertyAccessExpression(p) && p.getExpression() === r) {
          const prop = p.getName()
          const call = p.getParent()
          if (prop === 'value') at(p, `state.${k}`)
          else if (prop === 'set' && Node.isCallExpression(call) && call.getExpression() === p)
            at(
              call,
              `state.set('${k}', ${call
                .getArguments()
                .map((x) => x.getText())
                .join(', ')})`,
            )
          else if (prop === 'set') at(p, `(v: number) => state.set('${k}', v)`)
          else at(r, `state.bind('${k}')`)
        } else if (Node.isShorthandPropertyAssignment(p)) at(p, `${k}: state.bind('${k}')`)
        else at(r, `state.bind('${k}')`)
      } else if (Node.isShorthandPropertyAssignment(p)) at(p, `${k}: state.${k}`)
      else at(r, `state.${k}`)
    }
    if (f.hook.setter) {
      for (const r of readsOf(fn, f.hook.setter).filter(
        (r) => !inRemoved(r) && bindingOf(r) === f.hook.setterBinding,
      )) {
        const p = r.getParent()!
        if (Node.isCallExpression(p) && p.getExpression() === r) {
          const arg = p.getArguments()[0]
          if (!arg || Node.isArrowFunction(arg) || Node.isFunctionExpression(arg))
            return `${f.hook.setter} takes an update function`
          at(p, `state.set('${k}', ${arg.getText()})`)
        } else at(r, `(v: ${f.valueType}) => state.set('${k}', v)`)
      }
    }
  }
  return edits
}

function removeLine(c: Ctx, node: Node) {
  const text = c.sf.getFullText()
  const lineStart = text.lastIndexOf('\n', node.getStart() - 1) + 1
  const lineEnd = text.indexOf('\n', node.getEnd())
  const onlyNode =
    text.slice(lineStart, node.getStart()).trim() === '' &&
    text.slice(node.getEnd(), lineEnd === -1 ? text.length : lineEnd).trim() === ''
  if (onlyNode) c.edit(lineStart, lineEnd === -1 ? text.length : lineEnd + 1, '')
  else c.replace(node, '')
}

/** Removes converted controls, then wrappers left empty; the whole attribute if nothing is left. */
function removeControls(c: Ctx, attr: { node: Node; init: Node | undefined }, removed: Set<Node>) {
  const empty = (n: Node): boolean => {
    if (removed.has(n)) return true
    if (Node.isJsxText(n)) return n.getText().trim() === ''
    if (Node.isJsxElement(n) || Node.isJsxFragment(n)) return n.getJsxChildren().every(empty)
    if (Node.isParenthesizedExpression(n)) return empty(n.getExpression())
    return false
  }
  if (attr.init && empty(attr.init)) {
    c.replace(attr.node, '')
    return
  }
  const visit = (n: Node) => {
    if (removed.has(n) || ((Node.isJsxElement(n) || Node.isJsxFragment(n)) && n !== attr.init && empty(n))) {
      removeLine(c, n)
      return
    }
    n.forEachChild(visit)
  }
  if (attr.init) visit(attr.init)
}

// ── Phase 3: XYChart / Heatmap → Plot + layers ─────────────────────────────────────────────────────────────────────

const LAYER_KEYS: Record<string, Set<string>> = {
  Curve: new Set(['name', 'x', 'y', 'slot', 'dashed', 'emphasis', 'muted', 'thin', 'color', 'showPoints', 'id']),
  Area: new Set(['name', 'x', 'y', 'slot', 'emphasis', 'muted', 'color', 'id']),
  Points: new Set(['name', 'x', 'y', 'slot', 'emphasis', 'muted', 'color', 'group', 'groupNames', 'colors', 'id']),
  Bars: new Set(['name', 'x', 'y', 'slot', 'emphasis', 'muted', 'color', 'colors', 'id']),
}

/** JSX attributes for an object literal's properties (`k: v` → `k={v}`, `k: true` → `k`, `'s'` → `"s"`). */
function objectAttrs(o: Node, drop: Node[] = [], rename = new Map<Node, string>()): string[] {
  return (o as never as { getProperties(): Node[] }).getProperties().flatMap((p) => {
    if (drop.includes(p)) return []
    const key = (p as never as { getName(): string }).getName()
    const name = rename.get(p) ?? key
    const v = Node.isPropertyAssignment(p) ? p.getInitializer()! : undefined
    if (!v) return [`${name}={${key}}`]
    if (Node.isStringLiteral(v) && !/["{}\\]/.test(v.getLiteralValue())) return [`${name}="${v.getLiteralValue()}"`]
    if (v.getText() === 'true') return [name]
    return [`${name}={${v.getText()}}`]
  })
}

/** The layer for one series object literal, or the reason it has none. */
function layerKind(obj: Node): { tag: string; drop: Node[]; rename: Map<Node, string> } | string {
  if (!Node.isObjectLiteralExpression(obj)) return 'a series that is not an object literal'
  let type: string | undefined
  let area = false
  const drop: Node[] = []
  const rename = new Map<Node, string>()
  for (const p of obj.getProperties()) {
    if (!(Node.isPropertyAssignment(p) || Node.isShorthandPropertyAssignment(p)))
      return 'a series with spread or methods'
    const k = p.getName()
    const v = Node.isPropertyAssignment(p) ? p.getInitializer()! : undefined
    if (k === 'type') {
      if (!v || !Node.isStringLiteral(v)) return 'a series whose type is computed'
      type = v.getLiteralValue()
      drop.push(p)
    } else if (k === 'area') {
      if (v?.getText() === 'true') area = true
      else if (v?.getText() !== 'false') return 'a series whose area flag is computed'
      drop.push(p)
    } else if (k === 'pointColors') rename.set(p, 'colors')
  }
  const tag =
    type === 'line' ? (area ? 'Area' : 'Curve') : type === 'scatter' ? 'Points' : type === 'bar' ? 'Bars' : undefined
  if (!tag) return `a series of type ${type ?? '(none)'}`
  for (const p of (obj as never as { getProperties(): Node[] }).getProperties()) {
    const k = (p as never as { getName(): string }).getName()
    const key = rename.get(p) ?? k
    if (drop.includes(p)) continue
    if (!LAYER_KEYS[tag].has(key)) return `series key ${k} on a ${tag}`
  }
  return { tag, drop, rename }
}

const unwrap = (n: Node): Node =>
  Node.isParenthesizedExpression(n) || Node.isAsExpression(n) ? unwrap(n.getExpression()) : n

/**
 * `result.series` where `const result = useMemo(() => { …; const series = [ … ]; return { series, … } }, deps)`: the
 * inner const is the source, spread as `result.series[i]`.
 */
function memoMemberSource(e: Node, body: Node): { array: Node; decl: Node; ref: string } | string | undefined {
  let outer: Node | undefined
  let prop: string
  let ref: string
  if (Node.isPropertyAccessExpression(e) && Node.isIdentifier(e.getExpression())) {
    outer = bindingOf(e.getExpression())
    prop = e.getName()
    ref = e.getText()
  } else if (Node.isIdentifier(e)) {
    // `const { series, … } = useMemo(…)`
    const b = bindingOf(e)
    if (!b || !Node.isBindingElement(b) || !Node.isObjectBindingPattern(b.getParent()!)) return undefined
    outer = b.getParent()!.getParent()
    prop = b.getPropertyNameNode()?.getText() ?? b.getName()
    ref = e.getText()
  } else return undefined
  const init = outer && Node.isVariableDeclaration(outer) ? outer.getInitializer() : undefined
  const fn =
    init && Node.isCallExpression(init) && init.getExpression().getText() === 'useMemo'
      ? init.getArguments()[0]
      : undefined
  if (!fn || !Node.isArrowFunction(fn) || !Node.isBlock(fn.getBody()) || !body.containsRange(fn.getPos(), fn.getEnd()))
    return 'series built by an expression'
  const block = fn.getBody() as never as { getStatements(): Node[] }
  const inner = block
    .getStatements()
    .flatMap((st) => (Node.isVariableStatement(st) ? st.getDeclarations() : []))
    .find((d) => d.getName() === prop)
  const arr = inner?.getInitializer() && unwrap(inner.getInitializer()!)
  if (!inner || !arr || !Node.isArrayLiteralExpression(arr)) return 'series built by an expression'
  const mutated = readsOf(fn, prop).some((r) => {
    const p = r.getParent()!
    return (
      bindingOf(r) === inner &&
      Node.isPropertyAccessExpression(p) &&
      /^(push|unshift|splice|pop|shift|sort|reverse|fill)$/.test(p.getName())
    )
  })
  if (mutated) return 'series array is mutated after it is built'
  return { array: arr, decl: inner, ref }
}

/** The array literal a series expression comes from, for a literal, a const array, or a useMemo returning one. */
function seriesSource(expr: Node, body: Node): { array: Node; memo?: Node; decl?: Node; ref?: string } | string {
  const e = unwrap(expr)
  if (Node.isArrayLiteralExpression(e)) return { array: e }
  const member = memoMemberSource(e, body)
  if (member) return member
  if (!Node.isIdentifier(e)) return 'series built by an expression'
  const decls = declarationsOf(body, e.getText()).concat(
    declarationsOf(e.getSourceFile(), e.getText()).filter((d) => !body.containsRange(d.getPos(), d.getEnd())),
  )
  if (decls.length !== 1 || !Node.isVariableDeclaration(decls[0]))
    return 'series declared more than once or not as a const'
  const decl = decls[0]
  const init = decl.getInitializer()
  if (!init) return 'series without an initialiser'
  const mutated = readsOf(e.getSourceFile(), e.getText()).some((r) => {
    const p = r.getParent()!
    return (
      bindingOf(r) === decl &&
      Node.isPropertyAccessExpression(p) &&
      /^(push|unshift|splice|pop|shift|sort|reverse|fill)$/.test(p.getName())
    )
  })
  if (mutated) return 'series array is mutated after it is built'
  const i = unwrap(init)
  if (Node.isArrayLiteralExpression(i)) return { array: i, decl }
  if (Node.isCallExpression(i) && i.getExpression().getText() === 'useMemo') {
    const fn = i.getArguments()[0]
    if (!fn || !Node.isArrowFunction(fn)) return 'series memo is not an arrow function'
    const b = fn.getBody()
    if (Node.isBlock(b)) {
      const returns = b.getDescendantsOfKind(SyntaxKind.ReturnStatement).filter((r) => nearestFunction(r) === fn)
      const last = b.getStatements().at(-1)
      if (returns.length !== 1 || returns[0] !== last) return 'series memo with several returns'
      const r = returns[0].getExpression()
      if (r && Node.isArrayLiteralExpression(unwrap(r))) return { array: unwrap(r), memo: fn, decl }
      return 'series memo does not return an array literal'
    }
    if (Node.isArrayLiteralExpression(unwrap(b))) return { array: unwrap(b), memo: fn, decl }
  }
  return 'series not a literal or a memoised array literal'
}

/** Is the series source annotated with a series type (`: XYSeries[]`, `(): XYSeries[] =>`, `useMemo<XYSeries[]>`)? */
function hasSeriesType(src: { memo?: Node; decl?: Node }): boolean {
  const texts = [
    src.decl && Node.isVariableDeclaration(src.decl) ? src.decl.getTypeNode()?.getText() : undefined,
    src.memo && Node.isArrowFunction(src.memo) ? src.memo.getReturnTypeNode()?.getText() : undefined,
    src.memo?.getParent() && Node.isCallExpression(src.memo.getParent()!)
      ? (src.memo.getParent() as never as { getTypeArguments(): Node[] }).getTypeArguments()[0]?.getText()
      : undefined,
  ]
  return texts.some((t) => !!t)
}

/**
 * Layers for a series prop. Literal arrays become one element per series with attributes; memoised arrays keep their
 * memo (types dropped, `as const`) and become `<Curve {...series[0]} />`.
 */
function seriesLayers(c: Ctx, expr: Node, body: Node, live: boolean, done: Set<Node>): string[] | string {
  const src = seriesSource(expr, body)
  if (typeof src === 'string') return src
  const elems = (src.array as never as { getElements(): Node[] }).getElements()
  const kinds = elems.map(layerKind)
  const bad = kinds.find((k) => typeof k === 'string')
  if (bad) {
    // The list goes to seriesLayers as it is; a const or memo literal needs `as const` to keep each `type` literal.
    const parent = src.array.getParent()!
    if (
      src.decl &&
      !(Node.isAsExpression(parent) && parent.getTypeNode()?.getText() === 'const') &&
      !hasSeriesType(src)
    )
      c.edit(src.array.getEnd(), src.array.getEnd(), ' as const')
    return bad as string
  }
  const liveAttr = live ? ' live' : ''
  if (!src.decl) {
    return elems.map((o, i) => {
      const k = kinds[i] as Exclude<ReturnType<typeof layerKind>, string>
      return `<${k.tag} ${objectAttrs(o, k.drop, k.rename).join(' ')}${liveAttr} />`
    })
  }
  // Memoised (or module const): edit the source once, spread each element.
  if (!done.has(src.array)) {
    done.add(src.array)
    elems.forEach((_, i) => {
      const k = kinds[i] as Exclude<ReturnType<typeof layerKind>, string>
      for (const p of k.drop) {
        const text = c.sf.getFullText()
        let end = p.getEnd()
        while (/[\s,]/.test(text[end]) && text[end] !== '\n') end++
        if (text[end] === '\n') end++
        const lineStart = text.lastIndexOf('\n', p.getStart() - 1) + 1
        const start = text.slice(lineStart, p.getStart()).trim() === '' ? lineStart : p.getStart()
        c.edit(start, end, '')
      }
      for (const [p, to] of k.rename)
        c.edit(
          (p as never as { getNameNode(): Node }).getNameNode().getStart(),
          (p as never as { getNameNode(): Node }).getNameNode().getEnd(),
          to,
        )
    })
    const parent = src.array.getParent()!
    if (!(Node.isAsExpression(parent) && parent.getTypeNode()?.getText() === 'const'))
      c.edit(src.array.getEnd(), src.array.getEnd(), ' as const')
    const call = src.memo?.getParent()
    if (call && Node.isCallExpression(call) && call.getTypeArguments().length) {
      const args = call.getTypeArguments()
      c.edit(args[0].getStart() - 1, args.at(-1)!.getEnd() + 1, '')
    }
    if (src.memo && Node.isArrowFunction(src.memo) && src.memo.getReturnTypeNode()) {
      const rt = src.memo.getReturnTypeNode()!
      const colon = c.sf.getFullText().lastIndexOf(':', rt.getStart())
      c.edit(colon, rt.getEnd(), '')
    }
    if (src.decl && Node.isVariableDeclaration(src.decl) && src.decl.getTypeNode()) {
      const tn = src.decl.getTypeNode()!
      c.edit(src.decl.getNameNode().getEnd(), tn.getEnd(), '')
    }
  }
  const id = src.ref ?? (src.decl as never as { getName(): string }).getName()
  return elems.map((_, i) => `<${(kinds[i] as { tag: string }).tag} {...${id}[${i}]}${liveAttr} />`)
}

/**
 * One handle as a layer: a state-bound handle (`at: state.k`, `onDrag: (v) => state.set('k', v)`) becomes
 * `<Handle {...state.handle('k', { label })} />`; any other plain object becomes attributes.
 */
function handleLayer(o: Node): string {
  const props = new Map<string, string>()
  for (const p of (o as never as { getProperties(): Node[] }).getProperties())
    props.set(
      (p as never as { getName(): string }).getName(),
      Node.isPropertyAssignment(p) ? p.getInitializer()!.getText() : (p as never as { getName(): string }).getName(),
    )
  const kind = props.get('kind')?.replace(/"/g, "'")
  const setter = (v: string | undefined) =>
    /^\((\w+)(?:: number)?\) => state\.set\('(\w+)', \1\)$/.exec(v ?? '')?.[2] ??
    /^state\.bind\('(\w+)'\)\.set$/.exec(v ?? '')?.[1]
  const key = setter(props.get('onDrag'))
  const extras = [...props.keys()].filter((k) => !['kind', 'at', 'onDrag', 'label', 'onRelease'].includes(k))
  if (key && (kind === "'x'" || kind === "'y'") && props.get('at') === `state.${key}` && !extras.length) {
    const opts = [
      ...(kind === "'y'" ? ["axis: 'y'"] : []),
      ...['label', 'onRelease'].flatMap((k) => (props.has(k) ? [`${k}: ${props.get(k)}`] : [])),
    ]
    return `<Handle {...state.handle('${key}'${opts.length ? `, { ${opts.join(', ')} }` : ''})} />`
  }
  return `<Handle ${objectAttrs(o).join(' ')} />`
}

/** Layers for a chart's `handles`: inline, or from a const array literal that only charts read (then removed). */
function handleLayers(c: Ctx, a: Attrs): string[] {
  const h = a.get('handles')
  if (!h?.init) return []
  const plain = (e: Node) =>
    Node.isObjectLiteralExpression(e) &&
    e.getProperties().every((p) => Node.isPropertyAssignment(p) || Node.isShorthandPropertyAssignment(p))
  let arr: Node | undefined = h.init
  let decl: Node | undefined
  if (Node.isIdentifier(h.init)) {
    decl = bindingOf(h.init)
    const init = decl && Node.isVariableDeclaration(decl) ? decl.getInitializer() : undefined
    const reads = decl ? readsOf(c.sf, h.init.getText()).filter((r) => bindingOf(r) === decl) : []
    const onlyCharts = reads.every((r) => r.getFirstAncestor((x) => ['XYChart', 'Heatmap'].includes(tagName(x) ?? '')))
    arr = init && onlyCharts && Node.isVariableStatement(decl!.getParent()!.getParent()!) ? unwrap(init) : undefined
  }
  if (arr && Node.isArrayLiteralExpression(arr) && arr.getElements().every(plain)) {
    if (decl) removeLine(c, decl.getParent()!.getParent()!)
    return arr.getElements().map(handleLayer)
  }
  // `?? []`: the prop was optional, so the list may be undefined (a handle shown only in some states).
  const maybe = Node.isIdentifier(h.init) || Node.isPropertyAccessExpression(h.init) || Node.isCallExpression(h.init)
  return [`{(${h.text}${maybe ? ' ?? []' : ''}).map((h, i) => <Handle key={i} {...h} />)}`]
}

const CHART_NAMES = [
  'Plot',
  'useAxis',
  'Curve',
  'Points',
  'Bars',
  'Area',
  'Segments',
  'Vectors',
  'Handle',
  'Raster',
  'Contours',
  'seriesLayers',
]

function phaseCharts(c: Ctx) {
  const done = new Set<Node>()
  for (const { fn, body, name } of components(c.sf)) {
    const charts = elements(body, new Set(['XYChart', 'Heatmap']))
    let n = 0
    const chosen = new Set<string>()
    const axisName = (base: string) => {
      let k = n === 0 ? base : `${base}${n + 1}`
      while (declaredInFile(c.sf, k) || chosen.has(k)) k += '_'
      chosen.add(k)
      return k
    }
    for (const el of charts) {
      const tag = tagName(el)!
      if (nearestFunction(el) !== fn) {
        c.mark(el, `${tag} inside a nested function: hoist its useAxis models by hand`)
        continue
      }
      if (!Node.isJsxSelfClosingElement(el)) {
        c.mark(el, `${tag} with children`)
        continue
      }
      const a = attrsOf(el)
      if (!a) {
        c.mark(el, `${tag} with spread props`)
        continue
      }
      // A local binding or another import under a layer's name would shadow it: leave the chart.
      const clash = CHART_NAMES.find((n) => taken(c.sf, n, 'aifn-render'))
      if (clash) {
        c.mark(el, `${tag}: ${clash} is a name in this file already; rename it, then rerun`)
        continue
      }
      const out = tag === 'XYChart' ? xyChart(c, a, body, done) : heatmap(c, a, body, done)
      if (typeof out === 'string') {
        c.mark(el, `${tag}: ${out}`)
        continue
      }
      const stmt = topStatement(body, el)
      const ind = indentOf(stmt)
      const xn = axisName('xAxis')
      const yn = axisName('yAxis')
      n++
      const yOpts = out.equal ? [...out.y, `equal: ${xn}`] : out.y
      const decl = `const ${xn} = useAxis({ ${out.x.join(', ')} })\n${ind}const ${yn} = useAxis({ ${yOpts.join(', ')} })\n${ind}`
      // Above the statement's MIGRATE lines, so they stay on the statement they describe.
      const marks = stmt.getLeadingCommentRanges().filter((r) => r.getText().startsWith('// MIGRATE:'))
      const at = marks.length ? marks[0].getPos() : stmt.getStart()
      c.edit(at, at, decl)
      c.replace(
        el,
        `<Plot x={${xn}} y={${yn}}${out.plot.map((p) => ` ${p}`).join('')}>\n${out.layers.join('\n')}\n</Plot>`,
      )
      c.add('aifn-render', 'Plot', 'useAxis', ...out.imports)
      c.changes.push(`${name}: ${tag} → Plot (${out.layers.length} layers)`)
    }
  }
}

type ChartOut = { x: string[]; y: string[]; equal: boolean; plot: string[]; layers: string[]; imports: string[] }

const tagsIn = (layers: string[]) => [
  ...new Set(layers.flatMap((l) => [...l.matchAll(/<([A-Z]\w*)|\b(seriesLayers)\(/g)].map((m) => m[1] ?? m[2]))),
]

function axisOpts(
  label: string | undefined,
  range: { text: string; init: Node | undefined } | undefined,
  hold: boolean,
  log: boolean,
) {
  const o: string[] = []
  if (label) o.push(`label: ${label}`)
  if (range) o.push(`range: ${range.text}`)
  const partial =
    range?.init &&
    Node.isArrayLiteralExpression(range.init) &&
    range.init.getElements().some((e) => e.getText() === 'undefined')
  if (hold && (!range || partial)) o.push(`hold: 'union'`)
  if (log) o.push('log: true')
  return o
}

function xyChart(c: Ctx, a: Attrs, body: Node, done: Set<Node>): ChartOut | string {
  const known = new Set([
    'series',
    'xLabel',
    'yLabel',
    'xRange',
    'yRange',
    'yLog',
    'segments',
    'vectors',
    'equalAspect',
    'bare',
    'handles',
    'onPlotClick',
    'height',
    'ariaLabel',
  ])
  const odd = [...a.keys()].find((k) => !known.has(k))
  if (odd) return `prop ${odd} has no automatic layer`
  const s = a.get('series')
  if (!s?.init) return 'no series'
  // Series built by code keep their list and go through `seriesLayers` (render's SeriesSpec).
  const expanded = seriesLayers(c, s.init, body, false, done)
  const layers = typeof expanded === 'string' ? [`{seriesLayers(${s.text})}`] : expanded
  if (a.get('segments')) layers.push(`<Segments segments={${a.get('segments')!.text}} />`)
  if (a.get('vectors')) layers.push(`<Vectors vectors={${a.get('vectors')!.text}} />`)
  layers.push(...handleLayers(c, a))
  return {
    x: axisOpts(a.get('xLabel')?.text, a.get('xRange'), true, false),
    y: axisOpts(a.get('yLabel')?.text, a.get('yRange'), true, a.get('yLog')?.text === 'true'),
    equal: a.get('equalAspect')?.text === 'true',
    plot: ['height', 'bare', 'onPlotClick', 'ariaLabel'].flatMap((k) =>
      a.get(k) ? [k === 'bare' && a.get(k)!.text === 'true' ? 'bare' : `${k}={${a.get(k)!.text}}`] : [],
    ),
    layers,
    imports: tagsIn(layers),
  }
}

function heatmap(c: Ctx, a: Attrs, body: Node, done: Set<Node>): ChartOut | string {
  const raster = ['scale', 'range', 'categoryNames', 'scaleTicks', 'fillOpacity', 'valueLabel', 'colorBar']
  const known = new Set([
    'x',
    'y',
    'z',
    'xLabel',
    'yLabel',
    'contours',
    'marker',
    'handles',
    'vectors',
    'overlay',
    'equalAspect',
    'height',
    'ariaLabel',
    'onPointer',
    ...raster,
  ])
  const odd = [...a.keys()].find((k) => !known.has(k))
  if (odd) return `prop ${odd} has no automatic layer`
  const [x, y, z] = ['x', 'y', 'z'].map((k) => a.get(k)?.text)
  if (!x || !y || !z) return 'missing x, y or z'
  const attr = (k: string) => (a.get(k) ? [`${k}={${a.get(k)!.text}}`] : [])
  const layers = [`<Raster x={${x}} y={${y}} z={${z}} ${raster.flatMap(attr).join(' ')} />`]
  const cont = a.get('contours')
  if (cont?.init) {
    if (Node.isObjectLiteralExpression(cont.init)) {
      const prop = (k: string) => {
        const p = (cont.init as never as { getProperty(k: string): Node | undefined }).getProperty(k)
        return p && Node.isPropertyAssignment(p)
          ? p.getInitializer()!.getText()
          : p && Node.isShorthandPropertyAssignment(p)
            ? k
            : undefined
      }
      layers.push(`<Contours x={${x}} y={${y}} z={${prop('field') ?? z}} levels={${prop('levels')}} />`)
    } else layers.push(`<Contours x={${x}} y={${y}} z={${cont.text}.field ?? ${z}} levels={${cont.text}.levels} />`)
  }
  const ov = a.get('overlay')
  if (ov?.init) {
    const more = seriesLayers(c, ov.init, body, true, done)
    layers.push(...(typeof more === 'string' ? [`{seriesLayers(${ov.text}, { live: true })}`] : more))
  }
  const m = a.get('marker')
  if (m?.init) {
    const [mx, my] =
      Node.isArrayLiteralExpression(m.init) && m.init.getElements().length === 2
        ? m.init.getElements().map((e) => e.getText())
        : [`${m.text}[0]`, `${m.text}[1]`]
    const literal = Node.isArrayLiteralExpression(m.init)
    const point = `<Points x={[${mx}]} y={[${my}]} emphasis live />`
    layers.push(literal ? point : `{${m.text} && ${point}}`)
  }
  if (a.get('vectors')) layers.push(`<Vectors vectors={${a.get('vectors')!.text}} />`)
  layers.push(...handleLayers(c, a))
  return {
    x: axisOpts(a.get('xLabel')?.text, undefined, false, false),
    y: axisOpts(a.get('yLabel')?.text, undefined, false, false),
    equal: a.get('equalAspect')?.text === 'true',
    plot: ['height', 'ariaLabel', 'onPointer'].flatMap(attr),
    layers,
    imports: tagsIn(layers),
  }
}

// ── Phase 4: Interactive → Figure, ParamButton → Button ─────────────────────────────────────────────────────────────

function phaseShells(c: Ctx) {
  for (const el of elements(c.sf, new Set(['Interactive', 'ParamButton']))) {
    const tag = tagName(el)!
    const to = tag === 'Interactive' ? 'Figure' : 'Button'
    const tagNode = (el as never as { getTagNameNode(): Node }).getTagNameNode()
    c.replace(tagNode, to)
    if (Node.isJsxOpeningElement(el))
      c.replace(
        (el.getParent() as never as { getClosingElement(): { getTagNameNode(): Node } })
          .getClosingElement()
          .getTagNameNode(),
        to,
      )
    const a = attrsOf(el) ?? new Map()
    if (tag === 'ParamButton') c.edit(tagNode.getEnd(), tagNode.getEnd(), ' variant="outline" size="sm"')
    else {
      const readout = a.get('readout')
      if (readout) c.replace((readout.node as never as { getNameNode(): Node }).getNameNode(), 'readouts')
      // No purpose: optional on Figure, and a note's prose already explains its figures.
    }
    c.add('aifn-render', to)
  }
  const n = c.edits.length
  if (n) c.changes.push('shells: Interactive → Figure, ParamButton → Button')
}

// ── Phase 5: imports ────────────────────────────────────────────────────────────────────────────────────────────────

function phaseImports(c: Ctx, touched: boolean) {
  const sf = c.sf
  const referenced = (local: string) => readsOf(sf, local).length > 0
  const imports = sf.getImportDeclarations()
  const handled = new Set<string>()
  const merged = new Set<Node>()
  for (const imp of imports) {
    const mod = imp.getModuleSpecifierValue()
    const typeOnly = imp.isTypeOnly()
    const adds = typeOnly ? undefined : c.adds.get(mod)
    const managed =
      touched && (RENDER_MODULES.has(mod) || mod === 'react' || mod.startsWith('aifn-compute/') || c.adds.has(mod))
    if (!managed || imp.getDefaultImport() || imp.getNamespaceImport()) continue
    if (handled.has(`${mod}|${typeOnly}`)) continue
    handled.add(`${mod}|${typeOnly}`)
    // Several declarations of one module (same type-only-ness) merge into this one; the others go.
    const same = imports.filter(
      (i) =>
        i !== imp &&
        i.getModuleSpecifierValue() === mod &&
        i.isTypeOnly() === typeOnly &&
        !i.getDefaultImport() &&
        !i.getNamespaceImport(),
    )
    for (const other of same) {
      removeLine(c, other)
      merged.add(other)
    }
    const specs = [imp, ...same].flatMap((d) =>
      d.getNamedImports().map((s) => (s.isTypeOnly() || typeOnly ? 'type ' : '') + s.getText().replace(/^type /, '')),
    )
    const names = new Set(specs)
    if (adds)
      for (const a of adds) {
        const bare = a.replace(/^type /, '')
        // A value import replaces a type-only one of the same name (`type Handle` → `Handle`, now drawn as a layer).
        if (!a.startsWith('type ') && names.has(`type ${bare}`)) names.delete(`type ${bare}`)
        if (![...names].some((s) => s.replace(/^type /, '').split(' as ')[0] === bare.split(' as ')[0])) names.add(a)
      }
    const live = [...names].filter((s) =>
      referenced(
        s
          .replace(/^type /, '')
          .split(' as ')
          .at(-1)!,
      ),
    )
    if (!same.length && live.length === specs.length && live.every((s, i) => s === specs[i])) continue
    if (live.length) c.replace(imp, `import ${typeOnly ? 'type ' : ''}{ ${sortSpecs(live).join(', ')} } from '${mod}'`)
    else removeLine(c, imp)
  }
  // Modules not imported yet.
  const last = imports.filter((i) => !merged.has(i)).at(-1)
  const fresh: string[] = []
  for (const [mod, names] of c.adds) {
    if (handled.has(`${mod}|false`) || imports.some((i) => i.getModuleSpecifierValue() === mod && !i.isTypeOnly()))
      continue
    const live = [...names].filter((s) =>
      referenced(
        s
          .replace(/^type /, '')
          .split(' as ')
          .at(-1)!,
      ),
    )
    if (live.length) fresh.push(`import { ${sortSpecs(live).join(', ')} } from '${mod}'`)
  }
  if (fresh.length)
    c.edit(
      last ? last.getEnd() : 0,
      last ? last.getEnd() : 0,
      `${last ? '\n' : ''}${fresh.join('\n')}${last ? '' : '\n'}`,
    )
}

const sortSpecs = (s: string[]) =>
  [...new Set(s)].sort((a, b) =>
    a.replace(/^type /, '').localeCompare(b.replace(/^type /, ''), 'en', { sensitivity: 'base' }),
  )

// ── Phase 6: mark what is left ──────────────────────────────────────────────────────────────────────────────────────

// ── Phase 4b: compat series types → SeriesSpec ──────────────────────────────────────────────────────────────────────

const SERIES_TYPES = new Set(['XYSeries', 'Series', 'HeatmapOverlay'])

/** `XYSeries`, `Series`, `HeatmapOverlay` → `SeriesSpec`; `pointColors` keys → `colors` in such a file. */
function phaseTypes(c: Ctx) {
  const sf = c.sf
  // While a compat chart remains, its props still want the compat types.
  if (elements(sf, new Set(['XYChart', 'Heatmap'])).length) return
  let renamed = false
  for (const imp of sf.getImportDeclarations()) {
    if (!RENDER_MODULES.has(imp.getModuleSpecifierValue())) continue
    for (const s of imp.getNamedImports()) {
      if (!SERIES_TYPES.has(s.getName())) continue
      const local = s.getAliasNode()?.getText() ?? s.getName()
      const reads = readsOf(sf, local)
      if (!reads.length || taken(sf, 'SeriesSpec', 'aifn-render')) continue
      for (const r of reads) c.replace(r, 'SeriesSpec')
      renamed = true
    }
  }
  if (!renamed) return
  for (const id of sf.getDescendantsOfKind(SyntaxKind.Identifier)) {
    if (id.getText() !== 'pointColors') continue
    const p = id.getParent()!
    if (Node.isPropertyAssignment(p) && p.getNameNode() === id) c.replace(id, 'colors')
    else if (Node.isShorthandPropertyAssignment(p)) c.replace(p, 'colors: pointColors')
    else if (Node.isPropertyAccessExpression(p) && p.getNameNode() === id) c.replace(id, 'colors')
  }
  c.add('aifn-render', 'type SeriesSpec')
  c.changes.push('types: compat series types → SeriesSpec')
}

function phaseMarks(c: Ctx) {
  const sf = c.sf
  for (const imp of sf.getImportDeclarations()) {
    if (!RENDER_MODULES.has(imp.getModuleSpecifierValue())) continue
    for (const s of imp.getNamedImports()) {
      if (!COMPAT.has(s.getName())) continue
      const local = s.getAliasNode()?.getText() ?? s.getName()
      for (const r of readsOf(sf, local)) {
        const stmt = statementOf(r)
        const leading = stmt.getFullText().slice(0, stmt.getStart() - stmt.getPos())
        if (/MIGRATE:/.test(leading) && leading.includes(s.getName())) continue
        if (Node.isTypeReference(r.getParent()!) || Node.isImportSpecifier(r.getParent()!)) {
          c.mark(r, `${s.getName()} (compat type) still used`)
        } else c.mark(r, `${s.getName()} has no automatic rewrite here (MIGRATING-NOTES.md)`)
      }
    }
  }
}

/** The first line of a file the codemod's output broke, reverted to its original. */
const REVERTED = '// MIGRATE: codemod output did not type-check'

// ── Driver ──────────────────────────────────────────────────────────────────────────────────────────────────────────

// noLib/noResolve: the binder's scopes are all the codemod asks of the checker (bindingOf).
const project = new Project({
  useInMemoryFileSystem: true,
  compilerOptions: { jsx: 1 /* preserve */, noLib: true, noResolve: true },
})

function migrate(file: string, source: string): { text: string; report: Report } {
  const c = new Ctx()
  const parse = (text: string) => (c.sf = project.createSourceFile('/f.tsx', text, { overwrite: true }))
  // MIGRATE lines are re-derived on every run; an empty purpose left by an earlier version goes with its marker.
  let text = source
    .replace(/\n[ \t]*\/\/ MIGRATE: write a one-line purpose\n[ \t]*purpose=""/g, '')
    .replace(/^[ \t]*\/\/ MIGRATE: .*\n/gm, '')
  parse(text)
  const report: Report = { file, changes: [], left: [], before: countLegacy(c.sf), after: 0 }
  const phases: [string, (c: Ctx) => void][] = [
    ['maths', phaseMaths],
    ['state', phaseState],
    ['charts', phaseCharts],
    ['shells', phaseShells],
    ['types', phaseTypes],
  ]
  let phasesChanged = false
  try {
    for (const [, phase] of phases) {
      c.edits = []
      phase(c)
      if (c.edits.length) parse((text = applyEdits(text, c.edits)))
      phasesChanged ||= c.edits.length > 0
    }
    c.edits = []
    phaseImports(c, phasesChanged)
    if (c.edits.length) parse((text = applyEdits(text, c.edits)))
    c.edits = []
    phaseMarks(c)
    if (c.edits.length) parse((text = applyEdits(text, c.edits)))
  } catch (e) {
    return { text: source, report: { ...report, error: String(e), after: report.before } }
  }
  report.changes = c.changes
  report.left = c.left
  report.after = countLegacy(c.sf)
  return { text, report }
}

function filesUnder(p: string): string[] {
  const st = fs.statSync(p)
  if (st.isFile()) return /\.tsx?$/.test(p) ? [p] : []
  return fs
    .readdirSync(p, { withFileTypes: true })
    .flatMap((e) => (e.name === 'node_modules' ? [] : filesUnder(path.join(p, e.name))))
}

async function main() {
  const args = process.argv.slice(2)
  const dry = args.includes('--dry-run')
  const quiet = args.includes('--quiet')
  const jsonAt = args.indexOf('--json')
  const jsonFile = jsonAt >= 0 ? args[jsonAt + 1] : undefined
  const roots = args.filter((a, i) => !a.startsWith('--') && (jsonAt < 0 || i !== jsonAt + 1))
  const files = (roots.length ? roots : ['content/notes']).flatMap(filesUnder)
  const prettier = dry ? undefined : await import('prettier')
  const reports: Report[] = []
  for (const file of files) {
    const source = fs.readFileSync(file, 'utf8')
    if (!/aifn-render|@\/lib\/(math|dsp|distributions)/.test(source)) continue
    // A file whose codemod output did not type-check was reverted and is left for a person.
    if (source.includes(REVERTED)) {
      reports.push({ file, changes: [], left: ['reverted: migrate by hand'], before: 1, after: 1 })
      continue
    }
    const { text, report } = migrate(file, source)
    if (report.before === 0 && text === source) continue
    reports.push(report)
    if (!dry && text !== source) {
      // A result prettier cannot parse is a codemod bug: keep the file as it was and say so.
      try {
        const options = (await prettier!.resolveConfig(file)) ?? {}
        fs.writeFileSync(file, await prettier!.format(text, { ...options, filepath: file }))
      } catch (e) {
        report.error = `not written: ${String(e).split('\n')[0]}`
        report.after = report.before
      }
    }
    if (args.includes('--print')) console.log(text)
    if (!quiet) {
      console.log(
        `${path.relative(process.cwd(), file)}  legacy ${report.before} → ${report.after}${report.error ? `  ERROR ${report.error}` : ''}`,
      )
      for (const ch of report.changes) console.log(`  + ${ch}`)
      const left = new Map<string, number>()
      for (const l of report.left) left.set(l, (left.get(l) ?? 0) + 1)
      for (const [l, k] of left) console.log(`  - ${l}${k > 1 ? ` ×${k}` : ''}`)
    }
  }
  const before = reports.reduce((s, r) => s + r.before, 0)
  const after = reports.reduce((s, r) => s + r.after, 0)
  const clean = reports.filter((r) => r.after === 0 && !r.left.length).length
  console.log(
    `\n${reports.length} files with legacy usages; ${before} usages before, ${after} after: ${before ? ((100 * (before - after)) / before).toFixed(1) : 0}% handled; ` +
      `${clean} files fully migrated; ${reports.filter((r) => r.error).length} errors${dry ? ' (dry run)' : ''}`,
  )
  if (jsonFile) fs.writeFileSync(jsonFile, JSON.stringify(reports, null, 1))
}

await main()
