/**
 * `make links`: finds where notes mention glossary entries or other notes without marking them, and proposes the
 * markup: `<Gloss name>` for glossary entries, `<NoteLink to>` for notes. It reads the tree the way the build does
 * (plugins/content-index.ts, plugins/glossary.ts) and matches prose only: frontmatter, code, maths, headings, JSX tags,
 * link text and existing <Gloss>, <NoteLink> and <Cite> are masked out.
 *
 * Each candidate is the first unmarked mention of a term in a note, with its sentence, and a decision:
 *   add       the markup is proposed;
 *   skip      with the reason (the note is the term's own note, the term is already linked or glossed, it is defined
 *             in bold here, it is too generic, an ambiguous acronym whose sense cannot be told, ...).
 * The report is for reading the decisions, not the notes. `--apply` writes every `add` into the notes.
 *
 *   node scripts/suggest-links.ts [--mode gloss|notes|both] [--apply] [--json file] [--md file] <taxonomy path | slug ...>
 *
 * Glossary matching: an acronym's short form case-sensitively (with a plural "s"); long forms and aliases
 * case-insensitively, with hyphens, spaces and dashes interchangeable and an optional plural. Note matching uses the
 * title (and the title before a parenthesis or colon) and aliases the same way; acronym-like aliases match
 * case-sensitively. Where matches overlap, the longest wins.
 */
import fs from 'node:fs'
import path from 'node:path'
import { buildIndex } from '../plugins/content-index.ts'
import { loadGlossary } from '../plugins/glossary.ts'
import type { GlossaryEntry, NoteMeta } from '../site/src/lib/content-schema.ts'

const root = path.resolve(import.meta.dirname, '..')
const contentDir = path.join(root, 'content')

// ---------------------------------------------------------------------------------------------------------------------
// Arguments

const argv = process.argv.slice(2)
const flag = (name: string) => {
  const i = argv.indexOf(name)
  if (i < 0) return undefined
  const v = argv[i + 1]
  argv.splice(i, 2)
  return v
}
const mode = (flag('--mode') ?? 'both') as 'gloss' | 'notes' | 'both'
const jsonOut = flag('--json')
const mdOut = flag('--md')
const apply = argv.includes('--apply')
const scopeArgs = argv.filter((a) => !a.startsWith('--')).map((a) => a.replace(/^\/+|\/+$/g, ''))
const inScope = (n: { slug: string; category: string }) =>
  scopeArgs.length === 0 || scopeArgs.some((a) => n.slug === a || n.category === a || n.category.startsWith(`${a}/`))

const { notes: allNotes } = buildIndex(contentDir, { inScope: () => false })
const glossary = loadGlossary(path.join(contentDir, 'glossary.yaml'))
const notes = allNotes.filter(inScope)
for (const a of scopeArgs)
  if (!allNotes.some((n) => n.slug === a || n.category === a || n.category.startsWith(`${a}/`)))
    throw new Error(`scope "${a}" matches no note slug or taxonomy path`)

// ---------------------------------------------------------------------------------------------------------------------
// Prose mask: true where a character is running prose that may take markup.

function proseMask(text: string): boolean[] {
  const ok = Array<boolean>(text.length).fill(true)
  const block = (from: number, to: number) => ok.fill(false, from, to)
  const all = (re: RegExp) => {
    for (const m of text.matchAll(re)) block(m.index, m.index + m[0].length)
  }
  all(/^---\n[\s\S]*?\n---\n/g) // frontmatter
  all(/^(import|export)\s.*$/gm)
  all(/```[\s\S]*?```/g)
  all(/\$\$[\s\S]*?\$\$/g)
  all(/(?<![\\$])\$(?!\$)(?:[^$\\\n]|\\[^\n]|\n(?![ \t]*\n))+?\$/g) // inline maths, which may wrap but not span a blank line
  all(/`[^`\n]+`/g)
  all(/^#{1,6}\s.*$/gm) // headings: the table of contents shows their source
  all(/<(NoteLink|Gloss|Cite|a)\b[^>]*\/>/g)
  all(/<(NoteLink|Gloss|a)\b[^>]*>[\s\S]*?<\/\1>/g)
  all(/<\/?[A-Za-z][^<>]*?>/g) // any other tag, attributes included; its children stay prose
  all(/\[[^\]\n]*\]\([^)\n]*\)/g) // markdown links
  all(/^\s*\|?\s*:?-{3,}.*$/gm) // table rules
  return ok
}

const eligible = (mask: boolean[], from: number, to: number) => {
  for (let i = from; i < to; i++) if (!mask[i]) return false
  return true
}

// ---------------------------------------------------------------------------------------------------------------------
// Patterns

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** A phrase as a pattern: separators interchangeable, apostrophes optional, an optional plural. */
function phrasePattern(phrase: string): string {
  const words = phrase
    .replace(/[’']/g, "'")
    .split(/[\s\-–—_]+/)
    .filter(Boolean)
  const body = words.map((w) => escape(w).replace(/'/g, "['’]?")).join('[\\s\\-–—]+')
  return `${body}(?:s|es)?`
}

const slugWords = (s: string) => s.replace(/-/g, ' ')
/** Short all-capital or mixed tokens (SVM, t-SNE, GPT-2) match case-sensitively; phrases do not. */
const acronymLike = (s: string) => !/\s/.test(s) && /[A-Z].*[A-Z0-9]|^[a-z]+-[A-Z]/.test(s) && s.length <= 12

/** A capital after the first character, or a single capital letter joined by a hyphen, makes a name case-sensitive. */
const caseMatters = (s: string) => /^[A-Z][-–]/.test(s) || /[A-Z]/.test(s.slice(1))

type Matcher = { re: RegExp; form: 'short' | 'long' | 'alias' | 'title' }

function matchers(names: { text: string; form: Matcher['form'] }[]): Matcher[] {
  const out: Matcher[] = []
  const seen = new Set<string>()
  for (const { text, form } of names) {
    const t = text.trim()
    if (t.length < 2 || seen.has(t.toLowerCase() + form)) continue
    seen.add(t.toLowerCase() + form)
    if (acronymLike(t)) out.push({ re: new RegExp(`(?<![\\w\\\\-])${escape(t)}s?(?![\\w-])`, 'g'), form })
    else if (form !== 'long' && /^[A-Z][-–]/.test(t))
      // A single capital joined by a hyphen (A-distance, H-divergence) is a symbol: matched exactly.
      out.push({ re: new RegExp(`(?<![\\w\\\\-])${escape(t)}s?(?![\\w-])`, 'g'), form })
    else if (form !== 'long' && caseMatters(t)) {
      // Capitals beyond the first letter are part of the name (A-distance, Kullback–Leibler); only the first letter may
      // change case, at the start of a sentence.
      const first = t[0]
      const rest = phrasePattern(t).slice(escape(first).length)
      out.push({
        re: new RegExp(`(?<![\\w\\\\-])[${first.toUpperCase()}${first.toLowerCase()}]${rest}(?![\\w-])`, 'g'),
        form,
      })
    } else out.push({ re: new RegExp(`(?<![\\w\\\\-])${phrasePattern(t)}(?![\\w-])`, 'gi'), form })
  }
  return out
}

// ---------------------------------------------------------------------------------------------------------------------
// Targets: glossary entries and notes, each with its matchers.

type Target = {
  kind: 'gloss' | 'note'
  id: string
  label: string
  matchers: Matcher[]
  entry?: GlossaryEntry
  note?: NoteMeta
}

const shortCount = new Map<string, number>()
for (const e of Object.values(glossary.entries))
  if (e.short) shortCount.set(e.short, (shortCount.get(e.short) ?? 0) + 1)

const GENERIC_MIN_WORDS = 2

/** Aliases that name something else as often as their target: never matched. */
const WEAK_ALIASES = new Set(
  [
    'cross terms',
    'weighted average',
    'relative error',
    'sample size',
    'product rule',
    'step size',
    'hidden state',
    'forward pass',
    'order of integration',
    'sum rule',
    'sampling distribution',
    'variational free energy',
    'confidence bound',
    'longest common subsequence',
    'standard deviation of the mean',
    'noise floor',
    'large deviations',
    'invariance principle',
    'step decays',
    'initial conditions',
    'squared prediction errors',
    'shortest path',
    'rich get richer',
    'sliding window',
    'cumulative sum',
    'encoder network',
    'gradient flow',
    'flow map',
    'adversarial training',
    'log',
    'gem',
    'ate',
    'node',
    'global error',
    'noise suppression',
    'location–scale model',
    'location-scale model',
  ].map((a) => a.replace(/[\s\-–—]+/g, ' ')),
)

/**
 * Entries whose acronym means something else in a whole topic, where the other sense has no entry of its own to
 * trigger the ambiguity check: ROC is the region of convergence in signal processing and control, and CORAL is
 * correlation alignment outside ordinal regression.
 */
const BLOCKED_IN_TOPIC: Record<string, (category: string) => boolean> = {
  roc: (c) => /^(signal-processing|control-theory)(\/|$)/.test(c),
  'consistent-rank-logits': (c) => !c.startsWith('supervised-learning/ordinal-regression'),
}

/** Subjects every other topic draws on: an alias may link into them from anywhere. */
const FOUNDATIONS = new Set(['maths', 'probability', 'probability-distributions', 'statistics'])
const top = (category: string) => category.split('/')[0]
const glossTargets: Target[] = Object.values(glossary.entries).map((e) => {
  const names: { text: string; form: Matcher['form'] }[] = []
  if (e.short) names.push({ text: e.short, form: 'short' })
  names.push({ text: e.long, form: 'long' })
  if (e.longPlural) names.push({ text: e.longPlural, form: 'long' })
  // Slug aliases name the same thing; the key and its acronym are covered by short, other aliases read as phrases.
  for (const a of e.aliases) {
    const words = slugWords(a)
    if (e.short && words.replace(/\s/g, '') === e.short.toLowerCase().replace(/[^a-z0-9]/g, '')) continue
    if (!/\s/.test(words) && words.length <= 5) continue // bare acronym aliases: the short form covers them
    names.push({ text: words, form: 'alias' })
  }
  return { kind: 'gloss', id: e.key, label: e.short ?? e.long, matchers: matchers(names), entry: e }
})

const noteTargets: Target[] = allNotes.map((n) => {
  const names: { text: string; form: Matcher['form'] }[] = [{ text: n.title, form: 'title' }]
  const core = n.title.split(/\s*[(:]/)[0]
  if (core !== n.title) names.push({ text: core, form: 'title' })
  for (const a of n.aliases) names.push({ text: a, form: 'alias' })
  return { kind: 'note', id: n.slug, label: n.title, matchers: matchers(names), note: n }
})

// ---------------------------------------------------------------------------------------------------------------------
// Scan

type Candidate = {
  note: string
  file: string
  kind: 'gloss' | 'note'
  target: string
  label: string
  form: Matcher['form']
  match: string
  start: number
  end: number
  line: number
  sentence: string
  decision: 'add' | 'skip'
  reason: string
  replacement?: string
}

function sentenceAround(text: string, start: number, end: number): string {
  const para0 = text.lastIndexOf('\n\n', start)
  const para1 = text.indexOf('\n\n', end)
  const from = Math.max(para0 < 0 ? 0 : para0 + 2, text.slice(0, start).search(/[.!?]\s+[^.!?]*$/) + 1 || 0)
  const stop = text.slice(end, para1 < 0 ? undefined : para1).search(/[.!?](\s|$)/)
  const to = stop < 0 ? (para1 < 0 ? text.length : para1) : end + stop + 1
  const s = `${text.slice(from, start)}[[${text.slice(start, end)}]]${text.slice(end, to)}`
  return s.replace(/\s+/g, ' ').trim().slice(0, 320)
}

const linkedIn = (body: string) =>
  new Set([...body.matchAll(/<NoteLink\b[^>]*\bto=["']([^"']+)["']/g)].map((m) => m[1]))
const glossedIn = (body: string) =>
  new Set([...body.matchAll(/<Gloss\b[^>]*\bname=["']([^"']+)["']/g)].map((m) => glossary.names.get(m[1]) ?? m[1]))
const boldIn = (body: string) =>
  [...body.matchAll(/\*\*([^*\n]+)\*\*/g)].map((m) => m[1].toLowerCase().replace(/[\s\-–—]+/g, ' '))

/** The sense of an ambiguous acronym that fits the note: same category branch, or its note related to this one. */
function senseFits(e: GlossaryEntry, n: NoteMeta): boolean {
  const rel = new Set([...n.requires, ...n.partOf, ...n.related, ...n.linked])
  const branch = (p: string, depth: number) => p.split('/').slice(0, depth).join('/')
  return (
    (e.note !== undefined && (rel.has(e.note) || e.note === n.slug)) ||
    e.see.some((s) => rel.has(s) || s === n.slug) ||
    e.category.some((c) => branch(c, 2) === branch(n.category, 2)) ||
    e.category.some((c) => branch(c, 1) === branch(n.category, 1))
  )
}

function glossReplacement(e: GlossaryEntry, match: string, form: Matcher['form'], text: string, end: number) {
  // "long form (SHORT)" in the source: the whole phrase becomes the first-use expansion.
  const paren = e.short ? new RegExp(`^\\s*\\(${escape(e.short)}s?\\)`).exec(text.slice(end)) : null
  // The author's words stay as written; the entry's Title Case long form is for the glossary and hover card.
  if (form !== 'short' && paren) {
    const words = text.slice(end - match.length, end + paren[0].length)
    return { text: `<Gloss name="${e.key}">${words}</Gloss>`, end: end + paren[0].length }
  }
  if (form === 'short') {
    const plural = match.length > e.short!.length
    return { text: `<Gloss name="${e.key}"${plural ? ' plural' : ''} />`, end }
  }
  return { text: `<Gloss name="${e.key}">${match}</Gloss>`, end }
}

const candidates: Candidate[] = []
for (const n of notes) {
  const file = path.join(contentDir, n.file)
  const text = fs.readFileSync(file, 'utf8')
  const mask = proseMask(text)
  const linked = linkedIn(text)
  const glossed = glossedIn(text)
  const bold = boldIn(text)
  const boldSpans = [...text.matchAll(/\*\*([^*\n]+)\*\*/g)].map((m) => ({
    start: m.index + 2,
    end: m.index + 2 + m[1].length,
    inner: m[1].replace(/[.:,;]+$/, '').trim(),
  }))
  // Part of a bold defined term (**Lie derivative**) names something else. A bold label that ends in a full stop or
  // colon (**Linear maps of Gaussian vectors.**) is a sentence and may carry links.
  const insideBoldTerm = (start: number, end: number) =>
    boldSpans.some(
      (b) =>
        b.start <= start &&
        end <= b.end &&
        !/[.:]$/.test(text.slice(b.start, b.end).trim()) &&
        text.slice(start, end).trim() !== b.inner,
    )
  // Terms the note is itself about: its title and aliases. It explains them; it does not mark them.
  const own = new Set(
    [n.title, ...n.aliases].map((a) =>
      a
        .toLowerCase()
        .replace(/[\s\-–—]+/g, ' ')
        .replace(/s$/, ''),
    ),
  )
  const ownTerm = (norm: string) => own.has(norm.replace(/s$/, ''))
  const lineOf = (i: number) => text.slice(0, i).split('\n').length

  // Every eligible match of every target, then longest-first so a longer term claims its span.
  type Hit = { t: Target; m: Matcher; start: number; end: number }
  const hits: Hit[] = []
  // A concept with a note of its own is marked by a NoteLink; as a glossary target it would only claim the words first.
  const glossPool =
    mode === 'both' ? glossTargets.filter((t) => !(t.entry!.kind === 'concept' && t.entry!.note)) : glossTargets
  const targets = [...(mode !== 'notes' ? glossPool : []), ...(mode !== 'gloss' ? noteTargets : [])]
  for (const t of targets) {
    if (t.kind === 'note' && t.id === n.slug) continue
    for (const m of t.matchers) {
      for (const x of text.matchAll(m.re)) {
        const start = x.index
        const end = start + x[0].length
        if (eligible(mask, start, end)) hits.push({ t, m, start, end })
      }
    }
  }
  hits.sort((a, b) => b.end - b.start - (a.end - a.start) || a.start - b.start)
  const claimed: [number, number, string][] = []
  const firstHit = new Map<string, Hit & { overlapped?: string }>()
  for (const h of hits) {
    const key = `${h.t.kind}:${h.t.id}`
    const over = claimed.find(([s, e, k]) => h.start < e && s < h.end && k !== key)
    if (over) continue // a longer term already covers these words
    claimed.push([h.start, h.end, key])
    const prev = firstHit.get(key)
    if (!prev || h.start < prev.start) firstHit.set(key, h)
  }

  const glossAdded = new Set<string>()
  const ordered = [...firstHit.values()].sort((a, b) =>
    a.t.kind === b.t.kind ? a.start - b.start : a.t.kind === 'gloss' ? -1 : 1,
  )
  for (const h of ordered) {
    const match = text.slice(h.start, h.end)
    const base = {
      note: n.slug,
      file: n.file,
      kind: h.t.kind,
      target: h.t.id,
      label: h.t.label,
      form: h.m.form,
      match,
      start: h.start,
      end: h.end,
      line: lineOf(h.start),
      sentence: sentenceAround(text, h.start, h.end),
    }
    const skip = (reason: string) => candidates.push({ ...base, decision: 'skip', reason })
    const norm = match.toLowerCase().replace(/[\s\-–—]+/g, ' ')
    // A short alias is a looser name than a title or long form, so it must also fit the note's subject.
    const wordCount = match.trim().split(/\s+/).length
    const looseAlias =
      (h.m.form === 'alias' && wordCount < 3) || (h.m.form === 'title' && wordCount === 1 && !acronymLike(match))
    if (insideBoldTerm(h.start, h.end)) {
      skip('part of a longer bold term')
      continue
    }
    if (h.m.form === 'alias' && WEAK_ALIASES.has(norm.replace(/s$/, ''))) {
      skip('weak alias: names other things as often')
      continue
    }

    if (h.t.kind === 'gloss') {
      const e = h.t.entry!
      if (BLOCKED_IN_TOPIC[e.key]?.(n.category)) {
        skip(`${e.short ?? e.key} means something else in this topic`)
        continue
      }
      if (e.note === n.slug) {
        skip('this note is the entry’s own note')
        continue
      }
      if (ownTerm(norm) || (e.short && ownTerm(e.short.toLowerCase())) || ownTerm(e.long.toLowerCase())) {
        skip('the note is about this term (its title or alias)')
        continue
      }
      if (glossed.has(e.key)) {
        skip('already glossed in this note')
        continue
      }
      if (e.note && linked.has(e.note)) {
        skip(`the note already links ${e.note}`)
        continue
      }
      if (bold.some((b) => b === norm || b === e.long.toLowerCase())) {
        skip('defined in bold in this note')
        continue
      }
      if (h.m.form === 'short' && (shortCount.get(e.short!) ?? 0) > 1 && !senseFits(e, n)) {
        skip(`ambiguous ${e.short}: this sense does not fit the note`)
        continue
      }
      if (e.kind === 'concept' && e.note) {
        skip('concept with its own note: a NoteLink is the better mark')
        continue
      }
      if (e.kind === 'concept' && e.long.split(/\s+/).length < GENERIC_MIN_WORDS) {
        skip('single-word concept: too generic to mark')
        continue
      }
      if (looseAlias && !senseFits(e, n)) {
        skip('loose name outside the entry’s subject')
        continue
      }
      glossAdded.add(e.key)
      const r = glossReplacement(e, match, h.m.form, text, h.end)
      candidates.push({
        ...base,
        end: r.end,
        decision: 'add',
        reason: `${e.kind}, ${h.m.form} form`,
        replacement: r.text,
      })
    } else {
      const t = h.t.note!
      if (linked.has(t.slug)) {
        skip('already linked in this note')
        continue
      }
      const glossHere = Object.values(glossary.entries).find((e) => e.note === t.slug && e.kind !== 'concept')
      if (glossHere && (glossed.has(glossHere.key) || glossAdded.has(glossHere.key))) {
        skip(`the <Gloss name="${glossHere.key}"> hover links this note`)
        continue
      }
      if (bold.some((b) => b === norm)) {
        skip('defined in bold in this note')
        continue
      }
      if (ownTerm(norm)) {
        skip('the note is about this term (its title or alias)')
        continue
      }
      const words = match.trim().split(/\s+/).length
      // A single word links only when it is the target's own title (Gradient, Variance); single-word aliases are too loose.
      if (words < GENERIC_MIN_WORDS && (!acronymLike(match) || h.m.re.flags.includes('i')) && h.m.form !== 'title') {
        skip('single-word alias: too generic to link without reading')
        continue
      }
      const rel = [...n.requires, ...n.partOf, ...n.related].includes(t.slug)
      if (looseAlias && !rel && top(t.category) !== top(n.category) && !FOUNDATIONS.has(top(t.category))) {
        skip('loose name outside the target’s subject')
        continue
      }
      candidates.push({
        ...base,
        decision: 'add',
        reason: `${h.m.form}${rel ? ', declared relation' : ''}`,
        replacement: `<NoteLink to="${t.slug}">${match}</NoteLink>`,
      })
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Report

const adds = candidates.filter((c) => c.decision === 'add')
const reasons = new Map<string, number>()
for (const c of candidates.filter((c) => c.decision === 'skip')) {
  const r = c.reason.replace(/"[^"]*"|\b[a-z0-9]+(-[a-z0-9]+)+\b|: this sense.*|^ambiguous \S+/g, (s) =>
    s.startsWith('ambiguous') ? 'ambiguous acronym' : s.startsWith(':') ? '' : '…',
  )
  reasons.set(r, (reasons.get(r) ?? 0) + 1)
}

const lines: string[] = [
  `# Link suggestions (${mode})`,
  '',
  `${notes.length} notes · ${candidates.length} candidates · ${adds.length} to add`,
  '',
  '| skipped because | count |',
  '| --- | --- |',
  ...[...reasons].sort((a, b) => b[1] - a[1]).map(([r, k]) => `| ${r} | ${k} |`),
  '',
]
for (const n of notes) {
  const cs = candidates.filter((c) => c.note === n.slug).sort((a, b) => a.start - b.start)
  if (!cs.length) continue
  lines.push(`## ${n.slug}`, '')
  for (const c of cs) {
    const mark = c.decision === 'add' ? '+' : '·'
    lines.push(
      `- ${mark} **${c.kind === 'gloss' ? 'gloss' : 'link'} ${c.target}** (line ${c.line}, ${c.decision === 'add' ? c.reason : `skip: ${c.reason}`}): ${c.sentence}`,
    )
  }
  lines.push('')
}
const md = lines.join('\n')
if (mdOut) fs.writeFileSync(mdOut, md)
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(candidates, null, 1))
if (!mdOut && !jsonOut) console.log(md)
console.error(`${notes.length} notes · ${candidates.length} candidates · ${adds.length} to add`)

// ---------------------------------------------------------------------------------------------------------------------
// Apply: rewrite each note from its last addition to its first, so earlier offsets stay valid.

if (apply) {
  const byFile = new Map<string, Candidate[]>()
  for (const c of adds) byFile.set(c.file, [...(byFile.get(c.file) ?? []), c])
  for (const [rel, cs] of byFile) {
    const file = path.join(contentDir, rel)
    let text = fs.readFileSync(file, 'utf8')
    for (const c of cs.sort((a, b) => b.start - a.start))
      text = text.slice(0, c.start) + c.replacement + text.slice(c.end)
    fs.writeFileSync(file, text)
  }
  console.error(`applied ${adds.length} additions to ${byFile.size} notes`)
}
