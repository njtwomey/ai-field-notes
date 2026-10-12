/**
 * `make refs`: search content/references.yaml before adding a reference, and find likely duplicates already in it.
 *
 * ```bash
 * node scripts/references.ts search kernel two-sample                 # free text over title, authors, key and venue
 * node scripts/references.ts search --title "kernel two sample" --author gretton --year 2012
 * node scripts/references.ts search --key gretton --limit 5
 * node scripts/references.ts check --title "A Kernel Two-Sample Test" --author Gretton --url https://jmlr.org/...
 * node scripts/references.ts check --file new-refs.yaml               # every entry of a YAML fragment, and its keys
 * node scripts/references.ts duplicates                               # scan the whole file
 * node scripts/references.ts duplicates --threshold 0.8 --json
 * ```
 *
 * Matching: titles are compared after folding case, accents and punctuation, by the Dice coefficient of their character
 * trigrams (robust to plurals, hyphens and small typos); a search fragment scores by how much of it the title covers.
 * Authors are compared by surname. URLs are reduced to an identifier first: a DOI, an arXiv id without its version, or
 * the host and path, so `https://doi.org/10.1/X` and `http://dx.doi.org/10.1/x` match.
 *
 * `check` exits with status 1 when an entry is a likely duplicate (same identifier, or a near-identical title), so it
 * can guard a script. `duplicates` reports repeated keys, shared identifiers and near-identical titles.
 */
import fs from 'node:fs'
import path from 'node:path'
import { parse } from 'yaml'

const root = path.resolve(import.meta.dirname, '..')
const file = path.join(root, 'content', 'references.yaml')

type Entry = { title?: string; authors?: string[]; year?: number; venue?: string; url?: string; type?: string }
type Ref = Entry & {
  key: string
  line: number
  normTitle: string
  grams: Set<string>
  words: Set<string>
  surnames: Set<string>
  id: string
}

const STOP = new Set(
  'a an and are as at by for from in into is its of on or the to via with without using towards toward'.split(' '),
)

/** Lower case, accents removed, every run of non-alphanumerics a single space. */
export const fold = (s: string) =>
  s
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()

const trigrams = (s: string) => {
  const t = ` ${s} `
  const out = new Set<string>()
  for (let i = 0; i + 3 <= t.length; i++) out.add(t.slice(i, i + 3))
  return out
}

const dice = (a: Set<string>, b: Set<string>) => {
  if (!a.size || !b.size) return 0
  let shared = 0
  for (const g of a) if (b.has(g)) shared++
  return (2 * shared) / (a.size + b.size)
}

/** The share of `query`'s trigrams that `target` contains: how well a fragment is found inside a title. */
const coverage = (query: Set<string>, target: Set<string>) => {
  if (!query.size) return 0
  let found = 0
  for (const g of query) if (target.has(g)) found++
  return found / query.size
}

/** An author's surname: the last word of the name, folded ("Schölkopf" and "Scholkopf" agree). */
const surname = (name: string) => fold(name).split(' ').filter(Boolean).at(-1) ?? ''

/** A URL reduced to what identifies the work: a DOI, an arXiv id without version, or host and path. */
export function identifier(url?: string): string {
  if (!url) return ''
  const u = url.trim().toLowerCase()
  const doi = u.match(/10\.\d{4,9}\/[^\s?#]+/)
  if (doi) return `doi:${doi[0].replace(/[.,;/]+$/, '')}`
  const arxiv = u.match(/arxiv\.org\/(?:abs|pdf)\/([a-z.-]*\/?\d{4}\.\d{4,5}|[a-z.-]+\/\d{7})(?:v\d+)?/)
  if (arxiv) return `arxiv:${arxiv[1]}`
  // The query stays (OpenReview and YouTube keep the work's id there); tracking parameters and fragments go.
  const [base, query = ''] = u.replace(/#.*$/, '').split('?')
  const kept = query
    .split('&')
    .filter((q) => q && !/^(utm_|ref=|source=)/.test(q))
    .join('&')
  const pathPart = base
    .replace(/^https?:\/\//, '')
    .replace(/^www\./, '')
    .replace(/(\.pdf|\.html?)$/, '')
    .replace(/\/+$/, '')
  return `url:${pathPart}${kept ? `?${kept}` : ''}`
}

function prepare(key: string, e: Entry, line = 0): Ref {
  const normTitle = fold(e.title ?? '')
  return {
    ...e,
    key,
    line,
    normTitle,
    grams: trigrams(normTitle),
    words: new Set(normTitle.split(' ').filter((w) => w && !STOP.has(w))),
    surnames: new Set((e.authors ?? []).map(surname).filter(Boolean)),
    id: identifier(e.url),
  }
}

/** Every entry, with the line its key is on; keys that appear more than once are reported separately. */
function load(): { refs: Ref[]; repeatedKeys: Map<string, number[]> } {
  const text = fs.readFileSync(file, 'utf8')
  const lines = new Map<string, number[]>()
  text.split('\n').forEach((l, i) => {
    const m = l.match(/^([A-Za-z0-9_.-]+):\s*$/)
    if (m) lines.set(m[1], [...(lines.get(m[1]) ?? []), i + 1])
  })
  const data = parse(text, { uniqueKeys: false }) as Record<string, Entry>
  const refs = Object.entries(data).map(([k, e]) => prepare(k, e, lines.get(k)?.[0] ?? 0))
  const repeatedKeys = new Map([...lines].filter(([, at]) => at.length > 1))
  return { refs, repeatedKeys }
}

const surnameOverlap = (a: Set<string>, b: Set<string>) => {
  if (!a.size || !b.size) return NaN
  let shared = 0
  for (const s of a) if (b.has(s)) shared++
  return shared / Math.min(a.size, b.size)
}

type Verdict = 'duplicate' | 'likely' | 'possible' | 'shared-url' | 'different'

/** How alike two entries are, and why: the identifier decides; otherwise the title, backed by authors and year. */
function compare(a: Ref, b: Ref): { score: number; verdict: Verdict; reasons: string[] } {
  const reasons: string[] = []
  const title = a.normTitle && b.normTitle ? dice(a.grams, b.grams) : 0
  const authors = surnameOverlap(a.surnames, b.surnames)
  const sameYear = a.year !== undefined && a.year === b.year
  if (a.id && a.id === b.id) reasons.push(`same ${a.id.split(':')[0]} (${a.id.slice(a.id.indexOf(':') + 1)})`)
  if (title > 0) reasons.push(`title ${title.toFixed(2)}`)
  if (!Number.isNaN(authors)) reasons.push(`authors ${authors.toFixed(2)}`)
  if (sameYear) reasons.push('same year')
  let verdict: Verdict = 'different'
  // One identifier with unrelated titles is a shared page (a docs site, a book cited twice), not a repeated entry.
  if (a.id && a.id === b.id) verdict = title >= 0.5 ? 'duplicate' : 'shared-url'
  // Same title and authors in different years is often a conference and a journal version, or a paper and a book:
  // worth a look, but not certainly one work. Different authors cap a title match at "possible" ("Calculus").
  else if (authors === 0) verdict = title >= 0.75 ? 'possible' : 'different'
  else if (title >= 0.97 && (sameYear || a.year === undefined || b.year === undefined)) verdict = 'duplicate'
  else if (title >= 0.9 || (title >= 0.75 && authors >= 0.5)) verdict = 'likely'
  else if (title >= 0.6 && (authors >= 0.5 || sameYear)) verdict = 'possible'
  const score =
    verdict === 'duplicate' && a.id === b.id ? 1 : title * (Number.isNaN(authors) ? 1 : 0.85 + 0.15 * authors)
  return { score, verdict, reasons }
}

const describe = (r: Ref) => {
  const who = r.authors?.length ? `${surname(r.authors[0])}${r.authors.length > 1 ? ' et al.' : ''}` : '—'
  return `${r.key} (line ${r.line}) · ${who} · ${r.year ?? '—'}\n      ${r.title}\n      ${r.url ?? ''}`
}

// ---------------------------------------------------------------------------------------------------------------------

type Options = { _: string[]; [flag: string]: string | boolean | string[] }

function parseArgs(argv: string[]): Options {
  const opts: Options = { _: [] }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (!a.startsWith('--')) opts._.push(a)
    else if (a === '--json' || a === '--all') opts[a.slice(2)] = true
    else opts[a.slice(2)] = argv[++i] ?? ''
  }
  return opts
}

/** Fuzzy search by any mix of fields; each given field must match, and the scores multiply. */
function search(refs: Ref[], o: Options) {
  const text = o._.join(' ')
  const want = {
    text: text ? trigrams(fold(text)) : null,
    title: typeof o.title === 'string' ? trigrams(fold(o.title)) : null,
    author: typeof o.author === 'string' ? fold(o.author) : null,
    key: typeof o.key === 'string' ? o.key.toLowerCase() : null,
    year: typeof o.year === 'string' ? Number(o.year) : null,
    id: typeof o.url === 'string' ? identifier(o.url) : typeof o.doi === 'string' ? identifier(o.doi) : null,
    venue: typeof o.venue === 'string' ? trigrams(fold(o.venue)) : null,
  }
  const limit = Number(o.limit ?? 10)
  const results = refs
    .map((r) => {
      let score = 1
      if (want.id) score *= r.id === want.id ? 1 : 0
      if (want.year !== null) score *= r.year === want.year ? 1 : r.year && Math.abs(r.year - want.year) === 1 ? 0.6 : 0
      if (want.key) score *= r.key.toLowerCase().includes(want.key) ? 1 : dice(trigrams(want.key), trigrams(r.key))
      if (want.author) {
        const names = (r.authors ?? []).map(fold)
        score *= names.some((n) => n.includes(want.author!))
          ? 1
          : Math.max(0, ...names.map((n) => coverage(trigrams(want.author!), trigrams(n))))
      }
      if (want.title) score *= Math.max(coverage(want.title, r.grams), dice(want.title, r.grams))
      if (want.venue) score *= coverage(want.venue, trigrams(fold(r.venue ?? '')))
      if (want.text) {
        const hay = trigrams(fold([r.key, r.title, ...(r.authors ?? []), r.venue].filter(Boolean).join(' ')))
        score *= coverage(want.text, hay)
      }
      return { r, score }
    })
    .filter((x) => x.score >= Number(o.min ?? 0.5))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
  if (o.json) return void console.log(JSON.stringify(results.map((x) => ({ key: x.r.key, score: x.score }))))
  if (!results.length) return void console.log('no matches')
  for (const { r, score } of results) console.log(`${score.toFixed(2)}  ${describe(r)}`)
}

/** Compare candidate entries with the file: the closest existing entries, a verdict, and key collisions. */
function check(refs: Ref[], o: Options): number {
  let candidates: Ref[]
  if (typeof o.file === 'string') {
    const data = parse(fs.readFileSync(o.file, 'utf8')) as Record<string, Entry>
    candidates = Object.entries(data).map(([k, e]) => prepare(k, e))
  } else {
    const authors = typeof o.author === 'string' ? o.author.split(/\s*;\s*|\s*,\s*(?=[A-Z])/) : []
    candidates = [
      prepare(String(o.key ?? '(new)'), {
        title: typeof o.title === 'string' ? o.title : o._.join(' '),
        authors,
        year: typeof o.year === 'string' ? Number(o.year) : undefined,
        url: typeof o.url === 'string' ? o.url : typeof o.doi === 'string' ? o.doi : undefined,
      }),
    ]
  }
  const keys = new Set(refs.map((r) => r.key))
  let worst = 0
  for (const c of candidates) {
    const matches = refs
      .map((r) => ({ r, ...compare(c, r) }))
      .filter((m) => m.verdict !== 'different')
      .sort((a, b) => b.score - a.score)
      .slice(0, Number(o.limit ?? 5))
    const clash = keys.has(c.key)
    const verdict = matches[0]?.verdict ?? 'new'
    if (verdict === 'duplicate' || verdict === 'likely' || clash) worst = 1
    if (o.json) {
      console.log(JSON.stringify({ key: c.key, keyTaken: clash, verdict, matches: matches.map((m) => m.r.key) }))
      continue
    }
    console.log(`${c.key}: ${verdict.toUpperCase()}${clash ? ' · KEY ALREADY USED' : ''}\n      ${c.title ?? ''}`)
    for (const m of matches) console.log(`  ${m.verdict.padEnd(9)} ${describe(m.r)}\n      ${m.reasons.join(', ')}`)
  }
  return worst
}

/** Scan the file: repeated keys, entries sharing an identifier, and pairs with near-identical titles. */
function duplicates(refs: Ref[], repeatedKeys: Map<string, number[]>, o: Options): number {
  const threshold = Number(o.threshold ?? 0.85)
  const found: { a: Ref; b: Ref; score: number; verdict: Verdict; reasons: string[] }[] = []

  // Candidate pairs share an identifier or at least two title words (one for one-word titles), so not all pairs are
  // compared.
  const byId = new Map<string, Ref[]>()
  const byWord = new Map<string, Ref[]>()
  for (const r of refs) {
    if (r.id) byId.set(r.id, [...(byId.get(r.id) ?? []), r])
    for (const w of r.words) byWord.set(w, [...(byWord.get(w) ?? []), r])
  }
  const seen = new Set<string>()
  const consider = (a: Ref, b: Ref) => {
    if (a === b) return
    const pair = a.key < b.key ? `${a.key}|${b.key}` : `${b.key}|${a.key}`
    if (seen.has(pair)) return
    seen.add(pair)
    const c = compare(a, b)
    if (c.verdict === 'duplicate' || (c.verdict !== 'different' && c.score >= threshold) || c.score >= threshold)
      found.push({ a, b, ...c })
  }
  for (const group of byId.values()) for (const a of group) for (const b of group) consider(a, b)
  for (const a of refs) {
    const counts = new Map<Ref, number>()
    for (const w of a.words) for (const b of byWord.get(w) ?? []) if (b !== a) counts.set(b, (counts.get(b) ?? 0) + 1)
    const need = Math.min(2, a.words.size)
    for (const [b, n] of counts) if (n >= need) consider(a, b)
  }
  found.sort((x, y) => y.score - x.score)

  if (o.json) {
    console.log(
      JSON.stringify({
        repeatedKeys: Object.fromEntries(repeatedKeys),
        pairs: found.map((f) => ({ a: f.a.key, b: f.b.key, score: f.score, verdict: f.verdict, reasons: f.reasons })),
      }),
    )
  } else {
    if (repeatedKeys.size) {
      console.log(`Repeated keys (${repeatedKeys.size}); YAML keeps only the last of each:`)
      for (const [k, at] of repeatedKeys) console.log(`  ${k}: lines ${at.join(', ')}`)
      console.log()
    }
    console.log(`Likely duplicate entries (${found.length}, threshold ${threshold}):`)
    for (const f of found)
      console.log(
        `\n${f.score.toFixed(2)} ${f.verdict}: ${f.reasons.join(', ')}\n  ${describe(f.a)}\n  ${describe(f.b)}`,
      )
  }
  return repeatedKeys.size || found.some((f) => f.verdict === 'duplicate') ? 1 : 0
}

const [command, ...rest] = process.argv.slice(2)
const opts = parseArgs(rest)
const { refs, repeatedKeys } = load()
if (command === 'search') search(refs, opts)
else if (command === 'check') process.exitCode = check(refs, opts)
else if (command === 'duplicates') process.exitCode = duplicates(refs, repeatedKeys, opts)
else {
  console.log('usage: node scripts/references.ts search|check|duplicates [options]  (see the file header)')
  process.exitCode = 2
}
