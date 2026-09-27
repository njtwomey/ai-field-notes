/**
 * Classical text processing for the text-representation widgets: normalisation, tokenisation, stop words, the
 * original Porter stemmer, a small dictionary lemmatiser, n-grams, weighting, shingles and MinHash. Everything is
 * small and synchronous so that it can run on every keystroke.
 */

// ---------------------------------------------------------------------------------------------------------------------
// Normalisation and tokenisation

export type NormaliseOptions = {
  /** NFKC also folds compatibility characters: ligatures, superscripts, full-width forms. */
  unicode: 'none' | 'NFC' | 'NFKC'
  casefold: boolean
  stripAccents: boolean
  expandContractions: boolean
  /** Replace every digit sequence by the placeholder token `<num>`. */
  numbers: boolean
}

const CONTRACTIONS: [RegExp, string][] = [
  [/\bwon't\b/g, 'will not'],
  [/\bcan't\b/g, 'can not'],
  [/\bshan't\b/g, 'shall not'],
  [/n't\b/g, ' not'],
  [/'re\b/g, ' are'],
  [/'ve\b/g, ' have'],
  [/'ll\b/g, ' will'],
  [/'m\b/g, ' am'],
  [/'d\b/g, ' would'],
  // "it's" is "it is"; after a noun, 's is usually the possessive and is dropped.
  [/\b(it|he|she|that|what|there|here|who|where)'s\b/gi, '$1 is'],
  [/\blet's\b/gi, 'let us'],
  [/'s\b/g, ''],
]

export function normalise(text: string, o: NormaliseOptions): string {
  let s = o.unicode === 'none' ? text : text.normalize(o.unicode)
  // Typographic apostrophes behave like the ASCII one for contractions.
  s = s.replace(/[’ʼ]/g, "'")
  if (o.casefold) s = s.toLowerCase()
  if (o.stripAccents)
    s = s
      .normalize('NFD')
      .replace(/\p{M}+/gu, '')
      .normalize('NFC')
  if (o.expandContractions) for (const [re, rep] of CONTRACTIONS) s = s.replace(re, rep)
  if (o.numbers) s = s.replace(/\p{N}+(?:[.,]\p{N}+)*/gu, ' <num> ')
  return s
}

/** Words (letters or digits, with inner apostrophes or hyphens), the `<num>` placeholder, and single punctuation marks. */
const TOKEN = /<num>|[\p{L}\p{N}]+(?:['-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu

export function tokenise(text: string, keepPunctuation = false): string[] {
  const out = text.match(TOKEN) ?? []
  return keepPunctuation ? out : out.filter((t) => t === '<num>' || /[\p{L}\p{N}]/u.test(t))
}

// ---------------------------------------------------------------------------------------------------------------------
// Stop words: NLTK's English list (198 words in NLTK 3.10), derived from the Snowball list.

export const STOP_WORDS = new Set(
  (
    "a about above after again against ain all am an and any are aren aren't as at be because been before being below " +
    "between both but by can couldn couldn't d did didn didn't do does doesn doesn't doing don don't down during each " +
    "few for from further had hadn hadn't has hasn hasn't have haven haven't having he he'd he'll her here hers herself " +
    "he's him himself his how i i'd if i'll i'm in into is isn isn't it it'd it'll it's its itself i've just ll m ma me " +
    "mightn mightn't more most mustn mustn't my myself needn needn't no nor not now o of off on once only or other our " +
    "ours ourselves out over own re s same shan shan't she she'd she'll she's should shouldn shouldn't should've so " +
    "some such t than that that'll the their theirs them themselves then there these they they'd they'll they're " +
    "they've this those through to too under until up ve very was wasn wasn't we we'd we'll we're were weren weren't " +
    "we've what when where which while who whom why will with won won't wouldn wouldn't y you you'd you'll your you're " +
    "yours yourself yourselves you've"
  ).split(' '),
)

export const NEGATIONS = new Set(['no', 'not', 'nor', "don't", "isn't", "wasn't", "won't", "can't", "didn't", 'never'])

// ---------------------------------------------------------------------------------------------------------------------
// The Porter stemmer, as published in 1980 (NLTK's ORIGINAL_ALGORITHM mode), with a trace of the rules that fire.

export type PorterStep = { step: string; rule: string; before: string; after: string }

function isConsonant(w: string, i: number): boolean {
  const c = w[i]
  if ('aeiou'.includes(c)) return false
  if (c === 'y') return i === 0 ? true : !isConsonant(w, i - 1)
  return true
}

/** The C/V form of a word, e.g. "trouble" → "CCVVCCV". */
export function cvForm(w: string): string {
  return [...w].map((_, i) => (isConsonant(w, i) ? 'C' : 'V')).join('')
}

/** Porter's measure m: the number of VC sequences in the form [C](VC)^m[V]. */
export function measure(stem: string): number {
  const form = cvForm(stem).replace(/C+/g, 'C').replace(/V+/g, 'V')
  return (form.match(/VC/g) ?? []).length
}

const hasVowel = (s: string) => /V/.test(cvForm(s))
const endsDoubleConsonant = (s: string) =>
  s.length >= 2 && s[s.length - 1] === s[s.length - 2] && isConsonant(s, s.length - 1)
/** *o: the stem ends consonant–vowel–consonant, and the last consonant is not w, x or y. */
const endsCvc = (s: string) => {
  const n = s.length
  if (n < 3) return false
  return isConsonant(s, n - 3) && !isConsonant(s, n - 2) && isConsonant(s, n - 1) && !'wxy'.includes(s[n - 1])
}

type Rule = [suffix: string, replacement: string, condition: (stem: string) => boolean, label?: string]
const m0 = (s: string) => measure(s) > 0
const m1 = (s: string) => measure(s) > 1

const STEP2: Rule[] = [
  ['ational', 'ate', m0],
  ['tional', 'tion', m0],
  ['enci', 'ence', m0],
  ['anci', 'ance', m0],
  ['izer', 'ize', m0],
  ['abli', 'able', m0],
  ['alli', 'al', m0],
  ['entli', 'ent', m0],
  ['eli', 'e', m0],
  ['ousli', 'ous', m0],
  ['ization', 'ize', m0],
  ['ation', 'ate', m0],
  ['ator', 'ate', m0],
  ['alism', 'al', m0],
  ['iveness', 'ive', m0],
  ['fulness', 'ful', m0],
  ['ousness', 'ous', m0],
  ['aliti', 'al', m0],
  ['iviti', 'ive', m0],
  ['biliti', 'ble', m0],
]
const STEP3: Rule[] = [
  ['icate', 'ic', m0],
  ['ative', '', m0],
  ['alize', 'al', m0],
  ['iciti', 'ic', m0],
  ['ical', 'ic', m0],
  ['ful', '', m0],
  ['ness', '', m0],
]
const STEP4: Rule[] = [
  'al ance ence er ic able ible ant ement ment ent'.split(' ').map((s): Rule => [s, '', m1]),
  [['ion', '', (s: string) => m1(s) && /[st]$/.test(s), '(m>1 and (*S or *T)) ION → ∅'] as Rule],
  'ou ism ate iti ous ive ize'.split(' ').map((s): Rule => [s, '', m1]),
].flat()

const CONDITION_LABEL = new Map<Rule[], string>([
  [STEP2, 'm>0'],
  [STEP3, 'm>0'],
  [STEP4, 'm>1'],
])

/** Longest matching suffix wins; if its condition fails, the step does nothing. */
function applyRules(w: string, rules: Rule[], step: string, trace: PorterStep[]): string {
  let best: Rule | null = null
  for (const r of rules) if (w.endsWith(r[0]) && (!best || r[0].length > best[0].length)) best = r
  if (!best) return w
  const [suffix, rep, cond, label] = best
  const stem = w.slice(0, w.length - suffix.length)
  if (!cond(stem)) return w
  const after = stem + rep
  const rule = label ?? `(${CONDITION_LABEL.get(rules)}) ${suffix.toUpperCase()} → ${rep.toUpperCase() || '∅'}`
  trace.push({ step, rule, before: w, after })
  return after
}

export function porterStem(word: string, trace: PorterStep[] = []): string {
  let w = word.toLowerCase()
  if (w.length <= 2 || !/^[a-z]+$/.test(w)) return w
  const push = (step: string, rule: string, after: string) => {
    trace.push({ step, rule, before: w, after })
    w = after
  }

  // Step 1a: plurals.
  if (w.endsWith('sses')) push('1a', 'SSES → SS', w.slice(0, -2))
  else if (w.endsWith('ies')) push('1a', 'IES → I', w.slice(0, -2))
  else if (w.endsWith('ss')) {
    // SS → SS changes nothing, but it stops the S rule.
  } else if (w.endsWith('s')) push('1a', 'S → ∅', w.slice(0, -1))

  // Step 1b: -ed and -ing, with a clean-up when either is removed.
  let cleanup = false
  if (w.endsWith('eed')) {
    if (m0(w.slice(0, -3))) push('1b', '(m>0) EED → EE', w.slice(0, -1))
  } else if (w.endsWith('ed') && hasVowel(w.slice(0, -2))) {
    push('1b', '(*v*) ED → ∅', w.slice(0, -2))
    cleanup = true
  } else if (w.endsWith('ing') && hasVowel(w.slice(0, -3))) {
    push('1b', '(*v*) ING → ∅', w.slice(0, -3))
    cleanup = true
  }
  if (cleanup) {
    if (w.endsWith('at')) push('1b', 'AT → ATE', w + 'e')
    else if (w.endsWith('bl')) push('1b', 'BL → BLE', w + 'e')
    else if (w.endsWith('iz')) push('1b', 'IZ → IZE', w + 'e')
    else if (endsDoubleConsonant(w) && !'lsz'.includes(w[w.length - 1]))
      push('1b', '(*d and not (*L or *S or *Z)) → single letter', w.slice(0, -1))
    else if (measure(w) === 1 && endsCvc(w)) push('1b', '(m=1 and *o) → E', w + 'e')
  }

  // Step 1c: terminal y.
  if (w.endsWith('y') && hasVowel(w.slice(0, -1))) push('1c', '(*v*) Y → I', w.slice(0, -1) + 'i')

  w = applyRules(w, STEP2, '2', trace)
  w = applyRules(w, STEP3, '3', trace)
  w = applyRules(w, STEP4, '4', trace)

  // Step 5a: final e. Step 5b: final ll.
  if (w.endsWith('e')) {
    const stem = w.slice(0, -1)
    const m = measure(stem)
    if (m > 1) push('5a', '(m>1) E → ∅', stem)
    else if (m === 1 && !endsCvc(stem)) push('5a', '(m=1 and not *o) E → ∅', stem)
  }
  if (measure(w) > 1 && w.endsWith('ll')) push('5b', '(m>1 and *d and *L) → single letter', w.slice(0, -1))
  return w
}

// ---------------------------------------------------------------------------------------------------------------------
// A small dictionary lemmatiser in the style of WordNet's morphy: an exception list for irregular forms, then suffix
// rules whose output must be a known base form. Words it cannot resolve are returned unchanged.

const EXCEPTIONS: Record<string, string> = {
  am: 'be',
  is: 'be',
  are: 'be',
  was: 'be',
  were: 'be',
  been: 'be',
  being: 'be',
  has: 'have',
  had: 'have',
  having: 'have',
  does: 'do',
  did: 'do',
  done: 'do',
  ran: 'run',
  went: 'go',
  gone: 'go',
  saw: 'see',
  seen: 'see',
  ate: 'eat',
  eaten: 'eat',
  caught: 'catch',
  taught: 'teach',
  thought: 'think',
  bought: 'buy',
  brought: 'bring',
  made: 'make',
  said: 'say',
  took: 'take',
  taken: 'take',
  gave: 'give',
  given: 'give',
  wrote: 'write',
  written: 'write',
  sang: 'sing',
  sung: 'sing',
  sat: 'sit',
  fed: 'feed',
  slept: 'sleep',
  kept: 'keep',
  left: 'leave',
  felt: 'feel',
  found: 'find',
  told: 'tell',
  mice: 'mouse',
  geese: 'goose',
  children: 'child',
  men: 'man',
  women: 'woman',
  feet: 'foot',
  teeth: 'tooth',
  people: 'person',
  wolves: 'wolf',
  knives: 'knife',
  better: 'good',
  best: 'good',
  worse: 'bad',
  worst: 'bad',
}

const LEXICON = new Set(
  (
    'a an the be have do go run see eat catch teach think buy bring make say take give write sing sleep keep leave ' +
    'feel find tell mouse goose child man woman foot tooth person good bad cat dog bird mat log garden film movie ' +
    'chase bark like play sit study university universe organ organization organize general generous relate ' +
    'relation relational happy happiness run runner walk talk jump fox box church dish wish class glass study city ' +
    'party story berry lady boy toy day key try cry fly die lie tie hope hop plan stop drop love live move use ' +
    'computer compute computation connect connection argue argument wolf leaf knife life wife shelf bus gas ' +
    'meeting meet building build writing reading read book paper word text document query search engine index ' +
    'mouse cheese milk bone chew drink fish swim ball throw fetch hunt tree car automobile engine drive boring ' +
    'bore interest interesting great terrible plot actor act watch enjoy hate kind nice fine friend enemy start course ' +
    'cafe resume away while all mat sit sing pet vet feed grain'
  ).split(' '),
)

/** morphy's detachment rules, per part of speech. */
const MORPHY: Record<'noun' | 'verb' | 'adj', [string, string][]> = {
  noun: [
    ['s', ''],
    ['ses', 's'],
    ['xes', 'x'],
    ['zes', 'z'],
    ['ches', 'ch'],
    ['shes', 'sh'],
    ['men', 'man'],
    ['ies', 'y'],
  ],
  verb: [
    ['s', ''],
    ['ies', 'y'],
    ['es', 'e'],
    ['es', ''],
    ['ed', 'e'],
    ['ed', ''],
    ['ing', 'e'],
    ['ing', ''],
  ],
  adj: [
    ['er', ''],
    ['est', ''],
    ['er', 'e'],
    ['est', 'e'],
  ],
}

export type Pos = 'noun' | 'verb' | 'adj'

export function lemmatise(word: string, pos?: Pos): string {
  const w = word.toLowerCase()
  if (w in EXCEPTIONS) return EXCEPTIONS[w]
  if (LEXICON.has(w)) return w
  const tags: Pos[] = pos ? [pos] : ['noun', 'verb', 'adj']
  for (const p of tags) {
    for (const [suffix, rep] of MORPHY[p]) {
      if (!w.endsWith(suffix)) continue
      const base = w.slice(0, w.length - suffix.length) + rep
      if (LEXICON.has(base)) return base
      // Doubled final consonant before -ing or -ed: "running" → "runn" → "run".
      if (p === 'verb' && rep === '' && /(.)\1$/.test(base) && LEXICON.has(base.slice(0, -1))) return base.slice(0, -1)
    }
  }
  return w
}

// ---------------------------------------------------------------------------------------------------------------------
// Features and weighting

export function ngrams(tokens: string[], n: number, joiner = ' '): string[] {
  const out: string[] = []
  for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(joiner))
  return out
}

export function counts<T>(items: T[]): Map<T, number> {
  const m = new Map<T, number>()
  for (const x of items) m.set(x, (m.get(x) ?? 0) + 1)
  return m
}

// ---------------------------------------------------------------------------------------------------------------------
// Shingles and MinHash

/** The set of k-shingles of a text: runs of k consecutive words, or k consecutive characters. */
export function shingles(text: string, unit: 'word' | 'char', k: number): Set<string> {
  const clean = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .replace(/\s+/g, ' ')
    .trim()
  if (unit === 'word') {
    const words = clean ? clean.split(' ') : []
    return new Set(words.length < k ? (words.length ? [words.join(' ')] : []) : ngrams(words, k))
  }
  const chars = [...clean]
  if (chars.length < k) return new Set(chars.length ? [clean] : [])
  const out = new Set<string>()
  for (let i = 0; i + k <= chars.length; i++) out.add(chars.slice(i, i + k).join(''))
  return out
}

export function jaccard<T>(a: Set<T>, b: Set<T>): number {
  if (a.size === 0 && b.size === 0) return 1
  let inter = 0
  for (const x of a) if (b.has(x)) inter++
  return inter / (a.size + b.size - inter)
}

/** 32-bit FNV-1a hash of a string. */
export function fnv1a(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** MurmurHash3's finaliser: a bijection on 32-bit integers that mixes every input bit into every output bit. */
function fmix32(x: number): number {
  x ^= x >>> 16
  x = Math.imul(x, 0x85ebca6b)
  x ^= x >>> 13
  x = Math.imul(x, 0xc2b2ae35)
  x ^= x >>> 16
  return x >>> 0
}

/**
 * MinHash signature: for each of the `salts`, the minimum over the set of a salted hash. Each salted hash acts as a
 * random ordering of the shingles, so two sets' minima agree with probability equal to their Jaccard similarity.
 */
export function minhash(set: Set<string>, salts: number[]): number[] {
  const hashes = [...set].map(fnv1a)
  return salts.map((salt) => {
    let min = 0xffffffff
    for (const h of hashes) {
      const v = fmix32(h ^ salt)
      if (v < min) min = v
    }
    return min
  })
}
