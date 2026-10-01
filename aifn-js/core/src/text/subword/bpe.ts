/**
 * Byte-pair encoding (Sennrich, Haddow & Birch 2016, after Gage 1994): start from every word split into characters
 * (or UTF-8 bytes) and repeatedly merge the most frequent adjacent pair of symbols, counting each pair once per
 * occurrence weighted by the word's count. The ordered merge list is the tokeniser: encoding replays it.
 */

import type { Status } from 'aifn/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { run, type Algorithm } from 'aifn/foundation/trace'
import type { Tokenisation, TokenPattern } from 'aifn/text/tokenise'
import { byteAlphabet, byteSymbols, textFromByteSymbols } from './bytes'
import { byCodePoint, characterPieces, encodeByWords, wordTable, type Piece, type WordCountsLike } from './words'

/** One merge: the left and right symbols, the merged symbol, and the pair's count when it was chosen. */
export interface BpeMerge {
  readonly left: string
  readonly right: string
  readonly merged: string
  readonly count: number
}

/** Options of {@link bpeSteps} and {@link bpe}. */
export interface BpeOptions {
  /** The number of merges to learn at most (default 1000). */
  merges?: number
  /** Stop once the vocabulary holds this many symbols (default no limit). */
  vocabularySize?: number
  /** Stop when the most frequent pair occurs fewer times than this (default 2: a pair seen once compresses nothing). */
  minCount?: number
  /** `character` (default): base symbols are code points; `byte`: the 256 UTF-8 byte symbols of GPT-2. */
  unit?: 'character' | 'byte'
  /** The end-of-word symbol appended to each word (default `</w>` for characters, none for bytes). */
  endOfWord?: string
  /** The pre-tokeniser an encoder applies to new text (default `words` for characters, `gpt2` for bytes). */
  pattern?: TokenPattern
}

/** The state of BPE training after `t` merges. */
export interface BpeState extends Status {
  /** The distinct training words, in order of first appearance. */
  readonly words: readonly string[]
  /** Their counts (float64 [W]). */
  readonly wordCounts: Tensor
  /** Each word's current segmentation into symbols. */
  readonly segmentations: readonly (readonly string[])[]
  /** The merges so far, in order. */
  readonly merges: readonly BpeMerge[]
  /** The base symbols, then one symbol per merge. */
  readonly vocabulary: readonly string[]
  /** The merge made by the last step (null at step 0 and when training has stopped). */
  readonly merge: BpeMerge | null
  /** The corpus length in symbols, Σ count × segmentation length (falls by the pair count at each merge). */
  readonly symbols: number
  /** True when no further merge is allowed (budget, vocabulary size or minimum count reached). */
  readonly done: boolean
}

/** A trained BPE tokeniser. */
export interface BpeModel {
  readonly kind: 'bpe'
  readonly merges: readonly BpeMerge[]
  readonly vocabulary: readonly string[]
  readonly unit: 'character' | 'byte'
  readonly endOfWord: string
  readonly pattern: TokenPattern
}

const SEP = '\u0000'

/** The symbols of a word before any merge, with their ranges in the word (the end-of-word symbol has zero width). */
function baseSymbols(word: string, unit: 'character' | 'byte', endOfWord: string): Piece[] {
  const pieces = unit === 'byte' ? byteSymbols(word) : characterPieces(word)
  if (endOfWord) pieces.push({ token: endOfWord, start: word.length, end: word.length })
  return pieces
}

/**
 * The count of every adjacent symbol pair, weighted by word counts, in order of first occurrence (reading the words in
 * order, each left to right): the `pairs` as [left, right] and their `counts` (float64 [P]).
 */
export function bpePairCounts(
  segmentations: readonly (readonly string[])[],
  wordCounts: Tensor | readonly number[],
): { pairs: [string, string][]; counts: Tensor } {
  const c = Array.isArray(wordCounts) ? (wordCounts as readonly number[]) : toFlat(wordCounts as Tensor)
  const m = new Map<string, number>()
  segmentations.forEach((seg, w) => {
    for (let i = 0; i + 1 < seg.length; i++) {
      const key = seg[i] + SEP + seg[i + 1]
      m.set(key, (m.get(key) ?? 0) + c[w])
    }
  })
  return {
    pairs: [...m.keys()].map((k) => k.split(SEP) as [string, string]),
    counts: fromData(Float64Array.from(m.values())),
  }
}

/** Replace every non-overlapping occurrence of `left right`, scanning left to right, by `merged`. */
function mergeIn(seg: readonly string[], left: string, right: string, merged: string): string[] {
  const out: string[] = []
  for (let i = 0; i < seg.length; i++) {
    if (i + 1 < seg.length && seg[i] === left && seg[i + 1] === right) {
      out.push(merged)
      i++
    } else out.push(seg[i])
  }
  return out
}

function resolve(options: BpeOptions) {
  const unit = options.unit ?? 'character'
  return {
    unit,
    maxMerges: options.merges ?? 1000,
    vocabularySize: options.vocabularySize ?? Infinity,
    minCount: options.minCount ?? 2,
    endOfWord: options.endOfWord ?? (unit === 'byte' ? '' : '</w>'),
    pattern: options.pattern ?? (unit === 'byte' ? 'gpt2' : 'words'),
  }
}

/**
 * BPE training as a traceable algorithm: step 0 holds the words split into base symbols; each step makes one merge,
 * the most frequent adjacent pair (ties: the pair met first reading the words in order of first appearance, each left
 * to right), and records it with its count. Training stops (`done`) at the merge budget, the vocabulary size, or when
 * the best pair occurs fewer than `minCount` times. Each step recounts every pair, O(S) for S symbols in the word
 * table: clear rather than fast, for corpora of up to a few thousand distinct words.
 */
export function bpeSteps(words: WordCountsLike, options: BpeOptions = {}): Algorithm<void, BpeState> {
  const table = wordTable(words, 'bpeSteps')
  const o = resolve(options)
  const counts = fromData(Float64Array.from(table.counts))
  const stopped = (merges: number, vocabulary: number) => merges >= o.maxMerges || vocabulary >= o.vocabularySize
  return {
    name: 'bpe',
    init: () => {
      const segmentations = table.words.map((w) => baseSymbols(w, o.unit, o.endOfWord).map((p) => p.token))
      const seen = new Set(segmentations.flat())
      const base =
        o.unit === 'byte' ? [...byteAlphabet()] : [...seen].filter((s) => s !== o.endOfWord).sort(byCodePoint)
      if (o.endOfWord) base.push(o.endOfWord)
      const symbols = segmentations.reduce((acc, seg, w) => acc + seg.length * table.counts[w], 0)
      return {
        t: 0,
        words: table.words,
        wordCounts: counts,
        segmentations,
        merges: [],
        vocabulary: base,
        merge: null,
        symbols,
        done: stopped(0, base.length),
      }
    },
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1, merge: null }
      const { pairs, counts: pc } = bpePairCounts(s.segmentations, s.wordCounts)
      const c = pc.data
      let best = -1
      for (let k = 0; k < pairs.length; k++) if (best < 0 || c[k] > c[best]) best = k
      if (best < 0 || c[best] < o.minCount) return { ...s, t: s.t + 1, merge: null, done: true }
      const [left, right] = pairs[best]
      const merge: BpeMerge = { left, right, merged: left + right, count: c[best] }
      const segmentations = s.segmentations.map((seg) => mergeIn(seg, left, right, merge.merged))
      const merges = [...s.merges, merge]
      // A merged symbol can repeat an earlier one (two routes to one string); the vocabulary lists it once.
      const vocabulary = s.vocabulary.includes(merge.merged) ? s.vocabulary : [...s.vocabulary, merge.merged]
      return {
        ...s,
        t: s.t + 1,
        segmentations,
        merges,
        vocabulary,
        merge,
        symbols: s.symbols - merge.count,
        done: stopped(merges.length, vocabulary.length),
      }
    },
    done: (s) => s.done,
  }
}

/** The tokeniser of a BPE training state (the merges learned so far). */
export function bpeModel(state: BpeState, options: BpeOptions = {}): BpeModel {
  const o = resolve(options)
  return {
    kind: 'bpe',
    merges: state.merges,
    vocabulary: state.vocabulary,
    unit: o.unit,
    endOfWord: o.endOfWord,
    pattern: o.pattern,
  }
}

/** Train BPE to the end (see {@link bpeSteps}) and return the tokeniser. */
export function bpe(words: WordCountsLike, options: BpeOptions = {}): BpeModel {
  const o = resolve(options)
  const final = run(bpeSteps(words, options), undefined, Math.min(o.maxMerges, 1e6) + 1)
  return bpeModel(final, options)
}

const ranks = new WeakMap<BpeModel, Map<string, number>>()

function rankTable(model: BpeModel): Map<string, number> {
  let r = ranks.get(model)
  if (!r) {
    r = new Map()
    model.merges.forEach((m, k) => {
      const key = m.left + SEP + m.right
      if (!r!.has(key)) r!.set(key, k)
    })
    ranks.set(model, r)
  }
  return r
}

/**
 * Segment one word with a BPE tokeniser, as pieces with their ranges in the word: start from base symbols and apply
 * merges in the order they were learned, by repeatedly merging the adjacent pair of lowest merge rank (equivalent to
 * replaying the list, and as GPT-2 encodes). `upTo` replays only the first `upTo` merges.
 */
export function bpeSegment(model: BpeModel, word: string, upTo = Infinity): Piece[] {
  const r = rankTable(model)
  let pieces = baseSymbols(word, model.unit, model.endOfWord)
  for (;;) {
    let best = -1
    let bestRank = Infinity
    for (let i = 0; i + 1 < pieces.length; i++) {
      const k = r.get(pieces[i].token + SEP + pieces[i + 1].token)
      if (k !== undefined && k < upTo && k < bestRank) [best, bestRank] = [i, k]
    }
    if (best < 0) return pieces
    const { left, right, merged } = model.merges[bestRank]
    const next: Piece[] = []
    for (let i = 0; i < pieces.length; i++) {
      if (i + 1 < pieces.length && pieces[i].token === left && pieces[i + 1].token === right) {
        next.push({ token: merged, start: pieces[i].start, end: pieces[i + 1].end })
        i++
      } else next.push(pieces[i])
    }
    pieces = next
  }
}

/** Encode text with a BPE tokeniser: pre-tokenise by the model's pattern, then segment each word. */
export function bpeEncode(model: BpeModel, text: string, options: { upTo?: number } = {}): Tokenisation {
  return encodeByWords(text, model.pattern, (w) => bpeSegment(model, w, options.upTo))
}

/**
 * Text from BPE tokens: the end-of-word symbol becomes a space (character level), byte symbols are decoded as UTF-8
 * (byte level). The inverse of encoding up to white space between words.
 */
export function bpeDecode(model: BpeModel, tokens: readonly string[]): string {
  const joined = tokens.join('')
  if (model.unit === 'byte') return textFromByteSymbols(joined)
  if (!model.endOfWord) return joined
  return joined.split(model.endOfWord).join(' ').trimEnd()
}
