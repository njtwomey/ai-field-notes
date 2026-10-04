/**
 * The sources the Huffman showcase codes: named distributions over symbols a, b, c, … (or English letters), each with
 * its symbol names. Symbol k keeps palette slot k everywhere on the page (tree, table, bars, bit string); the palette
 * has eight slots, so a source with more symbols shows the ninth and later in the muted chrome colour.
 */
import { chrome, SLOTS, seriesColor, type Mode } from '@lab/design'

export type SourceName = 'zipf' | 'uniform' | 'dyadic' | 'skewed' | 'ties' | 'english'

export const SOURCE_OPTIONS: { value: SourceName; label: string }[] = [
  { value: 'zipf', label: 'Zipf(s)' },
  { value: 'uniform', label: 'uniform' },
  { value: 'dyadic', label: 'dyadic (powers of ½)' },
  { value: 'skewed', label: 'skewed (one likely symbol)' },
  { value: 'ties', label: 'ties: .4 .2 .2 .1 .1' },
  { value: 'english', label: 'English letters' },
]

/**
 * English letter frequencies in per cent, commonest first (Lewand, 2000, "Cryptological Mathematics", as tabulated in
 * Wikipedia's "Letter frequency").
 */
const ENGLISH: [string, number][] = [
  ['e', 12.7],
  ['t', 9.06],
  ['a', 8.17],
  ['o', 7.51],
  ['i', 6.97],
  ['n', 6.75],
  ['s', 6.33],
  ['h', 6.09],
  ['r', 5.99],
  ['d', 4.25],
  ['l', 4.03],
  ['c', 2.78],
  ['u', 2.76],
  ['m', 2.41],
  ['w', 2.36],
  ['f', 2.23],
  ['g', 2.02],
  ['y', 1.97],
  ['p', 1.93],
  ['b', 1.29],
  ['v', 0.98],
  ['k', 0.77],
  ['j', 0.15],
  ['x', 0.15],
  ['q', 0.1],
  ['z', 0.07],
]

const LETTERS = 'abcdefghijklmnopqrstuvwxyz'

export type Source = { names: string[]; p: number[]; note: string }

const normalise = (w: number[]) => {
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((v) => v / total)
}

/** The distribution of a source with K symbols (ignored by the fixed ones), Zipf exponent s and skew q. */
export function makeSource(name: SourceName, k: number, s: number, q: number): Source {
  const letters = (n: number) => LETTERS.slice(0, n).split('')
  switch (name) {
    case 'zipf':
      return {
        names: letters(k),
        p: normalise(Array.from({ length: k }, (_, i) => 1 / (i + 1) ** s)),
        note: `pₖ ∝ 1/kˢ with s = ${s}`,
      }
    case 'uniform':
      return { names: letters(k), p: Array.from({ length: k }, () => 1 / k), note: `every symbol 1/${k}` }
    case 'dyadic':
      return {
        names: letters(k),
        p: Array.from({ length: k }, (_, i) => 2 ** -Math.min(i + 1, k - 1)),
        note: 'every probability a power of ½, so −log₂ p is a whole number',
      }
    case 'skewed':
      return {
        names: letters(k),
        p: [q, ...Array.from({ length: k - 1 }, () => (1 - q) / (k - 1))],
        note: `a has probability ${q}, the rest share ${Number((1 - q).toFixed(2))}`,
      }
    case 'ties':
      return { names: letters(5), p: [0.4, 0.2, 0.2, 0.1, 0.1], note: 'merged nodes tie with original leaves' }
    case 'english': {
      const top = ENGLISH.slice(0, k)
      return {
        names: top.map(([c]) => c),
        p: normalise(top.map(([, f]) => f)),
        note: k < ENGLISH.length ? `the ${k} commonest letters, renormalised` : 'all 26 letters',
      }
    }
  }
}

/** The colour of symbol k: its palette slot, or the muted chrome colour past the eighth. */
export function symbolColor(mode: Mode, k: number): string {
  return k < SLOTS ? seriesColor(mode, k) : chrome(mode).muted
}

/** Probabilities as ".07", "1"; small ones with more digits. */
export function fmtP(v: number): string {
  if (v === 0) return '0'
  if (v >= 0.995) return '1'
  const digits = v < 0.01 ? 3 : 2
  return v.toFixed(digits).replace(/^0\./, '.')
}
