/**
 * Source and channel coding: Huffman, Shannon–Fano and Shannon codes with their expected lengths, the Kraft sum,
 * the arithmetic-coding interval of a message, Hamming distance and weight, the minimum distance of a code, and the
 * Hamming (sphere-packing), Singleton and Plotkin bounds on the size of a code. Codes are binary; codewords are
 * strings of '0' and '1'.
 */

import { treeFromChildren, type Tree } from 'aifn/graph'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn/tensor'
import { run, type Algorithm } from 'aifn/trace'
import { flatProbabilities, type Probabilities } from './measures'

/** A prefix code for symbols 0 … K − 1. */
export type PrefixCode = {
  /** The codeword of each symbol. */
  codewords: string[]
  /** The length of each codeword (int32, length K). */
  lengths: Tensor
  /** Σₖ pₖ ℓₖ in bits per symbol. */
  expectedLength: number
  /** The entropy H(p) in bits, the lower bound on the expected length of any prefix code. */
  entropy: number
}

/** Node data of a Huffman tree: leaves carry their `symbol`, internal nodes −1; every node its total probability. */
export type HuffmanNodeData = { probability: number; symbol: number }
/** Edge data of a Huffman tree: the bit the branch appends (0 for the first node merged, 1 for the second). */
export type HuffmanEdgeData = { bit: 0 | 1 }
/**
 * A Huffman tree as an `aifn/graph` `Tree` (binary, `arity: 2`): leaves are ids 0 … K − 1 (symbol k is node k),
 * internal nodes follow in merge order and the root is last. Edges are labelled '0' and '1' (slot 0 is the 0 branch);
 * a symbol's codeword is the edge labels on its path from the root.
 */
export type HuffmanTree = Tree<HuffmanNodeData, HuffmanEdgeData>

/** One state of Huffman's algorithm: a forest of merged nodes and the queue of roots still to merge. */
export type HuffmanState = {
  /** Every node so far: leaves 0 … K − 1, then one internal node per merge. */
  nodes: readonly {
    probability: number
    symbol: number
    parent: number | null
    /** [the 0 branch, the 1 branch], or null for a leaf. */
    children: readonly [number, number] | null
  }[]
  /** The roots still to merge, least probable first (ties: older first). */
  queue: readonly number[]
  /** The two nodes merged by the last step ([0-branch, 1-branch]), or null. */
  merged: readonly [number, number] | null
  /** The node the last step created, or −1. */
  created: number
  /**
   * The codewords so far. A merge prepends a bit to every symbol under the two merged nodes, so codewords grow from
   * their last bit to their first as the queue shrinks.
   */
  codewords: readonly string[]
  done: boolean
}

/**
 * Huffman's algorithm as a traceable `Algorithm` (Huffman, 1952). Options: the probabilities (any shape, flattened).
 * Each step merges the two least probable roots of the queue into a new node (ties broken by creation order, older
 * first; the first taken gets the 0 branch) and prepends 0 or 1 to the codewords under them. The run is done when one
 * root is left: K − 1 steps for K symbols. `huffmanTree(state)` turns the final state into a `Tree`.
 */
export const huffmanSteps: Algorithm<Probabilities, HuffmanState> = {
  name: 'huffman',
  init: (probabilities) => {
    const p = flatProbabilities(probabilities, 'huffmanSteps')
    if (p.length === 0) throw new RangeError('huffmanSteps: needs at least one symbol')
    const nodes = p.map((v, k) => ({ probability: v, symbol: k, parent: null, children: null }))
    return {
      nodes,
      queue: sortQueue(
        nodes,
        nodes.map((_, k) => k),
      ),
      merged: null,
      created: -1,
      codewords: p.map(() => ''),
      done: p.length === 1,
    }
  },
  step: (s) => {
    if (s.done) return s
    const [a, b, ...rest] = s.queue
    const id = s.nodes.length
    const nodes = s.nodes.map((n, i) => (i === a || i === b ? { ...n, parent: id } : n))
    nodes.push({
      probability: s.nodes[a].probability + s.nodes[b].probability,
      symbol: -1,
      parent: null,
      children: [a, b],
    })
    const codewords = [...s.codewords]
    const prepend = (v: number, bit: string) => {
      const c = nodes[v].children
      if (c === null) codewords[v] = bit + codewords[v]
      else c.forEach((w) => prepend(w, bit))
    }
    prepend(a, '0')
    prepend(b, '1')
    const queue = sortQueue(nodes, [...rest, id])
    return { nodes, queue, merged: [a, b], created: id, codewords, done: queue.length === 1 }
  },
  done: (s) => s.done,
}

/** Queue order: probability, then creation order. A sorted list is enough for the sizes a figure uses. */
function sortQueue(nodes: HuffmanState['nodes'], ids: number[]): number[] {
  return [...ids].sort((x, y) => nodes[x].probability - nodes[y].probability || x - y)
}

/** The Huffman tree of a finished `huffmanSteps` state as a `Tree`, rooted at the last node created. */
export function huffmanTree(state: HuffmanState): HuffmanTree {
  if (!state.done) throw new Error('huffmanTree: the state is not finished (the queue holds more than one root)')
  const root = state.queue[0]
  const tree = treeFromChildren<HuffmanNodeData, HuffmanEdgeData>(
    state.nodes.map((n) => (n.children ? [...n.children] : [])),
    root,
    {
      data: (i) => ({ probability: state.nodes[i].probability, symbol: state.nodes[i].symbol }),
      edge: (c, p) => {
        const bit = state.nodes[p].children![0] === c ? 0 : 1
        return { bit, label: String(bit) }
      },
    },
  )
  // Mark sides so a binary layout keeps the 0 branch on the left.
  for (const n of tree.nodes) if (n.parent !== null) n.slot = tree.edges[n.id]!.bit
  return { ...tree, arity: 2 }
}

function entropyBits(p: number[]): number {
  let h = 0
  for (const v of p) if (v > 0) h -= v * Math.log2(v)
  return h
}

function finish(p: number[], codewords: string[]): PrefixCode {
  const lengths = Int32Array.from(codewords, (c) => c.length)
  let expected = 0
  p.forEach((v, k) => (expected += v * lengths[k]))
  return { codewords, lengths: fromData(lengths, [lengths.length]), expectedLength: expected, entropy: entropyBits(p) }
}

/**
 * A binary Huffman code (Huffman, 1952, "A method for the construction of minimum-redundancy codes", Proc. IRE 40):
 * repeatedly merge the two least probable nodes (`huffmanSteps` run to the end). The result is an optimal prefix code,
 * with H(p) ≤ expected length < H(p) + 1. Ties are broken by creation order (older nodes first), and the first node
 * taken gets the 0 branch, so the code is deterministic. A single symbol gets the codeword "0". Returns the code and
 * its `tree` (a `HuffmanTree`: leaves are ids 0 … K − 1, the root is last).
 */
export function huffmanCode(probabilities: Probabilities): PrefixCode & { tree: HuffmanTree } {
  const p = flatProbabilities(probabilities, 'huffmanCode')
  if (p.length === 0) throw new RangeError('huffmanCode: needs at least one symbol')
  const s = run(huffmanSteps, p, p.length)
  const tree = huffmanTree(s)
  return { ...finish(p, p.length === 1 ? ['0'] : [...s.codewords]), tree }
}

/**
 * A Shannon–Fano code by Fano's method (Fano, 1949, "The transmission of information", MIT RLE TR 65): sort the
 * symbols by decreasing probability (stable), split the list where the two parts' totals are closest, give the first
 * part 0 and the second 1, and recurse. Not always optimal; its expected length is below H(p) + 2.
 */
export function shannonFanoCode(probabilities: Probabilities): PrefixCode {
  const p = flatProbabilities(probabilities, 'shannonFanoCode')
  const order = p.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b)
  const codewords = new Array<string>(p.length).fill('')
  const split = (items: number[], prefix: string) => {
    if (items.length === 1) {
      codewords[items[0]] = prefix || '0'
      return
    }
    const total = items.reduce((s, k) => s + p[k], 0)
    let left = 0
    let best = 1
    let bestGap = Infinity
    for (let i = 1; i < items.length; i++) {
      left += p[items[i - 1]]
      const gap = Math.abs(total - 2 * left)
      if (gap < bestGap) {
        bestGap = gap
        best = i
      }
    }
    split(items.slice(0, best), prefix + '0')
    split(items.slice(best), prefix + '1')
  }
  split(order, '')
  return finish(p, codewords)
}

/**
 * The Shannon code (Shannon, 1948, §9): symbols sorted by decreasing probability get lengths ℓₖ = ⌈−log₂ pₖ⌉ and, as
 * codewords, the first ℓₖ bits of the binary expansion of the cumulative probability of the symbols before them.
 * Its expected length is below H(p) + 1. Zero-probability symbols get no codeword ('').
 */
export function shannonCode(probabilities: Probabilities): PrefixCode {
  const p = flatProbabilities(probabilities, 'shannonCode')
  const order = p.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b)
  const codewords = new Array<string>(p.length).fill('')
  let cumulative = 0
  for (const k of order) {
    if (p[k] === 0) continue
    const length = Math.max(1, Math.ceil(-Math.log2(p[k]) - 1e-12))
    let bits = ''
    let f = cumulative
    for (let i = 0; i < length; i++) {
      f *= 2
      const bit = f >= 1 ? 1 : 0
      bits += bit
      f -= bit
    }
    codewords[k] = bits
    cumulative += p[k]
  }
  return finish(p, codewords)
}

/**
 * The Kraft sum Σₖ D^{−ℓₖ} of codeword lengths for a D-ary alphabet (default 2). A prefix code with these lengths
 * exists if and only if the sum is at most 1 (Kraft, 1949; McMillan, 1956, for uniquely decodable codes).
 */
export function kraftSum(lengths: ArrayLike<number> | Tensor, arity = 2): number {
  const values = isTensor(lengths) ? toFlat(lengths) : Array.from(lengths)
  return values.reduce((s, l) => s + arity ** -l, 0)
}

/** The interval of a message under arithmetic coding, and how it narrows symbol by symbol. */
export type ArithmeticInterval = {
  low: number
  high: number
  /** high − low = Π p(symbolᵢ), the probability of the message. */
  width: number
  /** ⌈−log₂ width⌉ + 1: enough bits to name a binary fraction inside the interval (Cover and Thomas, 2006, §13.3). */
  bits: number
  /** The interval after each symbol (entry 0 is [0, 1)). */
  steps: { low: number; high: number }[]
}

/**
 * The arithmetic-coding interval of a message (symbol indices) for i.i.d. symbols with probabilities p: start from
 * [0, 1) and, for each symbol s, keep the sub-interval [F(s − 1), F(s)) of the current one, with F the cumulative
 * distribution in index order (Rissanen, 1976; Witten, Neal and Cleary, 1987). In double precision, so messages whose
 * probability underflows (about 1e-300) cannot be represented: the width then reaches 0.
 */
export function arithmeticInterval(message: ArrayLike<number>, probabilities: Probabilities): ArithmeticInterval {
  const p = flatProbabilities(probabilities, 'arithmeticInterval')
  const cumulative = [0]
  for (const v of p) cumulative.push(cumulative[cumulative.length - 1] + v)
  let low = 0
  let high = 1
  const steps = [{ low, high }]
  for (let i = 0; i < message.length; i++) {
    const s = message[i]
    if (!(Number.isInteger(s) && s >= 0 && s < p.length)) throw new RangeError(`arithmeticInterval: bad symbol ${s}`)
    const width = high - low
    high = low + width * cumulative[s + 1]
    low = low + width * cumulative[s]
    steps.push({ low, high })
  }
  const width = high - low
  return { low, high, width, bits: Math.ceil(-Math.log2(width)) + 1, steps }
}

// ── Error-correcting codes ───────────────────────────────────────────────────────────────────────────────────────────

/** A word: a string of symbols or an array of numbers. */
export type Word = string | ArrayLike<number>

/** The number of positions where two words of equal length differ (Hamming, 1950). */
export function hammingDistance(a: Word, b: Word): number {
  if (a.length !== b.length) throw new RangeError('hammingDistance: words of different lengths')
  let d = 0
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++
  return d
}

/** The number of non-zero positions of a word (characters other than '0' in a string). */
export function hammingWeight(a: Word): number {
  let w = 0
  for (let i = 0; i < a.length; i++) if (typeof a === 'string' ? a[i] !== '0' : a[i] !== 0) w++
  return w
}

/** The minimum Hamming distance between distinct codewords of a code (∞ for fewer than two codewords). */
export function minimumDistance(code: readonly Word[]): number {
  let best = Infinity
  for (let i = 0; i < code.length; i++)
    for (let j = i + 1; j < code.length; j++) best = Math.min(best, hammingDistance(code[i], code[j]))
  return best
}

/** n choose k as a float (exact while below 2⁵³). */
function choose(n: number, k: number): number {
  let c = 1
  for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i
  return Math.round(c)
}

function checkCode(n: number, d: number, q: number, where: string): void {
  if (!(Number.isInteger(n) && n >= 1)) throw new RangeError(`${where}: n must be a positive integer`)
  if (!(Number.isInteger(d) && d >= 1 && d <= n)) throw new RangeError(`${where}: need 1 ≤ d ≤ n`)
  if (!(Number.isInteger(q) && q >= 2)) throw new RangeError(`${where}: q must be an integer ≥ 2`)
}

/**
 * The Hamming (sphere-packing) bound on the number of codewords A_q(n, d) of a q-ary code of length n and minimum
 * distance d: ⌊qⁿ / Σ_{i ≤ t} C(n, i)(q − 1)ⁱ⌋ with t = ⌊(d − 1)/2⌋ (Hamming, 1950). Perfect codes attain it.
 */
export function hammingBound(n: number, d: number, q = 2): number {
  checkCode(n, d, q, 'hammingBound')
  const t = Math.floor((d - 1) / 2)
  let volume = 0
  for (let i = 0; i <= t; i++) volume += choose(n, i) * (q - 1) ** i
  return Math.floor(q ** n / volume)
}

/** The Singleton bound A_q(n, d) ≤ q^{n − d + 1} (Singleton, 1964). MDS codes (e.g. Reed–Solomon) attain it. */
export function singletonBound(n: number, d: number, q = 2): number {
  checkCode(n, d, q, 'singletonBound')
  return q ** (n - d + 1)
}

/**
 * The Plotkin bound (Plotkin, 1960). For binary codes, with d even: A ≤ 2⌊d/(2d − n)⌋ when 2d > n and A ≤ 4d when
 * n = 2d; with d odd: A ≤ 2⌊(d + 1)/(2d + 1 − n)⌋ when 2d + 1 > n and A ≤ 4d + 4 when n = 2d + 1 (MacWilliams and
 * Sloane, 1977, ch. 2 Theorem 8). For q-ary codes with θ = 1 − 1/q and d > θn: A ≤ ⌊d/(d − θn)⌋. Returns ∞ when the
 * bound does not apply (the minimum distance is too small relative to n).
 */
export function plotkinBound(n: number, d: number, q = 2): number {
  checkCode(n, d, q, 'plotkinBound')
  if (q === 2) {
    if (d % 2 === 0) {
      if (2 * d > n) return 2 * Math.floor(d / (2 * d - n))
      if (n === 2 * d) return 4 * d
    } else {
      if (2 * d + 1 > n) return 2 * Math.floor((d + 1) / (2 * d + 1 - n))
      if (n === 2 * d + 1) return 4 * d + 4
    }
    return Infinity
  }
  const theta = 1 - 1 / q
  return d > theta * n ? Math.floor(d / (d - theta * n)) : Infinity
}
