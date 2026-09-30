/**
 * Keyed, counter-based random streams.
 *
 * A stream is identified by a path of names, e.g. `7/chain:3/env` for `stream(7).child('chain', 3).child('env')`. The
 * path is hashed to 128 bits (see `absorb` in philox.ts); the hash supplies the Philox key and the upper half of the
 * counter, and the lower 64 bits of the counter count the stream's output blocks. So a stream's n-th value depends only
 * on its path and n: drawing from a parent or a sibling never changes a child's values, and every path gets its own
 * sequence.
 */

import { absorb, philox4x32, ROOT_HASH, type KeyHash } from './philox'

/** A keyed random stream (the contract in the aifn README, plus `int`). */
export interface Stream {
  /** The human-readable path, e.g. `"7/chain:3/env"`. Separators inside names are percent-encoded. */
  readonly key: string
  /**
   * An independent substream named by `path` (one level; several parts are joined by `:` in the key). It never collides
   * with its siblings, its parent or any other path, and its values do not depend on how much the parent has drawn.
   * Numbers name the same child as their decimal strings: `child(3)` equals `child('3')`.
   */
  child(...path: (string | number)[]): Stream
  /** A uniform double in [0, 1) with 53 random bits (uses two 32-bit words). */
  uniform(): number
  /** A uniform 32-bit unsigned integer. */
  uint32(): number
  /** A uniform integer in {0, …, n − 1} for integer 1 ≤ n ≤ 2⁵³, without modulo bias (rejection sampling). */
  int(n: number): number
}

const LEVEL = 0x2f2f2f2f // marks the start of a path level
const ROOT = 0x5eed5eed // marks the root

/** Encode one level of a path injectively as 32-bit words: marker, part count, then each part's length and code units. */
function encodeLevel(marker: number, parts: string[]): number[] {
  const words = [marker, parts.length]
  for (const p of parts) {
    words.push(p.length)
    for (let i = 0; i < p.length; i++) words.push(p.charCodeAt(i))
  }
  return words
}

/** The display form of one path part: `%`, `/` and `:` are percent-encoded so that keys map one-to-one to paths. */
function escapePart(part: string): string {
  return part.replace(/[%/:]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`)
}

function partName(part: string | number): string {
  if (typeof part === 'number' && !Number.isFinite(part)) throw new RangeError(`stream path part ${part} is not finite`)
  return String(part)
}

const TWO_POW_32 = 4294967296
const TWO_POW_26 = 67108864
const TWO_POW_53 = 9007199254740992

class KeyedStream implements Stream {
  readonly key: string
  private readonly hash: KeyHash
  /** Number of 32-bit words drawn so far. */
  private position = 0
  private block = -1
  private readonly buffer = new Uint32Array(4)

  constructor(key: string, hash: KeyHash) {
    this.key = key
    this.hash = hash
  }

  child(...path: (string | number)[]): Stream {
    const parts = path.map(partName)
    const hash = absorb(this.hash, encodeLevel(LEVEL, parts))
    return new KeyedStream(`${this.key}/${parts.map(escapePart).join(':')}`, hash)
  }

  uint32(): number {
    const p = this.position++
    const b = Math.floor(p / 4)
    if (b !== this.block) {
      // Counter = (block low 32 bits, block high bits, h2, h3); key = (h0, h1).
      philox4x32(
        b >>> 0,
        Math.floor(b / TWO_POW_32) >>> 0,
        this.hash[2],
        this.hash[3],
        this.hash[0],
        this.hash[1],
        this.buffer,
      )
      this.block = b
    }
    return this.buffer[p & 3]
  }

  uniform(): number {
    // 27 + 26 bits, as in the reference MT19937 genrand_res53.
    const a = this.uint32() >>> 5
    const b = this.uint32() >>> 6
    return (a * TWO_POW_26 + b) / TWO_POW_53
  }

  int(n: number): number {
    if (!(Number.isInteger(n) && n >= 1 && n <= TWO_POW_53))
      throw new RangeError(`int(n) needs an integer 1 ≤ n ≤ 2^53`)
    if (n <= TWO_POW_32) {
      // Reject the top partial copy of {0, …, n − 1} so every residue is equally likely.
      const limit = TWO_POW_32 - (TWO_POW_32 % n)
      for (;;) {
        const u = this.uint32()
        if (u < limit) return u % n
      }
    }
    const limit = TWO_POW_53 - (TWO_POW_53 % n)
    for (;;) {
      const u = this.uniform() * TWO_POW_53
      if (u < limit) return u % n
    }
  }
}

/**
 * The root stream for a seed. Numbers and strings name the same root when their decimal forms agree:
 * `stream(7)` equals `stream('7')`.
 */
export function stream(seed: number | string): Stream {
  const name = partName(seed)
  return new KeyedStream(escapePart(name), absorb(ROOT_HASH, encodeLevel(ROOT, [name])))
}
