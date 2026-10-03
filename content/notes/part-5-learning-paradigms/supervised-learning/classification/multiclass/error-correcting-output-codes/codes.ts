/**
 * Output codes for K classes and the probability that Hamming decoding recovers the true class when every binary
 * classifier is wrong independently with probability p. A code is K rows of L entries in {−1, 0, +1}.
 */
import { rng } from '@/lib/math'

export type Code = number[][]
export type CodeName = 'ovr' | 'ovo' | 'exhaustive' | 'random'

/** One-versus-rest: column k is +1 for class k and −1 for every other class. */
export function oneVersusRest(k: number): Code {
  return Array.from({ length: k }, (_, r) => Array.from({ length: k }, (_, s) => (r === s ? 1 : -1)))
}

/** One-versus-one: column (i, j) is +1 for class i, −1 for class j and 0 (not trained on) for the rest. */
export function oneVersusOne(k: number): Code {
  const rows: Code = Array.from({ length: k }, () => [])
  for (let i = 0; i < k; i++)
    for (let j = i + 1; j < k; j++) for (let r = 0; r < k; r++) rows[r].push(r === i ? 1 : r === j ? -1 : 0)
  return rows
}

/**
 * Dietterich and Bakiri's exhaustive code: every split of the classes into two non-empty groups once, 2^(K−1) − 1
 * columns. Row 1 is all +1; row i alternates runs of 2^(K−i) entries −1 and +1, dropping the constant last column.
 */
export function exhaustive(k: number): Code {
  const length = 2 ** (k - 1) - 1
  return Array.from({ length: k }, (_, r) =>
    Array.from({ length }, (_, c) => (r === 0 ? 1 : Math.floor(c / 2 ** (k - 1 - r)) % 2 === 1 ? 1 : -1)),
  )
}

/** Entries ±1 with probability 1/2 each; a constant column (no binary problem to learn) is redrawn. */
export function randomCode(k: number, length: number, seed: number): Code {
  const g = rng(seed)
  const rows: Code = Array.from({ length: k }, () => [])
  for (let s = 0; s < length; s++) {
    let column: number[]
    do column = Array.from({ length: k }, () => (g.uniform() < 0.5 ? 1 : -1))
    while (column.every((v) => v === column[0]))
    column.forEach((v, r) => rows[r].push(v))
  }
  return rows
}

/**
 * Minimum distance between rows, counting only columns where both rows are non-zero. For a ternary code the others do
 * not count: a classifier's output on a class it was never trained on is arbitrary.
 */
export function minimumDistance(code: Code): number {
  let best = Infinity
  for (let a = 0; a < code.length; a++)
    for (let b = a + 1; b < code.length; b++) {
      let d = 0
      code[a].forEach((v, s) => {
        if (v !== 0 && code[b][s] !== 0 && v !== code[b][s]) d++
      })
      best = Math.min(best, d)
    }
  return best
}

/**
 * Hamming decoding with zeros counted as 1/2 (Allwein, Schapire and Singer): the share of the true class among the
 * nearest rows, so a tie is broken uniformly at random.
 */
function decodeShare(code: Code, bits: ArrayLike<number>, truth: number): number {
  let best = Infinity
  let ties = 0
  let truthNearest = false
  for (let r = 0; r < code.length; r++) {
    const row = code[r]
    let d = 0
    for (let s = 0; s < row.length; s++) d += row[s] === 0 ? 0.5 : row[s] === bits[s] ? 0 : 1
    if (d < best) {
      best = d
      ties = 1
      truthNearest = r === truth
    } else if (d === best) {
      ties++
      if (r === truth) truthNearest = true
    }
  }
  return truthNearest ? 1 / ties : 0
}

/** Exhaustive enumeration up to this many columns (2^L received words per class); Monte Carlo above it. */
const EXACT_MAX = 14
const DRAWS = 300

export type Evaluator = { exact: boolean; at: (p: number) => number }

/**
 * P(decoded class = true class), averaged over classes, when each bit on a non-zero entry of the true row is flipped
 * independently with probability p and each bit on a zero entry is a fair coin.
 */
export function successProbability(code: Code, seed: number): Evaluator {
  const k = code.length
  const length = code[0].length
  if (length <= EXACT_MAX) {
    // Group received words by the number w of flipped non-zero bits: P = Σ_w share[w] p^w (1 − p)^(n − w) / 2^zeros.
    const tables = code.map((row, y) => {
      const nonzero = row.filter((v) => v !== 0).length
      const share = new Float64Array(nonzero + 1)
      const bits = new Int8Array(length)
      for (let word = 0; word < 2 ** length; word++) {
        let flips = 0
        for (let s = 0; s < length; s++) {
          bits[s] = (word >> s) & 1 ? 1 : -1
          if (row[s] !== 0 && bits[s] !== row[s]) flips++
        }
        share[flips] += decodeShare(code, bits, y)
      }
      return { share, nonzero, coin: 2 ** -(length - nonzero) }
    })
    return {
      exact: true,
      at: (p) => {
        let total = 0
        for (const { share, nonzero, coin } of tables)
          share.forEach((c, w) => (total += c * coin * p ** w * (1 - p) ** (nonzero - w)))
        return total / k
      },
    }
  }
  // Common random numbers: the same uniforms serve every p, so the curve is smooth and never decreases by chance.
  const g = rng(seed)
  const flipDraws = Array.from({ length: k * DRAWS }, () => Float64Array.from({ length }, () => g.uniform()))
  const coinDraws = Array.from({ length: k * DRAWS }, () => Float64Array.from({ length }, () => g.uniform()))
  const bits = new Int8Array(length)
  return {
    exact: false,
    at: (p) => {
      let total = 0
      for (let y = 0; y < k; y++) {
        const row = code[y]
        for (let n = 0; n < DRAWS; n++) {
          const u = flipDraws[y * DRAWS + n]
          const c = coinDraws[y * DRAWS + n]
          for (let s = 0; s < length; s++) bits[s] = row[s] === 0 ? (c[s] < 0.5 ? 1 : -1) : u[s] < p ? -row[s] : row[s]
          total += decodeShare(code, bits, y)
        }
      }
      return total / (k * DRAWS)
    },
  }
}
