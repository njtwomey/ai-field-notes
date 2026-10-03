/**
 * Matrix profile algorithms for the notes in time-series/similarity-search, sized for in-browser figures (n up to a
 * few thousand). Distances are z-normalised Euclidean: d² = 2m(1 − ρ), with ρ the Pearson correlation of the two
 * subsequences. Indices are 0-based.
 */
import { fft, ifft, nextPowerOfTwo } from '@/lib/dsp'

export type Stats = { mean: Float64Array; sd: Float64Array }

/** Means and standard deviations of every length-m subsequence, from cumulative sums in O(n). */
export function movingStats(x: ArrayLike<number>, m: number): Stats {
  const n = x.length - m + 1
  const mean = new Float64Array(n)
  const sd = new Float64Array(n)
  let s = 0
  let s2 = 0
  for (let k = 0; k < m; k++) {
    s += x[k]
    s2 += x[k] * x[k]
  }
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      s += x[i + m - 1] - x[i - 1]
      s2 += x[i + m - 1] ** 2 - x[i - 1] ** 2
    }
    const mu = s / m
    mean[i] = mu
    sd[i] = Math.sqrt(Math.max(s2 / m - mu * mu, 1e-12))
  }
  return { mean, sd }
}

/** z-normalised distance from a dot product and the two subsequences' statistics. */
export const zDistance = (qt: number, m: number, mi: number, si: number, mj: number, sj: number) => {
  const rho = (qt - m * mi * mj) / (m * si * sj)
  return Math.sqrt(Math.max(2 * m * (1 - rho), 0))
}

/**
 * Sliding dot products of query q with every length-|q| subsequence of x, by FFT in O(n log n): the convolution of x
 * with the reversed query, read off from index m − 1.
 */
export function slidingDotProduct(q: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const m = q.length
  const n = x.length
  const size = nextPowerOfTwo(n + m)
  const reversed = Array.from({ length: m }, (_, k) => q[m - 1 - k])
  const X = fft(x, size)
  const Q = fft(reversed, size)
  const re = new Float64Array(size)
  const im = new Float64Array(size)
  for (let k = 0; k < size; k++) {
    re[k] = X.re[k] * Q.re[k] - X.im[k] * Q.im[k]
    im[k] = X.re[k] * Q.im[k] + X.im[k] * Q.re[k]
  }
  const product = ifft(re, im).re
  return product.slice(m - 1, n)
}

/** MASS: the distance profile of query q against every subsequence of x. */
export function mass(q: ArrayLike<number>, x: ArrayLike<number>, stats = movingStats(x, q.length)): Float64Array {
  const m = q.length
  const qt = slidingDotProduct(q, x)
  const { mean: qm, sd: qs } = movingStats(q, m)
  return qt.map((v, i) => zDistance(v, m, qm[0], qs[0], stats.mean[i], stats.sd[i]))
}

export type Profile = {
  /** P[i]: distance from subsequence i to its nearest non-trivial neighbour anywhere. */
  profile: Float64Array
  index: Int32Array
  /** Left profile: nearest neighbour among earlier subsequences only (j < i − e). */
  left: Float64Array
  leftIndex: Int32Array
  /** Right profile: nearest neighbour among later subsequences only (j > i + e). */
  right: Float64Array
  rightIndex: Int32Array
}

/** Default exclusion zone, ⌈m/4⌉: neighbours closer than this overlap too much to count as matches. */
export const exclusionZone = (m: number) => Math.ceil(m / 4)

/**
 * STOMP-style self-join in O(n²): walk each diagonal j − i = k > e, updating the dot product in O(1) per step with
 * QT(i+1, j+1) = QT(i, j) − x[i] x[j] + x[i+m] x[j+m]. Every pair is visited once, and updates both ends.
 */
export function selfJoin(x: ArrayLike<number>, m: number, e = exclusionZone(m)): Profile {
  const n = x.length - m + 1
  const { mean, sd } = movingStats(x, m)
  const inf = () => new Float64Array(n).fill(Infinity)
  const neg = () => new Int32Array(n).fill(-1)
  const out: Profile = {
    profile: inf(),
    index: neg(),
    left: inf(),
    leftIndex: neg(),
    right: inf(),
    rightIndex: neg(),
  }
  for (let k = e + 1; k < n; k++) {
    let qt = 0
    for (let t = 0; t < m; t++) qt += x[t] * x[k + t]
    for (let i = 0; i + k < n; i++) {
      const j = i + k
      if (i > 0) qt += -x[i - 1] * x[j - 1] + x[i + m - 1] * x[j + m - 1]
      const d = zDistance(qt, m, mean[i], sd[i], mean[j], sd[j])
      if (d < out.profile[i]) {
        out.profile[i] = d
        out.index[i] = j
      }
      if (d < out.profile[j]) {
        out.profile[j] = d
        out.index[j] = i
      }
      // j is later than i: j is a right neighbour of i, and i a left neighbour of j.
      if (d < out.right[i]) {
        out.right[i] = d
        out.rightIndex[i] = j
      }
      if (d < out.left[j]) {
        out.left[j] = d
        out.leftIndex[j] = i
      }
    }
  }
  return out
}

/**
 * The full z-normalised distance matrix, n × n with the exclusion zone set to Infinity. For figures that replay an
 * anytime algorithm; O(n²) memory, so keep n small.
 */
export function distanceMatrix(x: ArrayLike<number>, m: number, e = exclusionZone(m)): Float64Array[] {
  const n = x.length - m + 1
  const { mean, sd } = movingStats(x, m)
  const D = Array.from({ length: n }, () => new Float64Array(n).fill(Infinity))
  for (let k = e + 1; k < n; k++) {
    let qt = 0
    for (let t = 0; t < m; t++) qt += x[t] * x[k + t]
    for (let i = 0; i + k < n; i++) {
      const j = i + k
      if (i > 0) qt += -x[i - 1] * x[j - 1] + x[i + m - 1] * x[j + m - 1]
      const d = zDistance(qt, m, mean[i], sd[i], mean[j], sd[j])
      D[i][j] = d
      D[j][i] = d
    }
  }
  return D
}

/**
 * STAMP replayed on a precomputed distance matrix: process rows in `order`, and after each row i update P[i] with its
 * row minimum and every P[j] with D[i][j] (the matrix is symmetric, so one row informs every column).
 */
export function anytimeProfile(D: Float64Array[], order: number[], rows: number): Float64Array {
  const n = D.length
  const p = new Float64Array(n).fill(Infinity)
  for (let r = 0; r < Math.min(rows, order.length); r++) {
    const i = order[r]
    const row = D[i]
    for (let j = 0; j < n; j++) {
      const d = row[j]
      if (d < p[i]) p[i] = d
      if (d < p[j]) p[j] = d
    }
  }
  return p
}

/** AB-join: for each subsequence of a, the distance to its nearest subsequence of b (no exclusion zone). */
export function abJoin(a: ArrayLike<number>, b: ArrayLike<number>, m: number) {
  const na = a.length - m + 1
  const nb = b.length - m + 1
  const sa = movingStats(a, m)
  const sb = movingStats(b, m)
  const profile = new Float64Array(na).fill(Infinity)
  const index = new Int32Array(na).fill(-1)
  // Diagonals of the na × nb rectangle, j − i = k for k from −(na − 1) to nb − 1.
  for (let k = -(na - 1); k < nb; k++) {
    const i0 = Math.max(0, -k)
    const j0 = i0 + k
    let qt = 0
    for (let t = 0; t < m; t++) qt += a[i0 + t] * b[j0 + t]
    for (let i = i0, j = j0; i < na && j < nb; i++, j++) {
      if (i > i0) qt += -a[i - 1] * b[j - 1] + a[i + m - 1] * b[j + m - 1]
      const d = zDistance(qt, m, sa.mean[i], sa.sd[i], sb.mean[j], sb.sd[j])
      if (d < profile[i]) {
        profile[i] = d
        index[i] = j
      }
    }
  }
  return { profile, index }
}

/**
 * FLUSS corrected arc curve. Each subsequence i draws an arc to its nearest neighbour index[i]; AC(i) counts arcs
 * passing over position i. Under no structure the expected count is the parabola IAC(i) = 2i(n − i)/n, so
 * CAC(i) = min(AC(i)/IAC(i), 1); values near 0 mark boundaries between regimes. The first and last `edge` positions
 * are set to 1 because arcs cannot cross there.
 */
export function correctedArcCurve(index: ArrayLike<number>, edge: number): Float64Array {
  const n = index.length
  const marks = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) {
    const j = index[i]
    if (j < 0) continue
    const lo = Math.min(i, j)
    const hi = Math.max(i, j)
    marks[lo + 1] += 1
    marks[hi] -= 1
  }
  const cac = new Float64Array(n)
  let running = 0
  for (let i = 0; i < n; i++) {
    running += marks[i]
    const ideal = (2 * i * (n - i)) / n
    cac[i] = i < edge || i >= n - edge || ideal <= 0 ? 1 : Math.min(running / ideal, 1)
  }
  return cac
}

/**
 * The time-series chain anchored at `start`: follow right neighbours while each step is confirmed by the next
 * subsequence's left neighbour pointing back (the bidirectional condition of Zhu et al.).
 */
export function chainFrom(start: number, rightIndex: ArrayLike<number>, leftIndex: ArrayLike<number>): number[] {
  const chain = [start]
  let i = start
  while (rightIndex[i] >= 0 && leftIndex[rightIndex[i]] === i) {
    i = rightIndex[i]
    chain.push(i)
  }
  return chain
}

/** The longest chain over all anchors (the unanchored chain), by following every start. */
export function longestChain(rightIndex: ArrayLike<number>, leftIndex: ArrayLike<number>): number[] {
  let best: number[] = []
  for (let s = 0; s < rightIndex.length; s++) {
    // Only start where no left link points in: otherwise s lies inside a longer chain.
    const l = leftIndex[s]
    if (l >= 0 && rightIndex[l] === s) continue
    const c = chainFrom(s, rightIndex, leftIndex)
    if (c.length > best.length) best = c
  }
  return best
}
