import { normalQuantile } from 'aifn-compute/numerics/special'
/**
 * Small time-series classification helpers shared by the shapelet, DTW and SAX notes: z-normalisation, subsequence
 * distance, information gain of a split, DTW with a Sakoe–Chiba window and its warping path, the LB_Keogh envelope,
 * PAA and SAX. Figure-sized inputs only.
 */

export function znorm(x: number[]): number[] {
  const n = x.length
  const mu = x.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(x.reduce((a, b) => a + (b - mu) ** 2, 0) / n)
  return x.map((v) => (sd > 1e-12 ? (v - mu) / sd : 0))
}

/**
 * Subsequence distance: the smallest length-normalised Euclidean distance between the z-normalised shapelet and any
 * z-normalised window of the series of the same length. Returns the distance and the best position.
 */
export function subsequenceDistance(series: number[], shapelet: number[]): { distance: number; position: number } {
  const m = shapelet.length
  const s = znorm(shapelet)
  let best = Infinity
  let position = 0
  for (let i = 0; i + m <= series.length; i++) {
    const w = znorm(series.slice(i, i + m))
    let d = 0
    for (let k = 0; k < m; k++) d += (w[k] - s[k]) ** 2
    if (d < best) {
      best = d
      position = i
    }
  }
  return { distance: Math.sqrt(best / m), position }
}

const entropyOf = (counts: number[]) => {
  const n = counts.reduce((a, b) => a + b, 0)
  return n === 0 ? 0 : -counts.reduce((h, c) => (c > 0 ? h + (c / n) * Math.log2(c / n) : h), 0)
}

/** Information gain in bits of splitting labelled points at `threshold` (distance < threshold goes left). */
export function informationGain(distances: number[], labels: number[], threshold: number, classes = 2): number {
  const count = (pick: (i: number) => boolean) => {
    const c = new Array<number>(classes).fill(0)
    labels.forEach((y, i) => pick(i) && c[y]++)
    return c
  }
  const all = count(() => true)
  const left = count((i) => distances[i] < threshold)
  const right = count((i) => distances[i] >= threshold)
  const n = labels.length
  const nl = left.reduce((a, b) => a + b, 0)
  return entropyOf(all) - (nl / n) * entropyOf(left) - ((n - nl) / n) * entropyOf(right)
}

/** Best threshold (midpoint between sorted distances) by information gain. */
export function bestSplit(distances: number[], labels: number[]): { threshold: number; gain: number } {
  const sorted = [...new Set(distances)].sort((a, b) => a - b)
  let best = { threshold: sorted[0] ?? 0, gain: -1 }
  for (let i = 0; i + 1 < sorted.length; i++) {
    const t = (sorted[i] + sorted[i + 1]) / 2
    const g = informationGain(distances, labels, t)
    if (g > best.gain) best = { threshold: t, gain: g }
  }
  return best
}

/**
 * Dynamic time warping with squared pointwise cost and a Sakoe–Chiba window of half-width `w` (Infinity = none).
 * Returns the accumulated cost matrix (Infinity outside the window), the DTW distance √D(n, m) and the optimal path.
 */
export function dtw(a: number[], b: number[], w = Infinity) {
  const n = a.length
  const m = b.length
  const D = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(Infinity))
  D[0][0] = 0
  for (let i = 1; i <= n; i++) {
    const lo = Math.max(1, Math.ceil(i - w))
    const hi = Math.min(m, Math.floor(i + w))
    for (let j = lo; j <= hi; j++) {
      const cost = (a[i - 1] - b[j - 1]) ** 2
      D[i][j] = cost + Math.min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    }
  }
  const path: [number, number][] = []
  let i = n
  let j = m
  if (Number.isFinite(D[n][m])) {
    while (i > 0 && j > 0) {
      path.push([i - 1, j - 1])
      const options: [number, number, number][] = [
        [D[i - 1][j - 1], i - 1, j - 1],
        [D[i - 1][j], i - 1, j],
        [D[i][j - 1], i, j - 1],
      ]
      options.sort((p, q) => p[0] - q[0])
      ;[, i, j] = options[0]
    }
    path.reverse()
  }
  return { cost: D, distance: Math.sqrt(D[n][m]), path }
}

/** LB_Keogh: upper and lower envelopes of `b` over a window of half-width r, and the lower bound on DTW(a, b). */
export function lbKeogh(a: number[], b: number[], r: number) {
  const n = b.length
  const upper = new Array<number>(n)
  const lower = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let u = -Infinity
    let l = Infinity
    for (let k = Math.max(0, i - r); k <= Math.min(n - 1, i + r); k++) {
      u = Math.max(u, b[k])
      l = Math.min(l, b[k])
    }
    upper[i] = u
    lower[i] = l
  }
  let s = 0
  for (let i = 0; i < a.length; i++) {
    if (a[i] > upper[i]) s += (a[i] - upper[i]) ** 2
    else if (a[i] < lower[i]) s += (a[i] - lower[i]) ** 2
  }
  return { upper, lower, bound: Math.sqrt(s) }
}

/** Lengths of the w segments that PAA splits n points into (equal when w divides n). */
export const segmentLengths = (n: number, w: number) =>
  Array.from({ length: w }, (_, i) => Math.round(((i + 1) * n) / w) - Math.round((i * n) / w))

/** Piecewise aggregate approximation: the mean of each of `w` near-equal segments. */
export function paa(x: number[], w: number): number[] {
  const n = x.length
  return Array.from({ length: w }, (_, i) => {
    const start = Math.round((i * n) / w)
    const end = Math.round(((i + 1) * n) / w)
    let s = 0
    for (let k = start; k < end; k++) s += x[k]
    return s / (end - start)
  })
}

/**
 * Distance between two PAA representations, weighting each segment by its length: √(Σ ℓᵢ (pᵢ − qᵢ)²). With equal
 * segments this is √(n/w) ‖p − q‖. It never exceeds the Euclidean distance between the original series.
 */
export function paaDistance(p: number[], q: number[], n: number): number {
  const lengths = segmentLengths(n, p.length)
  return Math.sqrt(p.reduce((s, v, i) => s + lengths[i] * (v - q[i]) ** 2, 0))
}

/** SAX breakpoints: the a−1 standard-normal quantiles that split the line into a equiprobable regions. */
export const saxBreakpoints = (alphabet: number) =>
  Array.from({ length: alphabet - 1 }, (_, i) => normalQuantile((i + 1) / alphabet))

/** SAX word (as symbol indices 0..a−1) of a z-normalised series. */
export function sax(x: number[], w: number, alphabet: number): { paa: number[]; symbols: number[] } {
  const p = paa(znorm(x), w)
  const cuts = saxBreakpoints(alphabet)
  return { paa: p, symbols: p.map((v) => cuts.filter((c) => v > c).length) }
}

/**
 * MINDIST between two SAX words for original series length n: √(Σ ℓᵢ dist(sᵢ, tᵢ)²), with ℓᵢ the segment lengths.
 * The symbol distance is 0 for equal or adjacent symbols and otherwise the gap between the breakpoints that separate
 * them. With equal segments this is the published √(n/w) √(Σ dist²).
 */
export function minDist(s: number[], t: number[], n: number, alphabet: number): number {
  const cuts = saxBreakpoints(alphabet)
  const lengths = segmentLengths(n, s.length)
  let sum = 0
  for (let i = 0; i < s.length; i++) {
    const lo = Math.min(s[i], t[i])
    const hi = Math.max(s[i], t[i])
    if (hi - lo > 1) sum += lengths[i] * (cuts[hi - 1] - cuts[lo]) ** 2
  }
  return Math.sqrt(sum)
}

export const letters = (symbols: number[]) => symbols.map((k) => String.fromCharCode(97 + k)).join('')
