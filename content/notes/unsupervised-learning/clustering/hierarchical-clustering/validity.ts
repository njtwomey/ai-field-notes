/**
 * Checks that separate cluster structure from noise: merge-height jumps, cophenetic correlation, the Hopkins statistic,
 * the gap statistic and subsample stability. Randomness comes in as a `uniform` function so the caller seeds it.
 */
import { agglomerate, cophenetic, cutK, type Linkage, type Merge } from './agglomerate'

export type Rows = number[][]
type Uniform = () => number

/** Columns rescaled to mean 0 and standard deviation 1 (divisor n); constant columns are only centred. */
export function standardise(x: Rows): Rows {
  const n = x.length
  const d = x[0].length
  const mean = Array.from({ length: d }, (_, c) => x.reduce((s, r) => s + r[c], 0) / n)
  const sd = mean.map((m, c) => Math.sqrt(x.reduce((s, r) => s + (r[c] - m) ** 2, 0) / n) || 1)
  return x.map((r) => r.map((v, c) => (v - mean[c]) / sd[c]))
}

/** Per-column minimum and maximum. */
export function boundingBox(x: Rows): { lo: number[]; hi: number[] } {
  const d = x[0].length
  const lo = Array.from({ length: d }, (_, c) => Math.min(...x.map((r) => r[c])))
  const hi = Array.from({ length: d }, (_, c) => Math.max(...x.map((r) => r[c])))
  return { lo, hi }
}

export function uniformInBox(n: number, box: { lo: number[]; hi: number[] }, u: Uniform): Rows {
  return Array.from({ length: n }, () => box.lo.map((l, c) => l + (box.hi[c] - l) * u()))
}

/** Ratio of the first merge height above a k-cluster cut to the last one below it. NaN when k is 1 or n. */
export function jumpRatio(merges: Merge[], k: number): number {
  const n = merges.length + 1
  if (k < 2 || k >= n) return NaN
  const below = merges[n - k - 1].height
  const above = merges[n - k].height
  return below > 0 ? above / below : Infinity
}

function dist(a: readonly number[], b: readonly number[]): number {
  let s = 0
  for (let c = 0; c < a.length; c++) s += (a[c] - b[c]) ** 2
  return Math.sqrt(s)
}

/** Pearson correlation between original and cophenetic distances over all pairs. */
export function copheneticCorrelation(x: Rows, merges: Merge[]): number {
  const n = x.length
  const C = cophenetic(n, merges)
  let sa = 0
  let sb = 0
  let saa = 0
  let sbb = 0
  let sab = 0
  let m = 0
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const a = dist(x[i], x[j])
      const b = C[i * n + j]
      sa += a
      sb += b
      saa += a * a
      sbb += b * b
      sab += a * b
      m++
    }
  const cov = sab / m - (sa / m) * (sb / m)
  return cov / Math.sqrt((saa / m - (sa / m) ** 2) * (sbb / m - (sb / m) ** 2))
}

/**
 * Hopkins statistic H = Σ uᵢᵈ / (Σ uᵢᵈ + Σ wᵢᵈ), averaged over `reps` draws. uᵢ is the distance from a uniform point in
 * the bounding box to the nearest data point, wᵢ the distance from a sampled data point to its nearest other data
 * point, with m of each. Under uniform data H follows Beta(m, m), centred on 0.5; clustered data push it towards 1.
 */
export function hopkins(x: Rows, m: number, reps: number, u: Uniform): number {
  const n = x.length
  const d = x[0].length
  const box = boundingBox(x)
  const nearest = (p: readonly number[], skip: number) => {
    let best = Infinity
    for (let i = 0; i < n; i++) if (i !== skip) best = Math.min(best, dist(p, x[i]))
    return best
  }
  let total = 0
  for (let r = 0; r < reps; r++) {
    const picked = new Set<number>()
    while (picked.size < Math.min(m, n)) picked.add(Math.floor(u() * n))
    let sw = 0
    for (const i of picked) sw += nearest(x[i], i) ** d
    let su = 0
    for (const p of uniformInBox(picked.size, box, u)) su += nearest(p, -1) ** d
    total += su / (su + sw)
  }
  return total / reps
}

/** Within-cluster sum of squares of a labelling. */
export function withinSS(x: Rows, labels: number[]): number {
  const d = x[0].length
  const sums = new Map<number, { n: number; s: number[]; ss: number }>()
  x.forEach((r, i) => {
    const g = sums.get(labels[i]) ?? { n: 0, s: new Array<number>(d).fill(0), ss: 0 }
    g.n++
    r.forEach((v, c) => {
      g.s[c] += v
      g.ss += v * v
    })
    sums.set(labels[i], g)
  })
  let w = 0
  for (const g of sums.values()) w += g.ss - g.s.reduce((a, v) => a + v * v, 0) / g.n
  return Math.max(w, 0)
}

/** Trees for B reference data sets drawn uniformly from the bounding box of x: the null of the gap statistic. */
export function referenceTrees(x: Rows, linkage: Linkage, B: number, u: Uniform): { x: Rows; merges: Merge[] }[] {
  const box = boundingBox(x)
  return Array.from({ length: B }, () => {
    const ref = uniformInBox(x.length, box, u)
    return { x: ref, merges: agglomerate(ref, linkage) }
  })
}

/** Gap(k) = mean over references of log W*ₖ − log Wₖ, with s = sd · √(1 + 1/B). */
export function gap(x: Rows, labels: number[], k: number, refs: { x: Rows; merges: Merge[] }[]) {
  const logs = refs.map((r) => Math.log(withinSS(r.x, cutK(r.x.length, r.merges, k))))
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length
  const sd = Math.sqrt(logs.reduce((a, b) => a + (b - mean) ** 2, 0) / logs.length)
  return { gap: mean - Math.log(withinSS(x, labels)), s: sd * Math.sqrt(1 + 1 / logs.length) }
}

/** Trees for B subsamples of x without replacement, each a fraction `frac` of the points. */
export function subsampleTrees(
  x: Rows,
  linkage: Linkage,
  B: number,
  frac: number,
  u: Uniform,
): { idx: number[]; merges: Merge[] }[] {
  const n = x.length
  const m = Math.max(2, Math.round(frac * n))
  return Array.from({ length: B }, () => {
    // Partial Fisher–Yates shuffle: the first m entries are a uniform subsample.
    const order = Array.from({ length: n }, (_, i) => i)
    for (let i = 0; i < m; i++) {
      const j = i + Math.floor(u() * (n - i))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    const idx = order.slice(0, m).sort((a, b) => a - b)
    return {
      idx,
      merges: agglomerate(
        idx.map((i) => x[i]),
        linkage,
      ),
    }
  })
}

/**
 * Cluster-wise stability (Hennig 2007): for each cluster of the full-data cut, restricted to the points in a
 * subsample, the best Jaccard similarity with a cluster of the subsample's own k-cluster cut. Returns the mean over
 * clusters and subsamples, in [0, 1].
 */
export function stability(labels: number[], k: number, subs: { idx: number[]; merges: Merge[] }[]): number {
  let total = 0
  let count = 0
  for (const { idx, merges } of subs) {
    const sub = cutK(idx.length, merges, Math.min(k, idx.length))
    const orig = idx.map((i) => labels[i])
    const origSize = new Map<number, number>()
    const subSize = new Map<number, number>()
    const both = new Map<string, number>()
    orig.forEach((a, t) => {
      const b = sub[t]
      origSize.set(a, (origSize.get(a) ?? 0) + 1)
      subSize.set(b, (subSize.get(b) ?? 0) + 1)
      both.set(`${a},${b}`, (both.get(`${a},${b}`) ?? 0) + 1)
    })
    for (const [a, na] of origSize) {
      let best = 0
      for (const [b, nb] of subSize) {
        const inter = both.get(`${a},${b}`) ?? 0
        if (inter) best = Math.max(best, inter / (na + nb - inter))
      }
      total += best
      count++
    }
  }
  return count ? total / count : NaN
}
