/**
 * Metrics for comparing two ranked lists of the same query, as in the "which ranking is better?" exercise. Built on
 * the single-list metrics in ranking.ts. `grades[i]` is the relevance grade of the document at rank i + 1 (0 = not
 * relevant) and `total` is R, the number of relevant documents in the collection.
 */
import { averagePrecision, isRelevant, precisionAt, recallAt, reciprocalRank } from './ranking'

export type MetricId = 'first' | 'p' | 'r' | 'rr' | 'ap' | 'apk' | 'ndcg' | 'rprec' | 'err'

export type MetricSpec = {
  id: MetricId
  /** Short label for chips and tables; pass a number for the current cut-off or 'k' for the generic name. */
  label: (k: number | 'k') => string
  /** True when a lower value is better (only the rank of the first relevant document). */
  lowerIsBetter?: boolean
  /** True when the metric has a cut-off, so it can be drawn as a curve over k. */
  usesK: boolean
  /** True when the metric uses relevance grades rather than relevant / not relevant. */
  graded?: boolean
  value: (grades: number[], k: number, total: number, maxGrade: number) => number
}

/** Rank of the first relevant document, or Infinity if the list has none. */
export function firstRelevantRank(grades: number[]): number {
  const i = grades.findIndex(isRelevant)
  return i < 0 ? Infinity : i + 1
}

/**
 * Average precision truncated at k and divided by min(R, k), the usual recommender convention: a ranking whose top k
 * is as good as it can be scores 1 even when R > k.
 */
export function averagePrecisionAt(grades: number[], k: number, total: number): number {
  const denominator = Math.min(total, k)
  return denominator ? averagePrecision(grades.slice(0, k), 1) / denominator : 0
}

const gainDcg = (grades: number[], k: number) => grades.slice(0, k).reduce((s, g, i) => s + g / Math.log2(i + 2), 0)

/**
 * NDCG at k with linear gains. The ideal ranking holds this list's relevant documents sorted by grade, plus the
 * relevant documents it did not retrieve (R minus the number retrieved), which are counted as grade 1.
 */
export function ndcgAt(grades: number[], k: number, total: number): number {
  const found = grades.filter(isRelevant).sort((a, b) => b - a)
  const ideal = [...found, ...new Array<number>(Math.max(total - found.length, 0)).fill(1)]
  const best = gainDcg(ideal, k)
  return best ? gainDcg(grades, k) / best : 0
}

/**
 * Expected reciprocal rank at k (Chapelle et al., 2009). Grade g satisfies the reader with probability
 * (2^g - 1) / 2^maxGrade; the reader scans down and stops when satisfied.
 */
export function errAt(grades: number[], k: number, maxGrade: number): number {
  let reach = 1
  let err = 0
  grades.slice(0, k).forEach((g, i) => {
    const stop = (2 ** g - 1) / 2 ** maxGrade
    err += (reach * stop) / (i + 1)
    reach *= 1 - stop
  })
  return err
}

export const METRICS: MetricSpec[] = [
  {
    id: 'first',
    label: () => 'first relevant rank',
    lowerIsBetter: true,
    usesK: false,
    value: (g) => firstRelevantRank(g),
  },
  { id: 'p', label: (k) => `P@${k}`, usesK: true, value: (g, k) => precisionAt(g, k) },
  { id: 'r', label: (k) => `R@${k}`, usesK: true, value: (g, k, R) => recallAt(g, k, R) },
  { id: 'rr', label: () => 'reciprocal rank', usesK: false, value: (g) => reciprocalRank(g) },
  { id: 'ap', label: () => 'AP', usesK: false, value: (g, _k, R) => averagePrecision(g, R) },
  { id: 'apk', label: (k) => `AP@${k}`, usesK: true, value: (g, k, R) => averagePrecisionAt(g, k, R) },
  { id: 'ndcg', label: (k) => `nDCG@${k}`, usesK: true, graded: true, value: (g, k, R) => ndcgAt(g, k, R) },
  { id: 'rprec', label: () => 'R-precision', usesK: false, value: (g, _k, R) => precisionAt(g, R) },
  { id: 'err', label: (k) => `ERR@${k}`, usesK: true, graded: true, value: (g, k, _R, m) => errAt(g, k, m) },
]

export type Verdict = 'A' | 'B' | 'tie'

/** Which list the metric prefers. Values within 1e-9 tie. */
export function verdict(spec: MetricSpec, a: number, b: number): Verdict {
  if (a === b || Math.abs(a - b) < 1e-9) return 'tie'
  const aBetter = spec.lowerIsBetter ? a < b : a > b
  return aBetter ? 'A' : 'B'
}

/** A ranking of n documents with the given grades at the given 1-based ranks. */
export function listWith(n: number, relevant: Record<number, number> | number[]): number[] {
  const grades = new Array<number>(n).fill(0)
  if (Array.isArray(relevant)) relevant.forEach((r) => (grades[r - 1] = 1))
  else for (const [r, g] of Object.entries(relevant)) grades[Number(r) - 1] = g
  return grades
}
