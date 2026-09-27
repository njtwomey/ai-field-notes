/**
 * Ranked-retrieval metrics for one query. `grades[i]` is the relevance grade of the document at rank i + 1 (0 = not
 * relevant). Binary metrics treat any positive grade as relevant. `totalRelevant` is the number of relevant documents
 * in the whole collection, including any the ranking did not retrieve.
 */

export const isRelevant = (g: number) => g > 0

/** Precision at k: the fraction of the top k that is relevant. */
export function precisionAt(grades: number[], k: number): number {
  const top = grades.slice(0, k)
  return top.length ? top.filter(isRelevant).length / k : 0
}

/** Recall at k: the fraction of all relevant documents that appear in the top k. */
export function recallAt(grades: number[], k: number, totalRelevant: number): number {
  return totalRelevant ? grades.slice(0, k).filter(isRelevant).length / totalRelevant : 0
}

/** Average precision: precision at the rank of each relevant document, averaged over all relevant documents. */
export function averagePrecision(grades: number[], totalRelevant: number): number {
  if (!totalRelevant) return 0
  let hits = 0
  let sum = 0
  grades.forEach((g, i) => {
    if (!isRelevant(g)) return
    hits += 1
    sum += hits / (i + 1)
  })
  return sum / totalRelevant
}

/** Reciprocal rank: 1 / (rank of the first relevant document), or 0 if none is retrieved. */
export function reciprocalRank(grades: number[]): number {
  const first = grades.findIndex(isRelevant)
  return first < 0 ? 0 : 1 / (first + 1)
}

/** Discounted cumulative gain at k with linear gains g / log₂(i + 1), the convention of scikit-learn's ndcg_score. */
export function dcg(grades: number[], k: number): number {
  return grades.slice(0, k).reduce((s, g, i) => s + g / Math.log2(i + 2), 0)
}

/** NDCG at k: DCG divided by the DCG of the ideal (sorted) ranking of the same grades. */
export function ndcg(grades: number[], k: number): number {
  const ideal = dcg(
    [...grades].sort((a, b) => b - a),
    k,
  )
  return ideal ? dcg(grades, k) / ideal : 0
}
