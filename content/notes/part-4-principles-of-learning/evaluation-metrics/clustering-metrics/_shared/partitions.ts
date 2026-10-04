import { logFactorial } from 'aifn/numerics/special'
/**
 * Scores comparing two partitions of the same n items: pair counting (Rand, adjusted Rand, Fowlkes–Mallows) and
 * information theory (mutual information, NMI, AMI, homogeneity, completeness, V-measure). Natural logarithms throughout,
 * matching scikit-learn.
 */

export type Scores = {
  rand: number
  ari: number
  fmi: number
  mi: number
  nmi: number
  ami: number
  homogeneity: number
  completeness: number
  vMeasure: number
}

type Table = { cells: number[][]; rows: number[]; cols: number[]; n: number }

/** Contingency table: rows are the true classes, columns the predicted clusters. */
export function contingency(truth: number[], pred: number[]): Table {
  const ti = [...new Set(truth)]
  const pi = [...new Set(pred)]
  const cells = ti.map(() => pi.map(() => 0))
  truth.forEach((t, k) => cells[ti.indexOf(t)][pi.indexOf(pred[k])]++)
  return {
    cells,
    rows: cells.map((r) => r.reduce((a, b) => a + b, 0)),
    cols: pi.map((_, j) => cells.reduce((a, r) => a + r[j], 0)),
    n: truth.length,
  }
}

const pairs = (m: number) => (m * (m - 1)) / 2
const entropy = (counts: number[], n: number) =>
  -counts.reduce((s, c) => (c > 0 ? s + (c / n) * Math.log(c / n) : s), 0)

/** E[MI] under random labellings with the same cluster sizes (Vinh, Epps & Bailey 2010). */
function expectedMutualInformation(rows: number[], cols: number[], n: number): number {
  let emi = 0
  const lfN = logFactorial(n)
  for (const a of rows) {
    for (const b of cols) {
      const start = Math.max(1, a + b - n)
      const end = Math.min(a, b)
      for (let nij = start; nij <= end; nij++) {
        const logP =
          logFactorial(a) +
          logFactorial(b) +
          logFactorial(n - a) +
          logFactorial(n - b) -
          lfN -
          logFactorial(nij) -
          logFactorial(a - nij) -
          logFactorial(b - nij) -
          logFactorial(n - a - b + nij)
        emi += (nij / n) * Math.log((n * nij) / (a * b)) * Math.exp(logP)
      }
    }
  }
  return emi
}

export function scores(truth: number[], pred: number[]): Scores {
  const { cells, rows, cols, n } = contingency(truth, pred)
  const sumCells = cells.flat().reduce((s, c) => s + pairs(c), 0)
  const sumRows = rows.reduce((s, c) => s + pairs(c), 0)
  const sumCols = cols.reduce((s, c) => s + pairs(c), 0)
  const total = pairs(n)
  // Pair counts: TP = pairs together in both, FP = together only in the prediction, FN = together only in the truth.
  const tp = sumCells
  const fp = sumCols - sumCells
  const fn = sumRows - sumCells
  const tn = total - tp - fp - fn
  const expected = (sumRows * sumCols) / total
  const maxIndex = (sumRows + sumCols) / 2
  const ari = maxIndex === expected ? 1 : (sumCells - expected) / (maxIndex - expected)

  const hTrue = entropy(rows, n)
  const hPred = entropy(cols, n)
  let mi = 0
  cells.forEach((row, i) =>
    row.forEach((c, j) => {
      if (c > 0) mi += (c / n) * Math.log((n * c) / (rows[i] * cols[j]))
    }),
  )
  const mean = (hTrue + hPred) / 2
  const emi = expectedMutualInformation(rows, cols, n)
  const homogeneity = hTrue === 0 ? 1 : mi / hTrue
  const completeness = hPred === 0 ? 1 : mi / hPred
  return {
    rand: (tp + tn) / total,
    ari,
    fmi: tp === 0 ? 0 : tp / Math.sqrt((tp + fp) * (tp + fn)),
    mi,
    nmi: mean === 0 ? 1 : mi / mean,
    ami: mean === emi ? 1 : (mi - emi) / (mean - emi),
    homogeneity,
    completeness,
    vMeasure: homogeneity + completeness === 0 ? 0 : (2 * homogeneity * completeness) / (homogeneity + completeness),
  }
}
