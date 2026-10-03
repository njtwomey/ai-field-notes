/** Ordinal classification metrics from a (possibly fractional) confusion matrix C[true][predicted]. */
export type Table = number[][]

export type OrdinalScores = {
  accuracy: number
  mae: number
  macroMae: number
  linearKappa: number
  quadraticKappa: number
  kendallTauB: number
  spearman: number
}

const total = (t: Table) => t.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0)

function weightedKappa(t: Table, power: 1 | 2): number {
  const k = t.length
  const n = total(t)
  const rows = t.map((row) => row.reduce((a, b) => a + b, 0))
  const cols = t[0].map((_, j) => t.reduce((s, row) => s + row[j], 0))
  let observed = 0
  let expected = 0
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      const w = Math.abs(i - j) ** power
      observed += w * t[i][j]
      expected += (w * rows[i] * cols[j]) / n
    }
  return expected > 0 ? 1 - observed / expected : NaN
}

/** Kendall's τ_b from a contingency table: concordant minus discordant pairs, corrected for ties on each side. */
function kendallTauB(t: Table): number {
  const k = t.length
  const n = total(t)
  let concordant = 0
  let discordant = 0
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++)
      for (let a = i + 1; a < k; a++)
        for (let b = 0; b < k; b++) {
          if (b > j) concordant += t[i][j] * t[a][b]
          else if (b < j) discordant += t[i][j] * t[a][b]
        }
  const pairs = (n * (n - 1)) / 2
  const rows = t.map((row) => row.reduce((a, b) => a + b, 0))
  const cols = t[0].map((_, j) => t.reduce((s, row) => s + row[j], 0))
  const tiesRows = rows.reduce((s, r) => s + (r * (r - 1)) / 2, 0)
  const tiesCols = cols.reduce((s, c) => s + (c * (c - 1)) / 2, 0)
  const denominator = Math.sqrt((pairs - tiesRows) * (pairs - tiesCols))
  return denominator > 1e-9 ? (concordant - discordant) / denominator : NaN
}

/** Spearman's ρ with mid-ranks: the Pearson correlation of the tied ranks, weighted by the table's counts. */
function spearman(t: Table): number {
  const k = t.length
  const rows = t.map((row) => row.reduce((a, b) => a + b, 0))
  const cols = t[0].map((_, j) => t.reduce((s, row) => s + row[j], 0))
  const midRanks = (counts: number[]) => {
    let before = 0
    return counts.map((c) => {
      const r = before + (c + 1) / 2
      before += c
      return r
    })
  }
  const rr = midRanks(rows)
  const rc = midRanks(cols)
  const n = total(t)
  const mean = (n + 1) / 2
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) sxy += t[i][j] * (rr[i] - mean) * (rc[j] - mean)
  rows.forEach((r, i) => (sxx += r * (rr[i] - mean) ** 2))
  cols.forEach((c, j) => (syy += c * (rc[j] - mean) ** 2))
  return sxx > 1e-9 && syy > 1e-9 ? sxy / Math.sqrt(sxx * syy) : NaN
}

export function ordinalScores(t: Table): OrdinalScores {
  const n = total(t)
  const k = t.length
  let correct = 0
  let absolute = 0
  const perClass: number[] = []
  t.forEach((row, i) => {
    const count = row.reduce((a, b) => a + b, 0)
    const err = row.reduce((s, c, j) => s + c * Math.abs(i - j), 0)
    correct += row[i]
    absolute += err
    if (count > 0) perClass.push(err / count)
  })
  return {
    accuracy: correct / n,
    mae: absolute / n,
    macroMae: perClass.reduce((a, b) => a + b, 0) / perClass.length,
    linearKappa: weightedKappa(t, 1),
    quadraticKappa: k > 1 ? weightedKappa(t, 2) : NaN,
    kendallTauB: kendallTauB(t),
    spearman: spearman(t),
  }
}

/**
 * The expected confusion matrix of n items with class shares `prior` when a fraction `rate` of each class is predicted
 * `distance` classes away: upwards if the scale allows, else downwards, else at the far end of the scale.
 */
export function shiftedTable(prior: number[], rate: number, distance: number, n = 1000): Table {
  const k = prior.length
  return prior.map((p, i) => {
    const row = Array(k).fill(0) as number[]
    const target = i + distance < k ? i + distance : i - distance >= 0 ? i - distance : i < k / 2 ? k - 1 : 0
    row[i] += n * p * (1 - rate)
    row[target] += n * p * rate
    return row
  })
}

/** Every item predicted as the most frequent class. */
export function majorityTable(prior: number[], n = 1000): Table {
  const top = prior.indexOf(Math.max(...prior))
  return prior.map((p) => prior.map((_, j) => (j === top ? n * p : 0)))
}
