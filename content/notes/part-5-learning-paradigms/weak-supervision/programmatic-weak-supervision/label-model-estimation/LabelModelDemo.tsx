import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'

const N = 3000

/** Accuracies spaced evenly from 0.55 up to the best labelling function's accuracy. */
const accuracies = (m: number, best: number) =>
  Array.from({ length: m }, (_, i) => (m === 1 ? best : 0.55 + ((best - 0.55) * i) / (m - 1))).reverse()

/**
 * Simulates labelling functions with votes in {−1, 0, +1}. Each covers a point with probability `coverage` and, when it
 * does, is right with its accuracy. With probability ρ, LF 2 copies LF 1's vote wherever LF 1 votes, which breaks
 * conditional independence.
 */
function simulate(m: number, best: number, coverage: number, rho: number, seed: number) {
  const g = stream(seed)
  const acc = accuracies(m, best)
  const y: number[] = []
  const L: number[][] = []
  for (let i = 0; i < N; i++) {
    const yi = uniform(g) < 0.5 ? 1 : -1
    const row = acc.map((a) => (uniform(g) < coverage ? (uniform(g) < a ? yi : -yi) : 0))
    if (m > 1 && row[0] !== 0 && uniform(g) < rho) row[1] = row[0]
    y.push(yi)
    L.push(row)
  }
  // Realised accuracy of each LF on the points it votes on; copying changes LF 2's accuracy towards LF 1's.
  const realised = acc.map((_, j) => {
    let votes = 0
    let right = 0
    L.forEach((row, i) => {
      if (row[j] === 0) return
      votes++
      if (row[j] === y[i]) right++
    })
    return votes ? right / votes : 0.5
  })
  return { acc: realised, y, L }
}

/**
 * Triplet method: under conditional independence and symmetric errors, E[λᵢλⱼ] = aᵢaⱼ with aᵢ = E[λᵢY]. Solving three
 * such equations gives |aᵢ| = √|E[λᵢλⱼ]E[λᵢλₖ] / E[λⱼλₖ]|, averaged over every pair (j, k).
 */
function tripletAccuracies(L: number[][], m: number) {
  const M = Array.from({ length: m }, () => new Array<number>(m).fill(0))
  const cover = new Array<number>(m).fill(0)
  for (const row of L) {
    for (let i = 0; i < m; i++) {
      if (row[i] !== 0) cover[i]++
      for (let j = i + 1; j < m; j++) M[i][j] += row[i] * row[j]
    }
  }
  for (let i = 0; i < m; i++) {
    cover[i] /= L.length
    for (let j = i + 1; j < m; j++) {
      M[i][j] /= L.length
      M[j][i] = M[i][j]
    }
  }
  return Array.from({ length: m }, (_, i) => {
    const est: number[] = []
    for (let j = 0; j < m; j++)
      for (let k = j + 1; k < m; k++) {
        if (j === i || k === i || Math.abs(M[j][k]) < 1e-6) continue
        est.push(Math.sqrt(Math.abs((M[i][j] * M[i][k]) / M[j][k])))
      }
    const a = est.length ? est.reduce((s, v) => s + v, 0) / est.length : 0
    // aᵢ = βᵢ(2αᵢ − 1), with βᵢ the coverage, so αᵢ = (1 + aᵢ/βᵢ)/2. Sign chosen as better than random.
    return Math.min(0.99, Math.max(0.5, (1 + a / Math.max(cover[i], 1e-9)) / 2))
  })
}

/** Accuracy of sign(Σ wᵢλᵢ) against y; points with no net vote are scored as a coin flip. */
function weightedVote(L: number[][], y: number[], w: number[]) {
  let hits = 0
  L.forEach((row, i) => {
    const s = row.reduce((t, l, j) => t + w[j] * l, 0)
    hits += Math.abs(s) < 1e-12 ? 0.5 : Math.sign(s) === y[i] ? 1 : 0
  })
  return hits / y.length
}

export function LabelModelDemo() {
  const state = useFigureState({
    lfs: int(5, { min: 3, max: 8, step: 1, label: 'labelling functions m', format: (v) => String(v) }),
    best: float(0.9, { min: 0.6, max: 0.95, step: 0.05, label: 'best LF accuracy' }),
    coverage: slider(0.1, 1, 0.5, { step: 0.05, label: 'coverage' }),
    rho: slider(0, 1, 0, { step: 0.05, label: 'correlation ρ of LF 1 and LF 2' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const m = state.lfs

  const r = useMemo(() => {
    const d = simulate(m, state.best, state.coverage, state.rho, state.seed)
    const est = tripletAccuracies(d.L, m)
    const logit = (a: number) => Math.log(a / (1 - a))
    return {
      acc: d.acc,
      est,
      mv: weightedVote(d.L, d.y, new Array<number>(m).fill(1)),
      lm: weightedVote(d.L, d.y, est.map(logit)),
      oracle: weightedVote(d.L, d.y, d.acc.map(logit)),
      err: est.reduce((s, a, i) => s + Math.abs(a - d.acc[i]), 0) / m,
    }
  }, [m, state.best, state.coverage, state.rho, state.seed])

  const index = Array.from({ length: m }, (_, i) => i + 1)
  const series = [
    { name: 'true accuracy', x: index, y: r.acc, slot: 0 },
    { name: 'estimated (triplet method)', x: index, y: r.est, slot: 1 },
  ] as const
  const segments = index.map((x, i) => ({
    from: [x, r.acc[i]] as [number, number],
    to: [x, r.est[i]] as [number, number],
  }))

  const xAxis = useAxis({ label: 'labelling function', range: [0.5, m + 0.5] })
  const yAxis = useAxis({ label: 'accuracy when voting', range: [0.45, 1] })
  return (
    <Figure
      title="Estimating labelling-function accuracies without labels"
      state={state}
      caption="Three thousand balanced binary points and m labelling functions (LFs). Each LF votes on a point with the chosen coverage and, when it votes, is right with its accuracy; accuracies run evenly from the best LF down to 0.55. The triplet method estimates every accuracy from agreement rates alone. The label model then weights each vote by the log-odds of its estimated accuracy, while majority vote weights all votes equally. With independent LFs the estimates sit on the true values and the label model beats majority vote. Raise the correlation ρ, the probability that LF 2 copies LF 1: the two now agree more than independence allows, their estimated accuracies inflate, and the label model counts one opinion twice."

      readouts={
        <>
          <Readout label="majority vote" value={formatNumber(r.mv)} />
          <Readout label="label model" value={formatNumber(r.lm)} />
          <Readout label="label model with true accuracies" value={formatNumber(r.oracle)} />
          <Readout label="mean |estimated − true|" value={formatNumber(r.err)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...series[0]} />
        <Points {...series[1]} />
        <Segments segments={segments} />
      </Plot>
    </Figure>
  )
}
