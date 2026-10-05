import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const M = 1000
const POSITIVE_SCORE = 2
// Popularity-rank bins [1], [2, 3], [4, 7], …, [512, 1000]: equal widths on a log scale.
const BIN_STARTS = Array.from({ length: 10 }, (_, b) => 2 ** b)
const binOf = (rank: number) => Math.min(9, Math.floor(Math.log2(rank)))
const BIN_X = BIN_STARTS.map((lo, b) => Math.log10(Math.sqrt(lo * Math.min(M, 2 ** (b + 1) - 1))))
const X_RANGE: [number, number] = [0, 3]
const Y_RANGE: [number, number] = [0.01, 1000]
const REFERENCE_X = [0, 3]
const REFERENCE_Y = [1, 1]

/** Scores of the catalogue's M negatives for one query, independent of popularity. */
function drawScores(seed: number, spread: number): number[] {
  const r = stream(seed)
  return Array.from({ length: M }, () => spread * normal(r))
}

/**
 * Expected push-down −∂L/∂s_j on each negative under sampled softmax, by Monte Carlo with common random numbers for the
 * plain and the logQ-corrected loss, summed per popularity bin and divided by the full softmax's exact push-down.
 */
function simulate(scores: number[], alpha: number, m: number, seed: number) {
  const weights = Array.from({ length: M }, (_, j) => (j + 1) ** -alpha)
  const total = weights.reduce((a, b) => a + b, 0)
  const q = weights.map((w) => w / total)
  const cdf: number[] = []
  q.reduce((acc, p, j) => (cdf[j] = acc + p), 0)
  const expS = scores.map((s) => Math.exp(s))
  const ePos = Math.exp(POSITIVE_SCORE)

  // Full softmax over the positive and all M negatives.
  const zFull = ePos + expS.reduce((a, b) => a + b, 0)
  const fullBins = Array<number>(10).fill(0)
  expS.forEach((e, j) => (fullBins[binOf(j + 1)] += e / zFull))
  const fullLoss = -Math.log(ePos / zFull)

  const trials = Math.max(200, Math.min(4000, Math.round(80000 / m)))
  const r = stream(seed)
  const sample = new Int32Array(m)
  const plainBins = Array<number>(10).fill(0)
  const fixedBins = Array<number>(10).fill(0)
  let plainLoss = 0
  let fixedLoss = 0
  for (let t = 0; t < trials; t++) {
    let dPlain = ePos
    let dFixed = ePos
    for (let k = 0; k < m; k++) {
      // Inverse-CDF draw by binary search.
      const u = uniform(r)
      let lo = 0
      let hi = M - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (cdf[mid] < u) lo = mid + 1
        else hi = mid
      }
      sample[k] = lo
      dPlain += expS[lo]
      dFixed += expS[lo] / (m * q[lo])
    }
    for (let k = 0; k < m; k++) {
      const j = sample[k]
      plainBins[binOf(j + 1)] += expS[j] / dPlain
      fixedBins[binOf(j + 1)] += expS[j] / (m * q[j]) / dFixed
    }
    plainLoss += Math.log(dPlain) - POSITIVE_SCORE
    fixedLoss += Math.log(dFixed) - POSITIVE_SCORE
  }
  const ratio = (bins: number[]) => bins.map((v, b) => Math.max(Y_RANGE[0], v / trials / fullBins[b]))
  return {
    plain: ratio(plainBins),
    fixed: ratio(fixedBins),
    fullLoss,
    plainLoss: plainLoss / trials,
    fixedLoss: fixedLoss / trials,
    topShare: q.slice(0, 10).reduce((a, b) => a + b, 0),
  }
}

/** Sampled softmax on a Zipf catalogue: how the logQ correction removes the popularity bias of sampled negatives. */
export function LogQCorrectionDemo() {
  const state = useFigureState({
    logM: int(4, { min: 0, max: 10, step: 1, label: 'sampled negatives m', format: (v) => String(2 ** v) }),
    alpha: float(1, { min: 0, max: 1.5, step: 0.05, label: 'Zipf exponent α' }),
    spread: float(1, { min: 0, max: 2, step: 0.1, label: 'score spread σ' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const m = 2 ** state.logM
  const scores = useMemo(() => drawScores(state.seed, state.spread), [state.seed, state.spread])
  const r = useMemo(() => simulate(scores, state.alpha, m, state.seed + 7), [scores, state.alpha, m, state.seed])

  const series = useMemo(
    () =>
      [
        { name: 'full softmax', x: REFERENCE_X, y: REFERENCE_Y, muted: true, dashed: true },
        { name: 'sampled, no correction', x: BIN_X, y: r.plain, slot: 0 },
        { name: 'sampled, logQ-corrected', x: BIN_X, y: r.fixed, slot: 1 },
      ] as const,
    [r],
  )

  const xAxis = useAxis({ label: 'log₁₀ popularity rank (1 = most popular)', range: X_RANGE })
  const yAxis = useAxis({ label: 'push-down ÷ full softmax', range: Y_RANGE, log: true })
  return (
    <Figure
      title="The logQ correction removes the popularity bias"
      state={state}
      caption="A catalogue of 1000 negatives has Zipf popularity q_j ∝ rank^(−α) and scores that ignore popularity; the positive scores 2. Each line shows, per popularity bin, the expected push-down on the negatives' scores under sampled softmax with m negatives drawn from q, divided by the push-down of the full softmax. The dashed line at 1 is the full softmax. Without the correction the ratio grows in proportion to q_j, so the most popular items are pushed down far too hard. With the correction the ratio tends to 1 as m grows. At small m the corrected pushes still follow popularity, because a softmax over a few candidates gives almost all of its push to whichever negative was drawn."

      readouts={
        <>
          <Readout label="popularity share of the top 10 items" value={formatNumber(r.topShare)} />
          <Readout label="full loss" value={formatNumber(r.fullLoss)} />
          <Readout label="expected sampled loss, no correction" value={formatNumber(r.plainLoss)} />
          <Readout label="expected sampled loss, corrected" value={formatNumber(r.fixedLoss)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
