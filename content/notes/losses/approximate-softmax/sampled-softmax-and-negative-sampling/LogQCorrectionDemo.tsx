import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

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
  const r = rng(seed)
  return Array.from({ length: M }, () => spread * r.normal())
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
  const r = rng(seed)
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
      const u = r.uniform()
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
  const [logM, setLogM] = useState(4)
  const [alpha, setAlpha] = useState(1)
  const [spread, setSpread] = useState(1)
  const [seed, setSeed] = useState(1)
  const m = 2 ** logM
  const scores = useMemo(() => drawScores(seed, spread), [seed, spread])
  const r = useMemo(() => simulate(scores, alpha, m, seed + 7), [scores, alpha, m, seed])

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'full softmax', type: 'line', x: REFERENCE_X, y: REFERENCE_Y, muted: true, dashed: true },
      { name: 'sampled, no correction', type: 'line', x: BIN_X, y: r.plain, slot: 0 },
      { name: 'sampled, logQ-corrected', type: 'line', x: BIN_X, y: r.fixed, slot: 1 },
    ],
    [r],
  )

  return (
    <Interactive
      title="The logQ correction removes the popularity bias"
      caption="A catalogue of 1000 negatives has Zipf popularity q_j ∝ rank^(−α) and scores that ignore popularity; the positive scores 2. Each line shows, per popularity bin, the expected push-down on the negatives' scores under sampled softmax with m negatives drawn from q, divided by the push-down of the full softmax. The dashed line at 1 is the full softmax. Without the correction the ratio grows in proportion to q_j, so the most popular items are pushed down far too hard. With the correction the ratio tends to 1 as m grows. At small m the corrected pushes still follow popularity, because a softmax over a few candidates gives almost all of its push to whichever negative was drawn."
      controls={
        <>
          <ParamSlider
            label="sampled negatives m"
            value={logM}
            onChange={setLogM}
            min={0}
            max={10}
            step={1}
            format={(v) => String(2 ** v)}
            withArrows
          />
          <ParamSlider label="Zipf exponent α" value={alpha} onChange={setAlpha} min={0} max={1.5} step={0.05} />
          <ParamSlider label="score spread σ" value={spread} onChange={setSpread} min={0} max={2} step={0.1} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New scores and samples</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="popularity share of the top 10 items" value={formatNumber(r.topShare)} />
          <Readout label="full loss" value={formatNumber(r.fullLoss)} />
          <Readout label="expected sampled loss, no correction" value={formatNumber(r.plainLoss)} />
          <Readout label="expected sampled loss, corrected" value={formatNumber(r.fixedLoss)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="log₁₀ popularity rank (1 = most popular)"
        yLabel="push-down ÷ full softmax"
        xRange={X_RANGE}
        yRange={Y_RANGE}
        yLog
        height={320}
      />
    </Interactive>
  )
}
