import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

const K_MAX = 40
const KS = Array.from({ length: K_MAX }, (_, i) => i + 1)
const FACT = Array.from({ length: K_MAX + 1 }, (_, i) => i).reduce<number[]>(
  (f, i) => (f.push(i === 0 ? 1 : f[i - 1] * i), f),
  [],
)
const choose = (n: number, r: number) => FACT[n] / (FACT[r] * FACT[n - r])

/**
 * Probability that the plurality answer of k independent chains is correct. Each chain is correct with probability p;
 * otherwise it gives one of m wrong answers, each with probability (1 − p)/m. Ties are broken uniformly at random.
 * For c correct chains, the w = k − c wrong ones fall into m cells; the exponential generating function
 * A(x) = Σ_{j<c} xʲ/j! counts cells below c, and a cell exactly at c is a tie.
 */
function pluralityAccuracy(k: number, p: number, m: number): number {
  let acc = 0
  for (let c = 1; c <= k; c++) {
    const w = k - c
    const a = Array.from({ length: w + 1 }, (_, j) => (j < c ? 1 / FACT[j] : 0))
    const powers: number[][] = [Array.from({ length: w + 1 }, (_, j) => (j === 0 ? 1 : 0))]
    for (let n = 1; n <= m; n++) {
      const prev = powers[n - 1]
      const next = new Array<number>(w + 1).fill(0)
      for (let i = 0; i <= w; i++) {
        if (prev[i] === 0) continue
        for (let j = 0; j <= Math.min(c - 1, w - i); j++) next[i + j] += prev[i] * a[j]
      }
      powers.push(next)
    }
    let inner = 0
    for (let t = 0; t <= m; t++) {
      if (t * c > w) break
      inner += (choose(m, t) * FACT[c] ** -t * powers[m - t][w - t * c]) / (1 + t)
    }
    inner *= FACT[w] / m ** w
    acc += choose(k, c) * p ** c * (1 - p) ** w * inner
  }
  return acc
}

/** Majority-vote (self-consistency) accuracy against the number of sampled reasoning chains. */
export function SelfConsistency() {
  const p = useParam(0.4, { min: 0.05, max: 0.95, step: 0.01 })
  const m = useParam(3, { min: 1, max: 5, step: 1 })
  const k = useParam(9, { min: 1, max: K_MAX, step: 1 })

  const curve = useMemo(() => KS.map((kk) => pluralityAccuracy(kk, p.value, m.value)), [p.value, m.value])

  const series: XYSeries[] = [
    { name: 'majority vote of k chains', type: 'line', x: KS, y: curve, slot: 0 },
    { name: 'one chain (p)', type: 'line', x: [1, K_MAX], y: [p.value, p.value], slot: 1, dashed: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: k.value, label: 'k', onDrag: k.set }]
  const wrongEach = (1 - p.value) / m.value
  const acc = curve[k.value - 1]

  return (
    <Interactive
      title="Self-consistency: majority vote over sampled chains"
      caption="Each sampled chain of thought is independently correct with probability p. A wrong chain gives one of m distinct wrong answers, each with probability (1 − p)/m. The line is the probability that the most common answer among k chains is correct, with ties broken at random. The vote converges to the correct answer when p exceeds (1 − p)/m, even when p is below one half, and to a wrong answer otherwise. Drag the line labelled k, or use its slider."
      controls={
        <>
          <ParamSlider label="one-chain accuracy p" param={p} />
          <ParamSlider label="distinct wrong answers m" param={m} format={(v) => String(v)} />
          <ParamSlider label="chains sampled k" param={k} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="accuracy with k chains" value={formatNumber(acc)} />
          <Readout label="gain over one chain" value={formatNumber(acc - p.value)} />
          <Readout label="each wrong answer (1 − p)/m" value={formatNumber(wrongEach)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="chains sampled k"
        yLabel="accuracy"
        xRange={[1, K_MAX]}
        yRange={[0, 1]}
        height={300}
        handles={handles}
      />
    </Interactive>
  )
}
