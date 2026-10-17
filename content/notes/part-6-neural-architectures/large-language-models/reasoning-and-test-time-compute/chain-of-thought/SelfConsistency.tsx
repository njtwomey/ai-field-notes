import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

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
  const state = useFigureState({
    p: float(0.4, { min: 0.05, max: 0.95, step: 0.01, label: 'one-chain accuracy p' }),
    m: int(3, { min: 1, max: 5, step: 1, label: 'distinct wrong answers m', format: (v) => String(v) }),
    k: int(9, { min: 1, max: K_MAX, step: 1, label: 'chains sampled k', format: (v) => String(v) }),
  })

  const curve = useMemo(() => KS.map((kk) => pluralityAccuracy(kk, state.p, state.m)), [state.p, state.m])

  const series = [
    { name: 'majority vote of k chains', x: KS, y: curve, slot: 0 },
    { name: 'one chain (p)', x: [1, K_MAX], y: [state.p, state.p], slot: 1, dashed: true },
  ] as const
  const wrongEach = (1 - state.p) / state.m
  const acc = curve[state.k - 1]

  const xAxis = useAxis({ label: 'chains sampled k', range: [1, K_MAX] })
  const yAxis = useAxis({ label: 'accuracy', range: [0, 1] })
  return (
    <Figure
      title="Self-consistency: majority vote over sampled chains"
      state={state}
      caption="Each sampled chain of thought is independently correct with probability p. A wrong chain gives one of m distinct wrong answers, each with probability (1 − p)/m. The line is the probability that the most common answer among k chains is correct, with ties broken at random. The vote converges to the correct answer when p exceeds (1 − p)/m, even when p is below one half, and to a wrong answer otherwise. Drag the line labelled k, or use its slider."

      readouts={
        <>
          <Readout label="accuracy with k chains" value={formatNumber(acc)} />
          <Readout label="gain over one chain" value={formatNumber(acc - state.p)} />
          <Readout label="each wrong answer (1 − p)/m" value={formatNumber(wrongEach)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle {...state.handle('k', { label: 'k' })} />
      </Plot>
    </Figure>
  )
}
