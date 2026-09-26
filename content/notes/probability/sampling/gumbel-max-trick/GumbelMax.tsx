import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const SAMPLES = 4000
const BINS = 40
const gumbel = (u: number) => -Math.log(-Math.log(Math.min(Math.max(u, 1e-12), 1 - 1e-12)))

/**
 * Left: how often argmax(logits + Gumbel noise) picks each category, against the softmax probabilities. Right: the
 * first coordinate of Gumbel-softmax samples at temperature τ: near 0 or 1 when τ is small, near its mean when τ is
 * large.
 */
export function GumbelMax() {
  const [logits, setLogits] = useState([1, 0, -0.5])
  const [tau, setTau] = useState(0.5)
  const setLogit = (k: number) => (v: number) => setLogits((l) => l.map((x, i) => (i === k ? v : x)))

  const probs = useMemo(() => {
    const e = logits.map(Math.exp)
    const s = e[0] + e[1] + e[2]
    return e.map((v) => v / s)
  }, [logits])

  const { freq, relaxed, relaxedMean } = useMemo(() => {
    const r = rng(21)
    const counts = [0, 0, 0]
    const first: number[] = []
    for (let i = 0; i < SAMPLES; i++) {
      const perturbed = logits.map((l) => l + gumbel(r.uniform()))
      counts[perturbed.indexOf(Math.max(...perturbed))]++
      // Gumbel-softmax with the same noise: softmax(perturbed / τ), computed stably.
      const top = Math.max(...perturbed)
      const w = perturbed.map((v) => Math.exp((v - top) / tau))
      first.push(w[0] / (w[0] + w[1] + w[2]))
    }
    const hist = new Array(BINS).fill(0)
    for (const y of first) hist[Math.min(Math.floor(y * BINS), BINS - 1)]++
    return {
      freq: counts.map((c) => c / SAMPLES),
      relaxed: {
        name: 'y₁ of Gumbel-softmax samples',
        type: 'bar',
        x: hist.map((_, i) => (i + 0.5) / BINS),
        y: hist.map((c) => (c * BINS) / SAMPLES),
        slot: 0,
      } satisfies XYSeries,
      relaxedMean: first.reduce((a, b) => a + b, 0) / SAMPLES,
    }
  }, [logits, tau])

  const categories = [1, 2, 3]
  const left: XYSeries[] = [
    { name: 'argmax frequency', type: 'bar', x: categories, y: freq, slot: 0 },
    { name: 'softmax probability πₖ', type: 'scatter', x: categories, y: probs, emphasis: true },
  ]

  return (
    <Interactive
      title="Sampling a category by perturbing its logits"
      caption="Add independent Gumbel noise to each logit and take the largest: over 4,000 draws, each category wins with exactly its softmax probability (left). Replacing the argmax by a softmax at temperature τ gives the Gumbel-softmax, a continuous sample on the simplex. The right panel shows its first coordinate. At small τ the samples are almost one-hot, so y₁ sits near 0 or 1. At large τ they flatten towards the centre of the simplex."
      controls={
        <>
          {logits.map((l, k) => (
            <ParamSlider
              key={k}
              label={`logit ${k + 1}`}
              value={l}
              onChange={setLogit(k)}
              min={-3}
              max={3}
              step={0.1}
            />
          ))}
          <ParamSlider label="temperature τ" value={tau} onChange={setTau} min={0.05} max={5} step={0.05} />
        </>
      }
      readout={
        <>
          {probs.map((p, k) => (
            <Readout
              key={k}
              label={`category ${k + 1}: π, frequency`}
              value={`${formatNumber(p)}, ${formatNumber(freq[k])}`}
            />
          ))}
          <Readout label="mean of y₁" value={formatNumber(relaxedMean)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={280}
          series={left}
          xRange={[0.4, 3.6]}
          yRange={[0, 1]}
          xLabel="category"
          yLabel="probability"
        />
        <XYChart height={280} series={[relaxed]} xRange={[0, 1]} yRange={[0, undefined]} xLabel="y₁" yLabel="density" />
      </div>
    </Interactive>
  )
}
