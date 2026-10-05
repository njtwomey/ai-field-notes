import { useMemo } from 'react'
import {
  Bars,
  Figure,
  float,
  formatNumber,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const SAMPLES = 4000
const BINS = 40
const gumbel = (u: number) => -Math.log(-Math.log(Math.min(Math.max(u, 1e-12), 1 - 1e-12)))

/**
 * Left: how often argmax(logits + Gumbel noise) picks each category, against the softmax probabilities. Right: the
 * first coordinate of Gumbel-softmax samples at temperature τ: near 0 or 1 when τ is small, near its mean when τ is
 * large.
 */
export function GumbelMax() {
  const state = useFigureState({
    l1: slider(-3, 3, 1, { step: 0.1, label: 'logit 1' }),
    l2: slider(-3, 3, 0, { step: 0.1, label: 'logit 2' }),
    l3: slider(-3, 3, -0.5, { step: 0.1, label: 'logit 3' }),
    tau: float(0.5, { min: 0.05, max: 5, step: 0.05, label: 'temperature τ' }),
  })
  const logits = useMemo(() => [state.l1, state.l2, state.l3], [state.l1, state.l2, state.l3])

  const probs = useMemo(() => {
    const e = logits.map(Math.exp)
    const s = e[0] + e[1] + e[2]
    return e.map((v) => v / s)
  }, [logits])

  const { freq, relaxed, relaxedMean } = useMemo(() => {
    const r = stream(21)
    const counts = [0, 0, 0]
    const first: number[] = []
    for (let i = 0; i < SAMPLES; i++) {
      const perturbed = logits.map((l) => l + gumbel(uniform(r)))
      counts[perturbed.indexOf(Math.max(...perturbed))]++
      // Gumbel-softmax with the same noise: softmax(perturbed / τ), computed stably.
      const top = Math.max(...perturbed)
      const w = perturbed.map((v) => Math.exp((v - top) / state.tau))
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
      } satisfies SeriesSpec,
      relaxedMean: first.reduce((a, b) => a + b, 0) / SAMPLES,
    }
  }, [logits, state.tau])

  const categories = [1, 2, 3]
  const left = [
    { name: 'argmax frequency', x: categories, y: freq, slot: 0 },
    { name: 'softmax probability πₖ', x: categories, y: probs, emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'category', range: [0.4, 3.6] })
  const yAxis = useAxis({ label: 'probability', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'y₁', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Sampling a category by perturbing its logits"
      state={state}
      caption="Add independent Gumbel noise to each logit and take the largest: over 4,000 draws, each category wins with exactly its softmax probability (left). Replacing the argmax by a softmax at temperature τ gives the Gumbel-softmax, a continuous sample on the simplex. The right panel shows its first coordinate. At small τ the samples are almost one-hot, so y₁ sits near 0 or 1. At large τ they flatten towards the centre of the simplex."
      readouts={
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
        <Plot x={xAxis} y={yAxis} height={280}>
          <Bars {...left[0]} />
          <Points {...left[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          {seriesLayers([relaxed])}
        </Plot>
      </div>
    </Figure>
  )
}
