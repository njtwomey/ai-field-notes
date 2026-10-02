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
} from 'aifn-render'
import { linspace } from '@/lib/math'

const log2 = (x: number) => Math.log(x) / Math.LN2
const entropy = (ps: number[]) => -ps.reduce((s, p) => (p > 0 ? s + p * log2(p) : s), 0)
const P_GRID = linspace(0.001, 0.999, 199)
const BERNOULLI_CURVE = P_GRID.map((p) => entropy([p, 1 - p]))

/**
 * Left: the entropy of a coin as a function of P(heads), with the chosen p on the curve. Right: a categorical
 * distribution over K outcomes whose logits fall linearly with index; sharpness 0 is uniform and entropy log K.
 */
export function EntropyExplorer() {
  const p = useParam(0.9, { min: 0.01, max: 0.99, step: 0.01 })
  const k = useParam(6, { min: 2, max: 12, step: 1 })
  const sharpness = useParam(0.6, { min: 0, max: 3, step: 0.05 })

  const coin = entropy([p.value, 1 - p.value])
  const coinSeries: XYSeries[] = [
    { name: 'H(p) in bits', type: 'line', x: P_GRID, y: BERNOULLI_CURVE, slot: 0 },
    { name: 'chosen p', type: 'scatter', x: [p.value], y: [coin], emphasis: true },
  ]
  // p is the horizontal position of a point on the curve, so dragging along the axis sets it.
  const handles: Handle[] = [{ kind: 'x', at: p.value, label: 'p', onDrag: (x) => p.set(x) }]

  const cat = useMemo(() => {
    const logits = Array.from({ length: k.value }, (_, i) => -sharpness.value * i)
    const top = Math.max(...logits)
    const w = logits.map((z) => Math.exp(z - top))
    const total = w.reduce((a, b) => a + b, 0)
    const probs = w.map((x) => x / total)
    return { probs, h: entropy(probs) }
  }, [k.value, sharpness.value])
  const outcomes = cat.probs.map((_, i) => i + 1)
  const catSeries: XYSeries[] = [
    { name: 'P(outcome)', type: 'bar', x: outcomes, y: cat.probs, slot: 0 },
    {
      name: 'uniform 1/K',
      type: 'line',
      x: [0.5, k.value + 0.5],
      y: [1 / k.value, 1 / k.value],
      dashed: true,
      slot: 1,
    },
  ]

  return (
    <Interactive
      title="Entropy of a coin and of a die with K faces"
      caption="Left: the entropy of a coin that lands heads with probability p, in bits. It is 1 bit for a fair coin and falls to 0 as the outcome becomes certain. Drag the line or use the slider. Right: a distribution over K outcomes whose probabilities fall geometrically with the outcome's index. At sharpness 0 it is uniform and its entropy reaches the maximum log₂ K; sharper distributions are more predictable and have lower entropy."
      controls={
        <>
          <ParamSlider label="coin: P(heads) p" param={p} />
          <ParamSlider label="outcomes K" param={k} format={(v) => String(v)} withArrows />
          <ParamSlider label="sharpness" param={sharpness} />
        </>
      }
      readout={
        <>
          <Readout label="coin entropy" value={`${formatNumber(coin)} bits`} />
          <Readout label="K-outcome entropy" value={`${formatNumber(cat.h)} bits`} />
          <Readout label="maximum log₂ K" value={`${formatNumber(log2(k.value))} bits`} />
          <Readout label="effective outcomes 2^H" value={formatNumber(2 ** cat.h)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={coinSeries}
          xLabel="p = P(heads)"
          yLabel="entropy (bits)"
          xRange={[0, 1]}
          yRange={[0, 1.05]}
          handles={handles}
          height={300}
        />
        <XYChart
          series={catSeries}
          xLabel="outcome"
          yLabel="probability"
          xRange={[0.5, 12.5]}
          yRange={[0, 1]}
          height={300}
        />
      </div>
    </Interactive>
  )
}
