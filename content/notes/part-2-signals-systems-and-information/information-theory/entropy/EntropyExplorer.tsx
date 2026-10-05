import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const log2 = (x: number) => Math.log(x) / Math.LN2
const entropy = (ps: number[]) => -ps.reduce((s, p) => (p > 0 ? s + p * log2(p) : s), 0)
const P_GRID = toFlat(linspace(0.001, 0.999, 199))
const BERNOULLI_CURVE = P_GRID.map((p) => entropy([p, 1 - p]))

/**
 * Left: the entropy of a coin as a function of P(heads), with the chosen p on the curve. Right: a categorical
 * distribution over K outcomes whose logits fall linearly with index; sharpness 0 is uniform and entropy log K.
 */
export function EntropyExplorer() {
  const state = useFigureState({
    p: float(0.9, { min: 0.01, max: 0.99, step: 0.01, label: 'coin: P(heads) p' }),
    k: int(6, { min: 2, max: 12, step: 1, label: 'outcomes K', format: (v) => String(v) }),
    sharpness: float(0.6, { min: 0, max: 3, step: 0.05, label: 'sharpness' }),
  })

  const coin = entropy([state.p, 1 - state.p])
  const coinSeries = [
    { name: 'H(p) in bits', x: P_GRID, y: BERNOULLI_CURVE, slot: 0 },
    { name: 'chosen p', x: [state.p], y: [coin], emphasis: true },
  ] as const
  // p is the horizontal position of a point on the curve, so dragging along the axis sets it.

  const cat = useMemo(() => {
    const logits = Array.from({ length: state.k }, (_, i) => -state.sharpness * i)
    const top = Math.max(...logits)
    const w = logits.map((z) => Math.exp(z - top))
    const total = w.reduce((a, b) => a + b, 0)
    const probs = w.map((x) => x / total)
    return { probs, h: entropy(probs) }
  }, [state.k, state.sharpness])
  const outcomes = cat.probs.map((_, i) => i + 1)
  const catSeries = [
    { name: 'P(outcome)', x: outcomes, y: cat.probs, slot: 0 },
    {
      name: 'uniform 1/K',
      x: [0.5, state.k + 0.5],
      y: [1 / state.k, 1 / state.k],
      dashed: true,
      slot: 1,
    },
  ] as const

  const xAxis = useAxis({ label: 'p = P(heads)', range: [0, 1] })
  const yAxis = useAxis({ label: 'entropy (bits)', range: [0, 1.05] })
  const xAxis2 = useAxis({ label: 'outcome', range: [0.5, 12.5] })
  const yAxis2 = useAxis({ label: 'probability', range: [0, 1] })
  return (
    <Figure
      title="Entropy of a coin and of a die with K faces"
      state={state}
      caption="Left: the entropy of a coin that lands heads with probability p, in bits. It is 1 bit for a fair coin and falls to 0 as the outcome becomes certain. Drag the line or use the slider. Right: a distribution over K outcomes whose probabilities fall geometrically with the outcome's index. At sharpness 0 it is uniform and its entropy reaches the maximum log₂ K; sharper distributions are more predictable and have lower entropy."

      readouts={
        <>
          <Readout label="coin entropy" value={`${formatNumber(coin)} bits`} />
          <Readout label="K-outcome entropy" value={`${formatNumber(cat.h)} bits`} />
          <Readout label="maximum log₂ K" value={`${formatNumber(log2(state.k))} bits`} />
          <Readout label="effective outcomes 2^H" value={formatNumber(2 ** cat.h)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...coinSeries[0]} />
          <Points {...coinSeries[1]} />
          <Handle {...state.handle('p', { label: 'p' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Bars {...catSeries[0]} />
          <Curve {...catSeries[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
