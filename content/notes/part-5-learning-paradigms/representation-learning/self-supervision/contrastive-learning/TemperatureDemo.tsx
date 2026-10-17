import { useMemo } from 'react'
import {
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
import { normal, stream } from 'aifn-compute/foundation/random'

const NEGATIVES = 256
const CURVE_X = Array.from({ length: 201 }, (_, i) => -1 + i / 100)

/** Cosine similarities of 256 negatives to the anchor: mostly unrelated, a few near the positive. */
function draw(seed: number): number[] {
  const r = stream(seed)
  return Array.from({ length: NEGATIVES }, () => Math.max(-1, Math.min(1, 0.1 + 0.2 * normal(r))))
}

/** InfoNCE for one anchor: the positive at similarity sPos against 256 negatives, at temperature τ. */
export function TemperatureDemo() {
  const state = useFigureState({
    logTau: float(-1, {
      min: -2,
      max: 0,
      step: 0.05,
      label: 'temperature τ',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    sPos: float(0.6, { min: -1, max: 1, step: 0.01, label: 'positive similarity' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const tau = 10 ** state.logTau
  const negs = useMemo(() => draw(state.seed), [state.seed])

  const r = useMemo(() => {
    // Softmax over the positive and all negatives, computed relative to the largest logit.
    const top = Math.max(state.sPos, ...negs)
    const ePos = Math.exp((state.sPos - top) / tau)
    const eNeg = negs.map((s) => Math.exp((s - top) / tau))
    const zNeg = eNeg.reduce((a, b) => a + b, 0)
    const pPos = ePos / (ePos + zNeg)
    // Each negative's share of the total gradient weight on negatives (∂L/∂s_j ∝ p_j).
    const share = eNeg.map((e) => e / zNeg)
    const effective = 1 / share.reduce((a, q) => a + q * q, 0)
    const hardest = Math.max(...negs)
    const relative = negs.map((s) => Math.exp((s - hardest) / tau))
    const curve = CURVE_X.map((x) => Math.min(1, Math.exp((x - hardest) / tau)))
    return { loss: -Math.log(pPos), pPos, effective, relative, curve }
  }, [negs, state.sPos, tau])

  const series = [
    { name: 'weight ∝ exp(s/τ)', x: CURVE_X, y: r.curve, muted: true },
    { name: 'negatives', x: negs, y: r.relative, slot: 0 },
  ] as const

  const xAxis = useAxis({ label: 'cosine similarity to the anchor', range: [-1, 1] })
  const yAxis = useAxis({ label: 'relative gradient weight', range: [0, 1.05] })
  return (
    <Figure
      title="Temperature decides which negatives matter"
      state={state}
      caption="Each dot is one of 256 negatives, placed at its cosine similarity to the anchor. Its height is its gradient weight in the InfoNCE loss, relative to the hardest negative. At high temperature every negative is pushed away about equally. At low temperature almost all the weight falls on the few most similar negatives. Drag the vertical line to move the positive's similarity."

      readouts={
        <>
          <Readout label="loss (nats)" value={formatNumber(r.loss)} />
          <Readout label="softmax weight on the positive" value={formatNumber(r.pPos)} />
          <Readout label="effective number of negatives" value={formatNumber(r.effective)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
        <Handle {...state.handle('sPos', { label: 'positive' })} />
      </Plot>
    </Figure>
  )
}
