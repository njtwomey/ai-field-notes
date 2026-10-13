import { useMemo } from 'react'
import { softThresholdSure } from 'aifn-compute/learning/selection'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { thresholding } from 'aifn-compute/optim/proximal'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  Curve,
  Figure,
  float,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const T_MAX = 4
const STEP = 0.05
const THRESHOLDS = Array.from({ length: Math.round(T_MAX / STEP) + 1 }, (_, i) => i * STEP)
const DRAWS = 50
const LOW = 0
const HIGH = 2

const clip = (v: number) => (Number.isFinite(v) ? Math.max(LOW, Math.min(HIGH, v)) : HIGH)
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—')
const argmin = (v: readonly number[]) => v.reduce((best, x, i) => (x < v[best] ? i : best), 0)

/** A sparse mean: the first k entries are ±signal, alternating in sign, and the rest are zero. */
const sparseMean = (n: number, k: number, signal: number) =>
  Float64Array.from({ length: n }, (_, i) => (i < k ? (i % 2 === 0 ? signal : -signal) : 0))

/** One observation y = μ + ε with unit noise, and the loss per coordinate of soft thresholding at every threshold. */
function draw(s: Stream, mu: Float64Array) {
  const eps = toFlat(normals(s, mu.length, 0, 1))
  const y = Float64Array.from(mu, (m, i) => m + eps[i])
  const loss = THRESHOLDS.map((t) => meanSquaredError(mu, thresholding(y, { rule: 'soft', lambda: t })))
  return { y, loss }
}

/**
 * SURE for soft thresholding a sparse Gaussian mean, per coordinate, against the threshold t: SURE of one sample, the
 * loss of the thresholded estimate on that sample, and the risk, the average loss over 50 noise draws.
 */
export function SureCurve() {
  const state = useFigureState({
    t: float(1, { min: 0, max: T_MAX, step: STEP, label: 'threshold t', format: (v) => v.toFixed(2) }),
    n: int(200, { min: 10, max: 5000, scale: 'log10', suggestions: [10, 50, 200, 1000, 5000], label: 'length n' }),
    sparsity: slider(0, 1, 0.1, { step: 0.05, label: 'fraction non-zero' }),
    signal: slider(0.5, 6, 3, { step: 0.25, label: 'size of non-zero means' }),
    seed: int(0, { ge: 0, label: 'seed' }),
  })
  const mu = useMemo(
    () => sparseMean(state.n, Math.round(state.sparsity * state.n), state.signal),
    [state.n, state.sparsity, state.signal],
  )
  const sample = useMemo(() => {
    const { y, loss } = draw(stream(`sure-curve/${state.seed}`), mu)
    const sure = Array.from(toFlat(softThresholdSure(y, THRESHOLDS, 1)), (v) => v / state.n)
    return { sure, loss }
  }, [mu, state.seed, state.n])
  const risk = useComputed(
    () => {
      const draws = Array.from({ length: DRAWS }, (_, r) => draw(child(stream('sure-curve/risk'), r), mu))
      return THRESHOLDS.map((_, j) => draws.reduce((a, d) => a + d.loss[j], 0) / DRAWS)
    },
    [mu],
    { mode: 'release' },
  )

  const at = Math.round(state.t / STEP)
  const sureBest = argmin(sample.sure)
  const riskBest = argmin(risk.value)
  const tAxis = useAxis({ label: 'threshold t', range: [0, T_MAX] })
  const lAxis = useAxis({ label: 'squared error per coordinate', range: [LOW, HIGH] })
  return (
    <Figure
      title="SURE against the risk it estimates"
      state={state}
      caption={`A mean vector of length n whose first entries are ±(size), alternating in sign, and the rest zero, observed with unit Gaussian noise and estimated by soft thresholding at t. Solid: SURE of one sample divided by n, computed from the observations alone. Thin: the loss of the thresholded estimate on that same sample, which needs the true mean. Dashed: the risk, the average loss over ${DRAWS} noise draws. SURE follows the risk on average; on one sample it scatters about it by an amount that shrinks like 1/√n. Shorten the vector to n = 10 to see SURE's minimiser wander, and lengthen it to see SURE settle on the risk curve. Drag the handle to move t. Values are clipped to [${LOW}, ${HIGH}].`}
      readouts={
        <>
          <Readout label="SURE / n at t" value={fmt(sample.sure[at])} />
          <Readout label="loss at t" value={fmt(sample.loss[at])} />
          <Readout label="risk at t" value={fmt(risk.value[at])} />
          <Readout label="t chosen by SURE" value={THRESHOLDS[sureBest].toFixed(2)} />
          <Readout label="t best on risk" value={THRESHOLDS[riskBest].toFixed(2)} />
          <Readout label="risk at SURE's t" value={fmt(risk.value[sureBest])} />
        </>
      }
    >
      <Plot x={tAxis} y={lAxis}>
        <Curve name="SURE / n (this sample)" slot={0} x={THRESHOLDS} y={sample.sure.map(clip)} />
        <Curve name="loss (this sample)" slot={1} muted x={THRESHOLDS} y={sample.loss.map(clip)} />
        <Curve name="risk" emphasis dashed x={THRESHOLDS} y={risk.value.map(clip)} stale={risk.stale} />
        <Handle {...state.handle('t', { label: 't' })} />
      </Plot>
    </Figure>
  )
}
