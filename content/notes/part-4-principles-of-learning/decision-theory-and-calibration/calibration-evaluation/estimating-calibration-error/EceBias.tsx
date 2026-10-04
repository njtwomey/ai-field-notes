import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn/foundation/random'

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const DATASETS = 200
const HIST_BINS = 30

type Estimator = 'l1' | 'l2' | 'l2debiased'

/**
 * Sampling distribution of binned calibration-error estimators. Each simulated test set has n cases with true
 * probability σ(z), z ~ N(0, 1.5²), a label drawn from it, and a model prediction σ(k z). The estimators are computed
 * on 200 such test sets; the dashed line is the population value, from one very large sample.
 */
export function EceBias() {
  const state = useFigureState({
    est: choice<Estimator>(
      [
        { value: 'l1', label: 'ECE (L1)' },
        { value: 'l2', label: 'squared (L2)' },
        { value: 'l2debiased', label: 'squared, debiased' },
      ],
      'l1',
      { label: 'estimator' },
    ),
    k: float(1, { min: 0.5, max: 2, step: 0.05, label: 'model sharpness k' }),
    n: int(500, { min: 100, max: 3000, step: 100, label: 'cases n', format: (v) => String(v) }),
    bins: int(15, { min: 2, max: 30, step: 1, label: 'bins', format: (v) => String(v) }),
  })

  // Population calibration error of the model σ(kz) against the truth σ(z), from a large fixed sample.
  const truth = useMemo(() => {
    const g = stream(123)
    let l1 = 0
    let l2 = 0
    const N = 40000
    for (let i = 0; i < N; i++) {
      const z = 1.5 * normal(g)
      const gap = sigmoid(z) - sigmoid(state.k * z)
      l1 += Math.abs(gap)
      l2 += gap * gap
    }
    return { l1: l1 / N, l2: l2 / N }
  }, [state.k])

  const values = useMemo(() => {
    const g = stream(5)
    const m = state.bins
    const out: number[] = []
    for (let d = 0; d < DATASETS; d++) {
      const sumP = new Float64Array(m)
      const sumY = new Float64Array(m)
      const cnt = new Float64Array(m)
      for (let i = 0; i < state.n; i++) {
        const z = 1.5 * normal(g)
        const y = uniform(g) < sigmoid(z) ? 1 : 0
        const p = sigmoid(state.k * z)
        const b = Math.min(m - 1, Math.floor(p * m))
        sumP[b] += p
        sumY[b] += y
        cnt[b]++
      }
      let e = 0
      for (let b = 0; b < m; b++) {
        if (!cnt[b]) continue
        const w = cnt[b] / state.n
        const gap = sumY[b] / cnt[b] - sumP[b] / cnt[b]
        if (state.est === 'l1') e += w * Math.abs(gap)
        else {
          e += w * gap * gap
          if (state.est === 'l2debiased' && cnt[b] > 1) {
            const f = sumY[b] / cnt[b]
            e -= (w * f * (1 - f)) / (cnt[b] - 1)
          }
        }
      }
      out.push(e)
    }
    return out
  }, [state.k, state.n, state.bins, state.est])

  const target = state.est === 'l1' ? truth.l1 : truth.l2
  const lo = Math.min(0, ...values)
  const hi = Math.max(...values, target) * 1.05 || 1e-3
  const width = (hi - lo) / HIST_BINS
  const counts = new Array<number>(HIST_BINS).fill(0)
  values.forEach((v) => counts[Math.min(HIST_BINS - 1, Math.floor((v - lo) / width))]++)
  const centres = counts.map((_, i) => lo + (i + 0.5) * width)
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const top = Math.max(...counts) * 1.1

  const xAxis = useAxis({ label: 'estimated calibration error', range: [lo, hi] })
  const yAxis = useAxis({ label: 'test sets', range: [0, top] })
  return (
    <Figure
      title="Binned calibration error is biased upwards"
      state={state}
      caption="Histogram of the binned estimate over 200 simulated test sets of n cases each. The dashed line is the population calibration error of the model. At sharpness k = 1 the model is perfectly calibrated, so the true value is 0, yet every L1 estimate is positive; more bins and fewer cases make it worse. The debiased squared estimator subtracts each bin's estimated sampling variance and is centred near the truth."

      readouts={
        <>
          <Readout label="population value" value={formatNumber(target)} />
          <Readout label="mean estimate" value={formatNumber(mean)} />
          <Readout label="bias" value={formatNumber(mean - target)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars name="estimates" x={centres} y={counts} slot={0} />
        <Curve name="population value" x={[target, target]} y={[0, top]} emphasis dashed />
      </Plot>
    </Figure>
  )
}
