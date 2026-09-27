import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { rng } from '@/lib/math'

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
  const k = useParam(1, { min: 0.5, max: 2, step: 0.05 })
  const n = useParam(500, { min: 100, max: 3000, step: 100 })
  const bins = useParam(15, { min: 2, max: 30, step: 1 })
  const [est, setEst] = useState<Estimator>('l1')

  // Population calibration error of the model σ(kz) against the truth σ(z), from a large fixed sample.
  const truth = useMemo(() => {
    const g = rng(123)
    let l1 = 0
    let l2 = 0
    const N = 40000
    for (let i = 0; i < N; i++) {
      const z = 1.5 * g.normal()
      const gap = sigmoid(z) - sigmoid(k.value * z)
      l1 += Math.abs(gap)
      l2 += gap * gap
    }
    return { l1: l1 / N, l2: l2 / N }
  }, [k.value])

  const values = useMemo(() => {
    const g = rng(5)
    const m = bins.value
    const out: number[] = []
    for (let d = 0; d < DATASETS; d++) {
      const sumP = new Float64Array(m)
      const sumY = new Float64Array(m)
      const cnt = new Float64Array(m)
      for (let i = 0; i < n.value; i++) {
        const z = 1.5 * g.normal()
        const y = g.uniform() < sigmoid(z) ? 1 : 0
        const p = sigmoid(k.value * z)
        const b = Math.min(m - 1, Math.floor(p * m))
        sumP[b] += p
        sumY[b] += y
        cnt[b]++
      }
      let e = 0
      for (let b = 0; b < m; b++) {
        if (!cnt[b]) continue
        const w = cnt[b] / n.value
        const gap = sumY[b] / cnt[b] - sumP[b] / cnt[b]
        if (est === 'l1') e += w * Math.abs(gap)
        else {
          e += w * gap * gap
          if (est === 'l2debiased' && cnt[b] > 1) {
            const f = sumY[b] / cnt[b]
            e -= (w * f * (1 - f)) / (cnt[b] - 1)
          }
        }
      }
      out.push(e)
    }
    return out
  }, [k.value, n.value, bins.value, est])

  const target = est === 'l1' ? truth.l1 : truth.l2
  const lo = Math.min(0, ...values)
  const hi = Math.max(...values, target) * 1.05 || 1e-3
  const width = (hi - lo) / HIST_BINS
  const counts = new Array<number>(HIST_BINS).fill(0)
  values.forEach((v) => counts[Math.min(HIST_BINS - 1, Math.floor((v - lo) / width))]++)
  const centres = counts.map((_, i) => lo + (i + 0.5) * width)
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const top = Math.max(...counts) * 1.1

  return (
    <Interactive
      title="Binned calibration error is biased upwards"
      caption="Histogram of the binned estimate over 200 simulated test sets of n cases each. The dashed line is the population calibration error of the model. At sharpness k = 1 the model is perfectly calibrated, so the true value is 0, yet every L1 estimate is positive; more bins and fewer cases make it worse. The debiased squared estimator subtracts each bin's estimated sampling variance and is centred near the truth."
      controls={
        <>
          <ParamChoice
            label="estimator"
            value={est}
            onChange={setEst}
            options={[
              { value: 'l1', label: 'ECE (L1)' },
              { value: 'l2', label: 'squared (L2)' },
              { value: 'l2debiased', label: 'squared, debiased' },
            ]}
          />
          <ParamSlider label="model sharpness k" param={k} />
          <ParamSlider label="cases n" param={n} format={(v) => String(v)} />
          <ParamSlider label="bins" param={bins} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="population value" value={formatNumber(target)} />
          <Readout label="mean estimate" value={formatNumber(mean)} />
          <Readout label="bias" value={formatNumber(mean - target)} />
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="estimated calibration error"
        yLabel="test sets"
        xRange={[lo, hi]}
        yRange={[0, top]}
        series={[
          { name: 'estimates', type: 'bar', x: centres, y: counts, slot: 0 },
          { name: 'population value', type: 'line', x: [target, target], y: [0, top], emphasis: true, dashed: true },
        ]}
      />
    </Interactive>
  )
}
