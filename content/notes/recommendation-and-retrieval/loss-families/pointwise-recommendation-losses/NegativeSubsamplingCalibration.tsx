import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamChoice, ParamSlider, Readout, XYChart, type XYSeries } from '@/components/viz'
import { linspace, rng, sigmoid } from '@/lib/math'

const N = 20000
const SLOPE = 1
const GRID = linspace(-3, 3, 61)
const X_RANGE: [number, number] = [-3, 3]
const Y_RANGE: [number, number] = [1e-4, 1]

type Base = 'rare' | 'medium' | 'common'
const BASES: Record<Base, { label: string; intercept: number }> = {
  rare: { label: 'rare clicks', intercept: -5.5 },
  medium: { label: 'medium', intercept: -4 },
  common: { label: 'common', intercept: -2.5 },
}

type Impressions = { x: Float64Array; y: Uint8Array; u: Float64Array }

/** N impressions with one feature x ~ N(0, 1) and clicks y ~ Bernoulli(σ(a + x)); u decides which negatives are kept. */
function simulate(seed: number, intercept: number): Impressions {
  const r = rng(seed)
  const x = new Float64Array(N)
  const y = new Uint8Array(N)
  const u = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    x[n] = r.normal()
    y[n] = r.uniform() < sigmoid(intercept + SLOPE * x[n]) ? 1 : 0
    u[n] = r.uniform()
  }
  return { x, y, u }
}

/** Weighted one-feature logistic regression by Newton's method; returns [intercept, slope]. */
function fitLogistic(data: Impressions, keep: number, negativeWeight: number): [number, number] {
  let a = 0
  let b = 0
  for (let iter = 0; iter < 25; iter++) {
    let g0 = 0
    let g1 = 0
    let h00 = 1e-6
    let h01 = 0
    let h11 = 1e-6
    for (let n = 0; n < N; n++) {
      const positive = data.y[n] === 1
      if (!positive && data.u[n] >= keep) continue
      const w = positive ? 1 : negativeWeight
      const p = sigmoid(a + b * data.x[n])
      const r = w * (p - data.y[n])
      const c = w * p * (1 - p)
      g0 += r
      g1 += r * data.x[n]
      h00 += c
      h01 += c * data.x[n]
      h11 += c * data.x[n] * data.x[n]
    }
    const det = h00 * h11 - h01 * h01
    const da = (h11 * g0 - h01 * g1) / det
    const db = (h00 * g1 - h01 * g0) / det
    a -= da
    b -= db
    if (Math.abs(da) + Math.abs(db) < 1e-9) break
  }
  return [a, b]
}

/** He et al.'s recalibration: maps a probability learnt with negatives kept at rate w back to the full-data scale. */
const recalibrate = (q: number, w: number) => q / (q + (1 - q) / w)

const pct = (v: number) => `${(100 * v).toFixed(2)}%`

/**
 * Negative subsampling and calibration. Clicks are simulated from a known logistic model; a logistic regression is fit
 * to all clicks and a fraction w of non-clicks, then corrected analytically or refit with importance weights 1/w.
 */
export function NegativeSubsamplingCalibration() {
  const [keep, setKeep] = useState(0.05)
  const [base, setBase] = useState<Base>('medium')
  const [seed, setSeed] = useState(7)
  const intercept = BASES[base].intercept

  const data = useMemo(() => simulate(seed, intercept), [seed, intercept])
  const raw = useMemo(() => fitLogistic(data, keep, 1), [data, keep])
  const weighted = useMemo(() => fitLogistic(data, keep, 1 / keep), [data, keep])

  const stats = useMemo(() => {
    let clicks = 0
    let kept = 0
    let sumRaw = 0
    let sumCorrected = 0
    let sumWeighted = 0
    for (let n = 0; n < N; n++) {
      clicks += data.y[n]
      if (data.y[n] === 1 || data.u[n] < keep) kept++
      const q = sigmoid(raw[0] + raw[1] * data.x[n])
      sumRaw += q
      sumCorrected += recalibrate(q, keep)
      sumWeighted += sigmoid(weighted[0] + weighted[1] * data.x[n])
    }
    return { clicks, kept, raw: sumRaw / N, corrected: sumCorrected / N, weighted: sumWeighted / N }
  }, [data, keep, raw, weighted])

  const series = useMemo((): XYSeries[] => {
    const rawCurve = GRID.map((x) => sigmoid(raw[0] + raw[1] * x))
    return [
      {
        name: 'true click probability',
        type: 'line',
        x: GRID,
        y: GRID.map((x) => sigmoid(intercept + SLOPE * x)),
        emphasis: true,
      },
      { name: 'fit on subsampled data', type: 'line', x: GRID, y: rawCurve, slot: 0 },
      {
        name: 'recalibrated q / (q + (1 − q)/w)',
        type: 'line',
        x: GRID,
        y: rawCurve.map((q) => recalibrate(q, keep)),
        slot: 1,
      },
      {
        name: 'refit with negatives weighted 1/w',
        type: 'line',
        x: GRID,
        y: GRID.map((x) => sigmoid(weighted[0] + weighted[1] * x)),
        slot: 2,
        dashed: true,
      },
    ]
  }, [raw, weighted, intercept, keep])

  return (
    <Interactive
      title="Negative subsampling inflates predicted click-through rates"
      caption="Twenty thousand simulated impressions with one feature x and a known logistic click model (black). A logistic regression is fit to every click and a fraction w of the non-clicks. Its predictions are too high by a factor of about 1/w in the odds. The recalibration formula maps them back onto the true curve without refitting. Refitting with each kept non-click weighted by 1/w also recovers the curve, but its estimate is noisier at small w. The y-axis is logarithmic."
      controls={
        <>
          <ParamSlider
            label="fraction of non-clicks kept, w"
            value={keep}
            onChange={setKeep}
            min={0.01}
            max={1}
            step={0.01}
          />
          <ParamChoice
            label="base rate"
            value={base}
            onChange={setBase}
            options={(Object.keys(BASES) as Base[]).map((b) => ({ value: b, label: BASES[b].label }))}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="observed CTR" value={pct(stats.clicks / N)} />
          <Readout label="mean prediction, subsampled fit" value={pct(stats.raw)} />
          <Readout label="recalibrated" value={pct(stats.corrected)} />
          <Readout label="importance-weighted" value={pct(stats.weighted)} />
          <Readout label="training rows" value={`${stats.kept} of ${N}`} />
          <Readout
            label="intercept shift"
            value={`${(raw[0] - weighted[0]).toFixed(2)} (−ln w = ${(-Math.log(keep)).toFixed(2)})`}
          />
        </>
      }
    >
      <XYChart
        height={320}
        series={series}
        xRange={X_RANGE}
        yRange={Y_RANGE}
        yLog
        xLabel="feature x"
        yLabel="P(click | x)"
      />
    </Interactive>
  )
}
