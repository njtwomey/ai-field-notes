import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'
import { sigmoid } from 'aifn/numerics/special'

const N = 20000
const SLOPE = 1
const GRID = toFlat(linspace(-3, 3, 61))
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
  const r = stream(seed)
  const x = new Float64Array(N)
  const y = new Uint8Array(N)
  const u = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    x[n] = normal(r)
    y[n] = uniform(r) < sigmoid(intercept + SLOPE * x[n]) ? 1 : 0
    u[n] = uniform(r)
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
  const state = useFigureState({
    keep: slider(0.01, 1, 0.05, { step: 0.01, label: 'fraction of non-clicks kept, w' }),
    base: choice<Base>(
      (Object.keys(BASES) as Base[]).map((b) => ({ value: b, label: BASES[b].label })),
      'medium',
      { label: 'base rate' },
    ),
    seed: int(7, { ge: 0, label: 'seed' }),
  })
  const intercept = BASES[state.base].intercept

  const data = useMemo(() => simulate(state.seed, intercept), [state.seed, intercept])
  const raw = useMemo(() => fitLogistic(data, state.keep, 1), [data, state.keep])
  const weighted = useMemo(() => fitLogistic(data, state.keep, 1 / state.keep), [data, state.keep])

  const stats = useMemo(() => {
    let clicks = 0
    let kept = 0
    let sumRaw = 0
    let sumCorrected = 0
    let sumWeighted = 0
    for (let n = 0; n < N; n++) {
      clicks += data.y[n]
      if (data.y[n] === 1 || data.u[n] < state.keep) kept++
      const q = sigmoid(raw[0] + raw[1] * data.x[n])
      sumRaw += q
      sumCorrected += recalibrate(q, state.keep)
      sumWeighted += sigmoid(weighted[0] + weighted[1] * data.x[n])
    }
    return { clicks, kept, raw: sumRaw / N, corrected: sumCorrected / N, weighted: sumWeighted / N }
  }, [data, state.keep, raw, weighted])

  const series = useMemo(() => {
    const rawCurve = GRID.map((x) => sigmoid(raw[0] + raw[1] * x))
    return [
      {
        name: 'true click probability',
        x: GRID,
        y: GRID.map((x) => sigmoid(intercept + SLOPE * x)),
        emphasis: true,
      },
      { name: 'fit on subsampled data', x: GRID, y: rawCurve, slot: 0 },
      {
        name: 'recalibrated q / (q + (1 − q)/w)',
        x: GRID,
        y: rawCurve.map((q) => recalibrate(q, state.keep)),
        slot: 1,
      },
      {
        name: 'refit with negatives weighted 1/w',
        x: GRID,
        y: GRID.map((x) => sigmoid(weighted[0] + weighted[1] * x)),
        slot: 2,
        dashed: true,
      },
    ] as const
  }, [raw, weighted, intercept, state.keep])

  const xAxis = useAxis({ label: 'feature x', range: X_RANGE })
  const yAxis = useAxis({ label: 'P(click | x)', range: Y_RANGE, log: true })
  return (
    <Figure
      title="Negative subsampling inflates predicted click-through rates"
      state={state}
      caption="Twenty thousand simulated impressions with one feature x and a known logistic click model (black). A logistic regression is fit to every click and a fraction w of the non-clicks. Its predictions are too high by a factor of about 1/w in the odds. The recalibration formula maps them back onto the true curve without refitting. Refitting with each kept non-click weighted by 1/w also recovers the curve, but its estimate is noisier at small w. The y-axis is logarithmic."

      readouts={
        <>
          <Readout label="observed CTR" value={pct(stats.clicks / N)} />
          <Readout label="mean prediction, subsampled fit" value={pct(stats.raw)} />
          <Readout label="recalibrated" value={pct(stats.corrected)} />
          <Readout label="importance-weighted" value={pct(stats.weighted)} />
          <Readout label="training rows" value={`${stats.kept} of ${N}`} />
          <Readout
            label="intercept shift"
            value={`${(raw[0] - weighted[0]).toFixed(2)} (−ln w = ${(-Math.log(state.keep)).toFixed(2)})`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
