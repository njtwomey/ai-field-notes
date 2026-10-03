import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

const T = 1000
const NOISE = 0.5
const SMOOTH = 50
const ROUNDS = Array.from({ length: T }, (_, t) => t + 1)
const SLOTS = { none: 4, forget: 0, window: 1 }

type Drift = 'abrupt' | 'gradual' | 'incremental' | 'recurring'

/** The true slope w*_t. For gradual drift the concept of each example is drawn, so the slope is per example. */
function makeStream(drift: Drift, seed: number) {
  const r = rng(seed)
  const x: number[] = []
  const y: number[] = []
  const truth: number[] = []
  for (let t = 0; t < T; t++) {
    let w: number
    if (drift === 'abrupt') w = t < 500 ? 1 : -1
    else if (drift === 'incremental') w = t < 300 ? 1 : t > 700 ? -1 : 1 - (2 * (t - 300)) / 400
    else if (drift === 'recurring') w = Math.floor(t / 250) % 2 === 0 ? 1 : -1
    else {
      // Gradual: between rounds 350 and 650 the new concept appears with linearly rising probability.
      const pNew = Math.min(1, Math.max(0, (t - 350) / 300))
      w = r.uniform() < pNew ? -1 : 1
    }
    const xt = r.normal()
    x.push(xt)
    y.push(w * xt + NOISE * r.normal())
    truth.push(w)
  }
  return { x, y, truth }
}

type Fit = { w: number[]; err: number[] }

/** One-dimensional least squares with exponential forgetting λ (λ = 1: no forgetting). Predicts before updating. */
function forgetting(x: number[], y: number[], lambda: number): Fit {
  let a = 1e-3
  let b = 0
  const w: number[] = []
  const err: number[] = []
  for (let t = 0; t < T; t++) {
    const wt = b / a
    w.push(wt)
    err.push((y[t] - wt * x[t]) ** 2)
    a = lambda * a + x[t] * x[t]
    b = lambda * b + x[t] * y[t]
  }
  return { w, err }
}

/** Least squares on the last W examples only. */
function sliding(x: number[], y: number[], width: number): Fit {
  let a = 1e-3
  let b = 0
  const w: number[] = []
  const err: number[] = []
  for (let t = 0; t < T; t++) {
    const wt = b / a
    w.push(wt)
    err.push((y[t] - wt * x[t]) ** 2)
    a += x[t] * x[t]
    b += x[t] * y[t]
    if (t >= width) {
      a -= x[t - width] * x[t - width]
      b -= x[t - width] * y[t - width]
    }
  }
  return { w, err }
}

/** Trailing moving average, so the error curve shows how quickly each learner recovers. */
function smooth(v: number[]): number[] {
  let s = 0
  return v.map((e, i) => {
    s += e
    if (i >= SMOOTH) s -= v[i - SMOOTH]
    return s / Math.min(i + 1, SMOOTH)
  })
}

const mean = (v: number[]) => v.reduce((p, q) => p + q, 0) / v.length

/** A linear learner under four kinds of drift, with no forgetting, a forgetting factor, and a sliding window. */
export function DriftFigure() {
  const [drift, setDrift] = useState<Drift>('abrupt')
  const lambda = useParam(0.97, { min: 0.8, max: 0.999, step: 0.001 })
  const width = useParam(60, { min: 5, max: 500, step: 5 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const stream = useMemo(() => makeStream(drift, seed.value), [drift, seed.value])
  const none = useMemo(() => forgetting(stream.x, stream.y, 1), [stream])
  const forget = useMemo(() => forgetting(stream.x, stream.y, lambda.value), [stream, lambda.value])
  const windowed = useMemo(() => sliding(stream.x, stream.y, width.value), [stream, width.value])

  const estimates: XYSeries[] = [
    { name: 'true slope', type: 'line', x: ROUNDS, y: stream.truth, emphasis: true, dashed: true },
    { name: 'no forgetting', type: 'line', x: ROUNDS, y: none.w, slot: SLOTS.none },
    { name: `forgetting λ = ${lambda.value}`, type: 'line', x: ROUNDS, y: forget.w, slot: SLOTS.forget },
    { name: `window W = ${width.value}`, type: 'line', x: ROUNDS, y: windowed.w, slot: SLOTS.window },
  ]
  const errors: XYSeries[] = [
    { name: 'no forgetting', type: 'line', x: ROUNDS, y: smooth(none.err), slot: SLOTS.none },
    { name: 'forgetting', type: 'line', x: ROUNDS, y: smooth(forget.err), slot: SLOTS.forget },
    { name: 'window', type: 'line', x: ROUNDS, y: smooth(windowed.err), slot: SLOTS.window },
    { name: 'noise floor σ²', type: 'line', x: [1, T], y: [NOISE ** 2, NOISE ** 2], muted: true, dashed: true },
  ]

  return (
    <Interactive
      title="Learning a slope that drifts"
      caption="Each example is y = w*·x + noise, with x standard normal and noise standard deviation 0.5; the learner predicts before it sees y. The true slope w* flips from 1 to −1 abruptly, gradually (examples from the new concept become more frequent), incrementally (w* slides), or recurrently (every 250 rounds). Left: each learner's slope estimate. Right: squared prediction error, averaged over the last 50 rounds. Without forgetting the learner averages both concepts and settles near 0. A forgetting factor λ keeps about 1/(1−λ) examples, a window W keeps exactly W; short memories recover fast but are noisier on stationary stretches."
      controls={
        <>
          <ParamChoice
            label="drift"
            value={drift}
            onChange={setDrift}
            options={[
              { value: 'abrupt', label: 'abrupt' },
              { value: 'gradual', label: 'gradual' },
              { value: 'incremental', label: 'incremental' },
              { value: 'recurring', label: 'recurring' },
            ]}
          />
          <ParamSlider label="forgetting factor λ" param={lambda} format={(v) => v.toFixed(3)} />
          <ParamSlider label="window W" param={width} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="mean squared error: no forgetting" value={formatNumber(mean(none.err))} />
          <Readout label="forgetting" value={formatNumber(mean(forget.err))} />
          <Readout label="window" value={formatNumber(mean(windowed.err))} />
          <Readout label="effective memory 1/(1−λ)" value={formatNumber(1 / (1 - lambda.value))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={estimates} xLabel="round t" yLabel="slope estimate" xRange={[0, T]} yRange={[-1.6, 1.6]} />
        <XYChart series={errors} xLabel="round t" yLabel="squared error (moving average)" xRange={[0, T]} />
      </div>
    </Interactive>
  )
}
