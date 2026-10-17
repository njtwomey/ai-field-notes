import { useMemo, useState } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { syntheticBeat, syntheticEcg } from 'aifn-methods/data/signals'
import { conv, pad } from 'aifn-compute/foundation/convolution'
import {
  add,
  fromData,
  mul,
  reshape,
  square,
  sub,
  sum,
  toArray,
  toFlat,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { normals, stream } from 'aifn-compute/foundation/random'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { trace } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { FixedHeight } from '../_shared/FixedHeight'

/** Kernel taps; the kernel reads 22 samples before the output position and 41 after, so tap 22 sits on the R peak. */
const K = 64
const OFFSET = 22
/** Samples in the training and test traces (about 30 and 40 beats at 60 Hz). */
const N_TRAIN = 1800
const N_TEST = 2400
/** Samples of the test trace drawn. */
const SHOWN = 600
/** Width of the moving average subtracted to remove baseline wander. */
const DETREND = 121
/** Width (standard deviation, samples) of the soft target around each R peak. */
const TARGET_WIDTH = 1.5
/** A detected peak counts as a hit within this many samples of an R peak of its kind. */
const TOLERANCE = 3
const KINDS = ['normal', 'ectopic'] as const

type Event = { t: number; kind: 0 | 1 }
type Trace = { x: Float64Array; events: Event[] }

/** The padding that makes output t read inputs t − OFFSET … t + K − 1 − OFFSET. */
const WINDOW = { padding: [[OFFSET, K - 1 - OFFSET]] as [number, number][], flip: false }

/**
 * A synthetic ECG with its baseline wander removed (a moving average over DETREND samples subtracted) and the R peak
 * of every complete beat: the beat's start plus the position of the peak in the beat shape of its length and kind.
 */
function ecgTrace(key: string, n: number, noise: number): Trace {
  const ecg = syntheticEcg(stream(key), { n, ectopic: 0.3, noise, wander: 0.2 })
  const y = reshape(ecg.y, [1, 1, n])
  const half = (DETREND - 1) / 2
  const box = fromData(new Float64Array(DETREND).fill(1 / DETREND), [1, 1, DETREND])
  const trend = conv(
    pad(
      y,
      [
        [0, 0],
        [0, 0],
        [half, half],
      ],
      'edge',
    ),
    box,
    { flip: false },
  )
  const x = Float64Array.from(toFlat(sub(y, trend) as Tensor))
  const events: Event[] = []
  for (let i = 0; i + 1 < ecg.beats.length; i++) {
    const b = ecg.beats[i]
    const shape = syntheticBeat(ecg.beats[i + 1].start - b.start, b.ectopic)
    const t = b.start + shape.indexOf(Math.max(...shape))
    if (t < n) events.push({ t, kind: b.ectopic ? 1 : 0 })
  }
  return { x, events }
}

/** Two soft target channels: a Gaussian bump at each R peak, in the channel of the beat's kind. */
function targets(tr: Trace): Tensor {
  const n = tr.x.length
  const y = new Float64Array(2 * n)
  for (const e of tr.events) {
    for (let t = Math.max(0, e.t - 8); t < Math.min(n, e.t + 9); t++) {
      const v = Math.exp(-((t - e.t) ** 2) / (2 * TARGET_WIDTH ** 2))
      y[e.kind * n + t] = Math.max(y[e.kind * n + t], v)
    }
  }
  return fromData(y, [1, 2, n])
}

/** The beat shape of a 60-sample beat of each kind, R peak on tap OFFSET, mean removed and scaled to unit norm. */
const TEMPLATES: number[][] = [false, true].map((ectopic) => {
  const b = syntheticBeat(60, ectopic)
  const r = b.indexOf(Math.max(...b))
  const w = Array.from({ length: K }, (_, u) => b[u - OFFSET + r] ?? 0)
  const m = w.reduce((a, v) => a + v, 0) / K
  const z = w.map((v) => v - m)
  const norm = Math.hypot(...z)
  return z.map((v) => v / norm)
})

type Params = { w: Tensor; b: Tensor }
const logits = (p: Params, x: Tensor) => add(conv(x, p.w, WINDOW), reshape(p.b, [1, 2, 1]))

/** Peaks above half the channel's largest response, matched to R peaks of the channel's kind. */
function detections(r: ArrayLike<number>, events: Event[], kind: 0 | 1) {
  let top = -Infinity
  for (let t = 0; t < r.length; t++) top = Math.max(top, r[t])
  const wanted = events.filter((e) => e.kind === kind).map((e) => e.t)
  const found = new Set<number>()
  let falseAlarms = 0
  for (let t = 1; t + 1 < r.length; t++) {
    if (r[t] <= 0.5 * top || r[t] < r[t - 1] || r[t] <= r[t + 1]) continue
    const hit = wanted.find((s) => Math.abs(s - t) <= TOLERANCE)
    if (hit === undefined) falseAlarms++
    else found.add(hit)
  }
  return { found: found.size, of: wanted.length, falseAlarms }
}

const cosine = (a: readonly number[], b: readonly number[]) => {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0)
  return dot / (Math.hypot(...a) * Math.hypot(...b))
}

/**
 * A one-layer one-dimensional convolutional network on an electrocardiogram: two kernels of 64 taps, a bias each and
 * a sigmoid, trained to put a peak on every R peak of its kind of beat. The trained kernels are drawn against the
 * matched filters, the beat shapes themselves.
 */
export function EcgFilterBank() {
  const state = useFigureState({
    kernels: choice(
      [
        { value: 'trained', label: 'trained kernels' },
        { value: 'matched', label: 'matched filters (the beat shapes)' },
      ],
      'trained',
      { label: 'filter bank' },
    ),
    noise: float(0.08, { ge: 0, le: 0.4, suggestions: [0, 0.08, 0.2, 0.3], label: 'noise σ' }),
    steps: int(300, { ge: 10, le: 2000, suggestions: [50, 150, 300, 600], label: 'training steps' }),
    rate: float(0.02, { ge: 0.001, le: 0.2, scale: 'log10', suggestions: [0.005, 0.02, 0.05], label: 'learning rate' }),
    decay: float(0.0001, { ge: 0, le: 0.1, suggestions: [0, 0.0001, 0.001, 0.01], label: 'weight decay λ' }),
  })
  const { kernels, noise, steps, rate, decay } = state

  const data = useMemo(() => {
    const train = ecgTrace('one-dimensional-cnn/train', N_TRAIN, noise)
    const test = ecgTrace('one-dimensional-cnn/test', N_TEST, noise)
    return {
      train,
      test,
      xTrain: fromData(train.x, [1, 1, N_TRAIN]),
      yTrain: targets(train),
      xTest: fromData(test.x, [1, 1, N_TEST]),
    }
  }, [noise])

  const every = Math.max(1, Math.round(steps / 30))
  const run = useComputed(
    () => {
      const loss = (p: Params, batch: { x: Tensor; y: Tensor }) =>
        add(binaryCrossEntropyWithLogits(logits(p, batch.x), batch.y), mul(decay, sum(square(p.w))))
      const start: Params = {
        w: normals(stream('one-dimensional-cnn/init'), [2, 1, K], 0, 0.05) as Tensor,
        b: fromData(Float64Array.of(-3, -3), [2]),
      }
      const tr = trace(
        trainingLoop({ loss, data: { x: data.xTrain, y: data.yTrain }, optimizer: adamRule({ stepSize: rate }) }),
        { params: start },
        steps,
        { every },
      )
      return tr.steps.map((s) => ({ t: s.t, loss: s.loss, params: s.params as Params }))
    },
    [data, steps, rate, decay, every],
    { mode: 'release' },
  )
  const states = run.value ?? []

  const key = `${noise}|${steps}|${rate}|${decay}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, states.length - 1) : 0
  const current = states[Math.max(step, 0)] as (typeof states)[number] | undefined

  const view = useMemo(() => {
    const trained = kernels === 'trained' && current !== undefined
    const w = trained ? (toArray(current.params.w) as number[][][]).map((k) => k[0]) : TEMPLATES
    let channels: number[][]
    if (trained) {
      const z = toArray(logits(current.params, data.xTest) as Tensor) as number[][][]
      channels = z[0].map((row) => row.map((v) => 1 / (1 + Math.exp(-v))))
    } else {
      const wt = fromData(Float64Array.from(TEMPLATES.flat()), [2, 1, K])
      channels = (toArray(conv(data.xTest, wt, WINDOW) as Tensor) as number[][][])[0]
    }
    // Each channel divided by its largest value on the test trace, so a peak above 0.5 is a detection.
    const scaled = channels.map((row) => {
      const top = Math.max(...row)
      return row.map((v) => v / top)
    })
    const found = [0, 1].map((c) => detections(channels[c], data.test.events, c as 0 | 1))
    return { w, scaled, found, cos: [0, 1].map((c) => cosine(w[c], TEMPLATES[c])) }
  }, [kernels, current, data])

  const time = useMemo(() => Array.from({ length: SHOWN }, (_, t) => t), [])
  const taps = useMemo(() => Array.from({ length: K }, (_, u) => u - OFFSET), [])
  const shownEvents = data.test.events.filter((e) => e.t < SHOWN)
  const amplitude = (e: Event) => data.test.x[e.t]
  // The template drawn at the kernel's norm, so the two are comparable by shape.
  const reference = (c: number) => {
    const norm = Math.hypot(...view.w[c])
    return TEMPLATES[c].map((v) => v * norm)
  }

  const tx = useAxis({ label: 'sample (60 per second)', range: [0, SHOWN - 1] })
  const ty = useAxis({ label: 'ECG (wander removed)', key: noise })
  const oy = useAxis({ label: 'response ÷ its maximum', range: [-0.4, 1.1] })
  const kx = useAxis({ label: 'tap, relative to the R peak', range: [-OFFSET - 1, K - OFFSET] })
  const ky0 = useAxis({ label: 'normal-beat kernel', key: `${kernels}|${key}` })
  const ky1 = useAxis({ label: 'ectopic-beat kernel', key: `${kernels}|${key}` })

  return (
    <Figure
      title="A learned filter bank on an electrocardiogram"
      defaultSize="XL"
      state={state}
      controls={
        <Player
          value={Math.max(step, 0)}
          onChange={(s) => setPos({ key, step: s })}
          count={Math.max(states.length, 1)}
          label="training"
          format={(s) => `step ${states[s]?.t ?? 0} of ${steps}`}
        />
      }
      caption={`A one-dimensional convolutional layer with two kernels of ${K} taps, a bias each and a sigmoid, trained by Adam to put a peak on the R spike of every normal beat (blue channel) and every ectopic beat (orange channel) of a ${N_TRAIN}-sample synthetic electrocardiogram (60 samples per second, 30% ectopic beats). Top: the first ${SHOWN} samples of a separate test trace, with its R peaks marked by kind. Middle: the two output channels, each divided by its largest value on the test trace; a peak above the dashed line at 0.5 is a detection. Bottom: the two kernels (solid) against the matched filters, the beat shapes themselves (dashed, scaled to the kernel's norm). Play the training (it opens at step 0): the kernels start as noise, and the first steps already give them the shape of their beats. Switch the filter bank to the matched filters: the normal-beat filter also fires on every ectopic beat, which shares its large deflection. Raise the weight decay λ to 0.01 and the trained kernels become the matched filters (cosine near 0.9 and 1), false alarms included; at λ = 0.0001 they keep the beat's main deflections, differ elsewhere, and fire on far fewer ectopic beats.`}
      readouts={
        <>
          <Readout
            label="normal beats found"
            value={`${view.found[0].found} of ${view.found[0].of}, ${view.found[0].falseAlarms} false alarms`}
          />
          <Readout
            label="ectopic beats found"
            value={`${view.found[1].found} of ${view.found[1].of}, ${view.found[1].falseAlarms} false alarms`}
          />
          <Readout
            label="cosine of each kernel with its beat shape"
            value={`${formatNumber(view.cos[0])}, ${formatNumber(view.cos[1])}`}
          />
          <Readout label="training loss" value={kernels === 'trained' && current ? formatNumber(current.loss) : '–'} />
          <Readout label="weights" value={`2 × ${K} + 2 biases = ${2 * K + 2}`} />
        </>
      }
    >
      <FixedHeight height={170}>
        <Plot x={tx} y={ty} ariaLabel="Test electrocardiogram with R peaks marked">
          <Curve name="ECG" x={time} y={Array.from(data.test.x.subarray(0, SHOWN))} muted />
          {KINDS.map((name, c) => {
            const ev = shownEvents.filter((e) => e.kind === c)
            return ev.length ? (
              <Points
                key={name}
                name={`${name} R peak`}
                x={ev.map((e) => e.t)}
                y={ev.map(amplitude)}
                slot={c}
                size={6}
              />
            ) : null
          })}
        </Plot>
      </FixedHeight>
      <FixedHeight height={150}>
        <Plot x={tx} y={oy} ariaLabel="Output channels of the filter bank">
          {KINDS.map((name, c) => (
            <Curve key={name} name={`${name} channel`} x={time} y={view.scaled[c].slice(0, SHOWN)} slot={c} />
          ))}
          <Annotation y={0.5} text="detection threshold" dashed />
        </Plot>
      </FixedHeight>
      <FixedHeight height={170}>
        <div className="grid gap-4 md:grid-cols-2">
          {[ky0, ky1].map((ky, c) => (
            <Plot key={c} x={kx} y={ky} ariaLabel={`${KINDS[c]}-beat kernel against its beat shape`}>
              <Curve name="kernel" x={taps} y={view.w[c]} slot={c} />
              <Curve name="beat shape" x={taps} y={reference(c)} slot={c} dashed />
            </Plot>
          ))}
        </div>
      </FixedHeight>
    </Figure>
  )
}
