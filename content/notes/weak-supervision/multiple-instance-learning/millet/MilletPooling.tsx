import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng, sigmoid } from '@/lib/math'

const T = 240
const MOTIF = 24
const PERIOD = 12
const WINDOW = 12
const X_RANGE: [number, number] = [0, T - 1]

type Pooling = 'instance' | 'attention' | 'additive' | 'conjunctive'

const POOLINGS: { value: Pooling; label: string }[] = [
  { value: 'instance', label: 'Instance' },
  { value: 'attention', label: 'Attention' },
  { value: 'additive', label: 'Additive' },
  { value: 'conjunctive', label: 'Conjunctive' },
]

/** NDCG@n of a ranking of time points against the known motif positions, with n the number of motif points. */
function ndcg(score: number[], truth: boolean[]): number {
  const n = truth.filter(Boolean).length
  if (n === 0) return NaN
  const order = score.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0])
  let dcg = 0
  let ideal = 0
  for (let j = 0; j < n; j++) {
    const discount = 1 / Math.log2(j + 2)
    ideal += discount
    if (truth[order[j][1]]) dcg += discount
  }
  return dcg / ideal
}

/**
 * A hand-built two-feature "backbone" and the four MILLET pooling heads applied to it. Nothing is trained: the weights
 * are fixed so that the mechanics of each pooling, and the kind of interpretation it returns, can be compared.
 */
export function MilletPooling() {
  const pos = useParam(60, { min: 0, max: T - MOTIF, step: 1 })
  const amplitude = useParam(1.5, { min: 0.4, max: 3, step: 0.1 })
  const noise = useParam(0.4, { min: 0.1, max: 1, step: 0.05 })
  const seed = useParam(3, { min: 1, max: 20, step: 1 })
  const [motif, setMotif] = useState(true)
  const [distractor, setDistractor] = useState(true)
  const [pooling, setPooling] = useState<Pooling>('conjunctive')

  const p = pos.value
  const a = amplitude.value
  const sigma = noise.value
  const s = seed.value

  const r = useMemo(() => {
    const g = rng(s)
    const x = Array.from({ length: T }, (_, t) => 0.8 * Math.sin((2 * Math.PI * t) / 60) + sigma * g.normal())
    const burst = Array.from({ length: MOTIF }, () => g.normal())
    if (motif) for (let u = 0; u < MOTIF; u++) x[p + u] += a * Math.sin((2 * Math.PI * u) / PERIOD)
    // The distractor is a burst of noise of similar energy, half a series away from the motif.
    const d = (p + T / 2) % (T - MOTIF)
    if (distractor) for (let u = 0; u < MOTIF; u++) x[d + u] += 0.9 * a * burst[u]

    // Features of the window centred on each time point, with replicate padding at the ends.
    const at = (i: number) => x[Math.min(T - 1, Math.max(0, i))]
    const amp: number[] = []
    const spread: number[] = []
    for (let t = 0; t < T; t++) {
      let sn = 0
      let cs = 0
      let m = 0
      let m2 = 0
      for (let u = 0; u < WINDOW; u++) {
        const v = at(t - WINDOW / 2 + u)
        sn += v * Math.sin((2 * Math.PI * u) / PERIOD)
        cs += v * Math.cos((2 * Math.PI * u) / PERIOD)
        m += v
        m2 += v * v
      }
      // z1: amplitude of the motif's frequency in the window (a phase-invariant motif detector).
      amp.push((2 / WINDOW) * Math.hypot(sn, cs))
      // z2: local standard deviation, a class-agnostic "something is happening here" feature.
      spread.push(Math.sqrt(Math.max(0, m2 / WINDOW - (m / WINDOW) ** 2)))
    }
    const sorted = [...spread].sort((u, v) => u - v)
    const median = sorted[Math.floor(T / 2)]
    const z2 = spread.map((v) => v / median)

    // Heads. Classifier: logit for the motif class from z1. Attention: sigmoid of z2.
    const attn = z2.map((v) => sigmoid(5 * (v - 1.3)))
    const instance = amp.map((v) => 4 * (v - 0.7))
    // Additive classifies attention-scaled embeddings; MILLET then weights those predictions by attention again.
    const additive = amp.map((v, t) => attn[t] * 4 * (attn[t] * v - 0.7))
    const conjunctive = instance.map((v, t) => attn[t] * v)
    const truth = Array.from({ length: T }, (_, t) => motif && t >= p && t < p + MOTIF)
    const series = { instance, attention: attn, additive, conjunctive }
    return {
      x,
      d,
      series,
      scores: {
        instance: ndcg(instance, truth),
        attention: ndcg(attn, truth),
        additive: ndcg(additive, truth),
        conjunctive: ndcg(conjunctive, truth),
      },
    }
  }, [p, a, sigma, s, motif, distractor])

  const t = Array.from({ length: T }, (_, i) => i)
  const motifIdx = Array.from({ length: MOTIF }, (_, u) => p + u)
  const burstWindow = Array.from({ length: MOTIF }, (_, u) => r.d + u)
  const top: XYSeries[] = [
    { name: 'time series', type: 'line', x: t, y: r.x, slot: 0 },
    ...(motif ? [{ name: 'motif', type: 'line' as const, x: motifIdx, y: motifIdx.map((i) => r.x[i]), slot: 1 }] : []),
    ...(distractor
      ? [{ name: 'noise burst', type: 'line' as const, x: burstWindow, y: burstWindow.map((i) => r.x[i]), slot: 2 }]
      : []),
  ]
  const score = r.series[pooling]
  const bottom: XYSeries[] = [
    {
      name: pooling === 'attention' ? 'attention weight' : 'motif-class score',
      type: 'line',
      x: t,
      y: score,
      area: true,
    },
    { name: 'zero', type: 'line', x: [0, T - 1], y: [0, 0], muted: true, dashed: true },
  ]
  const handles: Handle[] = motif
    ? [{ kind: 'x', at: p + MOTIF / 2, label: 'motif', onDrag: (v) => pos.set(Math.round(v - MOTIF / 2)) }]
    : []

  return (
    <Interactive
      title="What each MILLET pooling returns as an interpretation"
      caption="A series of 240 time points with a 24-point motif (two cycles of a period-12 wave) and, optionally, a burst of noise of similar energy. Drag the motif along the series. A fixed, untrained 'backbone' gives two features per time point: the amplitude of the motif's frequency in a 12-point window, and the window's standard deviation. The classifier head scores the motif class from the first feature; the attention head reads the second, so it fires on anything unusual. Instance returns the classifier's score at every time point, support above zero and refutation below; Additive and Conjunctive return that score weighted by attention, so quiet stretches drop to zero; Attention returns a class-agnostic weight, which also lights up the noise burst. NDCG@24 scores each interpretation's ranking against the true motif positions, as in the MILLET WebTraffic evaluation. The weights are hand-set, so the figure shows mechanics, not the paper's results."
      controls={
        <>
          <ParamChoice label="pooling" value={pooling} onChange={setPooling} options={POOLINGS} />
          <ParamSlider label="motif amplitude" param={amplitude} />
          <ParamSlider label="noise σ" param={noise} />
          <ParamSlider label="motif start" param={pos} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} />
          <div className="flex flex-col gap-3">
            <ParamSwitch label="inject motif" checked={motif} onChange={setMotif} />
            <ParamSwitch label="noise burst" checked={distractor} onChange={setDistractor} />
          </div>
        </>
      }
      readout={
        <>
          {POOLINGS.map((o) => (
            <Readout key={o.value} label={`NDCG@24 ${o.label}`} value={motif ? formatNumber(r.scores[o.value]) : '–'} />
          ))}
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <XYChart series={top} xLabel="time point" yLabel="value" xRange={X_RANGE} height={220} handles={handles} />
        <XYChart
          series={bottom}
          xLabel="time point"
          yLabel={pooling === 'attention' ? 'attention a_t' : 'interpretation'}
          xRange={X_RANGE}
          height={200}
        />
      </div>
    </Interactive>
  )
}
