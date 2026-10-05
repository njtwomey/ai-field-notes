import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { sigmoid } from 'aifn-compute/numerics/special'

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
  const state = useFigureState({
    pooling: choice<Pooling>(POOLINGS, 'conjunctive', { label: 'pooling' }),
    amplitude: float(1.5, { min: 0.4, max: 3, step: 0.1, label: 'motif amplitude' }),
    noise: float(0.4, { min: 0.1, max: 1, step: 0.05, label: 'noise σ' }),
    pos: float(60, { min: 0, max: T - MOTIF, step: 1, label: 'motif start', format: (v) => String(v) }),
    seed: int(3, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
    motif: setting(true, 'inject motif'),
    distractor: setting(true, 'noise burst'),
  })

  const p = state.pos
  const a = state.amplitude
  const sigma = state.noise
  const s = state.seed

  const r = useMemo(() => {
    const g = stream(s)
    const x = Array.from({ length: T }, (_, t) => 0.8 * Math.sin((2 * Math.PI * t) / 60) + sigma * normal(g))
    const burst = Array.from({ length: MOTIF }, () => normal(g))
    if (state.motif) for (let u = 0; u < MOTIF; u++) x[p + u] += a * Math.sin((2 * Math.PI * u) / PERIOD)
    // The distractor is a burst of noise of similar energy, half a series away from the motif.
    const d = (p + T / 2) % (T - MOTIF)
    if (state.distractor) for (let u = 0; u < MOTIF; u++) x[d + u] += 0.9 * a * burst[u]

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
    const truth = Array.from({ length: T }, (_, t) => state.motif && t >= p && t < p + MOTIF)
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
  }, [p, a, sigma, s, state.motif, state.distractor])

  const t = Array.from({ length: T }, (_, i) => i)
  const motifIdx = Array.from({ length: MOTIF }, (_, u) => p + u)
  const burstWindow = Array.from({ length: MOTIF }, (_, u) => r.d + u)
  const top: SeriesSpec[] = [
    { name: 'time series', type: 'line', x: t, y: r.x, slot: 0 },
    ...(state.motif
      ? [{ name: 'motif', type: 'line' as const, x: motifIdx, y: motifIdx.map((i) => r.x[i]), slot: 1 }]
      : []),
    ...(state.distractor
      ? [{ name: 'noise burst', type: 'line' as const, x: burstWindow, y: burstWindow.map((i) => r.x[i]), slot: 2 }]
      : []),
  ]
  const score = r.series[state.pooling]
  const bottom = [
    {
      name: state.pooling === 'attention' ? 'attention weight' : 'motif-class score',
      x: t,
      y: score,
    },
    { name: 'zero', x: [0, T - 1], y: [0, 0], muted: true, dashed: true },
  ] as const
  const handles: Handle[] = state.motif
    ? [{ kind: 'x', at: p + MOTIF / 2, label: 'motif', onDrag: (v) => state.set('pos', Math.round(v - MOTIF / 2)) }]
    : []

  const xAxis = useAxis({ label: 'time point', range: X_RANGE })
  const yAxis = useAxis({ label: 'value', hold: 'union' })
  const xAxis2 = useAxis({ label: 'time point', range: X_RANGE })
  const yAxis2 = useAxis({ label: state.pooling === 'attention' ? 'attention a_t' : 'interpretation', hold: 'union' })
  return (
    <Figure
      title="What each MILLET pooling returns as an interpretation"
      state={state}
      caption="A series of 240 time points with a 24-point motif (two cycles of a period-12 wave) and, optionally, a burst of noise of similar energy. Drag the motif along the series. A fixed, untrained 'backbone' gives two features per time point: the amplitude of the motif's frequency in a 12-point window, and the window's standard deviation. The classifier head scores the motif class from the first feature; the attention head reads the second, so it fires on anything unusual. Instance returns the classifier's score at every time point, support above zero and refutation below; Additive and Conjunctive return that score weighted by attention, so quiet stretches drop to zero; Attention returns a class-agnostic weight, which also lights up the noise burst. NDCG@24 scores each interpretation's ranking against the true motif positions, as in the MILLET WebTraffic evaluation. The weights are hand-set, so the figure shows mechanics, not the paper's results."

      readouts={
        <>
          {POOLINGS.map((o) => (
            <Readout
              key={o.value}
              label={`NDCG@24 ${o.label}`}
              value={state.motif ? formatNumber(r.scores[o.value]) : '–'}
            />
          ))}
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <Plot x={xAxis} y={yAxis} height={220}>
          {seriesLayers(top)}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          <Area {...bottom[0]} />
          <Curve {...bottom[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
