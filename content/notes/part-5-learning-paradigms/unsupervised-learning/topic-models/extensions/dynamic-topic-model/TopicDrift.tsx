import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const WORDS = ['valve', 'tube', 'circuit', 'transistor', 'chip', 'software', 'network', 'data'] as const
type Word = (typeof WORDS)[number]
const V = WORDS.length
const T = 12
const SLICES = Array.from({ length: T }, (_, t) => t + 1)

const softmax = (b: number[]) => {
  const m = Math.max(...b)
  const e = b.map((x) => Math.exp(x - m))
  const s = e.reduce((a, c) => a + c, 0)
  return e.map((x) => x / s)
}

/** Natural parameters of one "computing" topic over 12 slices: smooth trends plus a small seeded random walk. */
function trueTopic(): number[][] {
  const r = stream(7)
  const walk = new Array<number>(V).fill(0)
  return SLICES.map((t) => {
    for (let w = 0; w < V; w++) walk[w] += 0.08 * normal(r)
    const trend = [
      1.6 - 0.3 * t,
      2 - 0.35 * t,
      1,
      0.2 + 1.8 * Math.exp(-((t - 4) ** 2) / 6),
      -0.5 + 2 / (1 + Math.exp(-(t - 7))),
      -1 + 0.25 * t,
      -1.5 + 0.3 * t,
      0.3,
    ]
    return trend.map((b, w) => b + walk[w])
  })
}

/** Word counts per slice: N tokens drawn from the slice's topic. */
function sampleCounts(p: number[][], n: number, seed: number): number[][] {
  const r = stream(seed)
  return p.map((pt) => {
    const c = new Array<number>(V).fill(0)
    for (let i = 0; i < n; i++) {
      let u = uniform(r)
      let w = 0
      while (w < V - 1 && (u -= pt[w]) > 0) w++
      c[w]++
    }
    return c
  })
}

/**
 * Kalman filter and RTS smoother for one word's natural parameter: a random walk with drift variance s2, observed
 * through a pseudo-observation log((n + ½)/(N + V/2)) with variance 1/(n + ½), the delta-method variance of a log count.
 */
function smoothWord(obs: number[], obsVar: number[], s2: number): number[] {
  const m: number[] = []
  const P: number[] = []
  let mPrev = obs[0]
  let pPrev = 10
  for (let t = 0; t < obs.length; t++) {
    const pred = t === 0 ? pPrev : pPrev + s2
    const gain = pred / (pred + obsVar[t])
    const mt = mPrev + gain * (obs[t] - mPrev)
    const pt = (1 - gain) * pred
    m.push(mt)
    P.push(pt)
    mPrev = mt
    pPrev = pt
  }
  const out = [...m]
  for (let t = obs.length - 2; t >= 0; t--) {
    const j = P[t] / (P[t] + s2)
    out[t] = m[t] + j * (out[t + 1] - m[t])
  }
  return out
}

export function TopicDrift() {
  const state = useFigureState({
    slice: slider(1, T, 4, { step: 1, label: 'time slice t' }),
    tokens: int(60, { min: 10, max: 600, step: 10, label: 'tokens per slice N' }),
    drift: float(0.3, { min: 0.02, max: 2, step: 0.02, label: 'drift standard deviation σ' }),
    word: choice<Word>(
      WORDS.map((w) => ({ value: w, label: w })),
      'transistor',
      { label: 'word' },
    ),
  })
  const wi = WORDS.indexOf(state.word)

  const truth = useMemo(() => trueTopic().map(softmax), [])
  const counts = useMemo(() => sampleCounts(truth, state.tokens, 3), [truth, state.tokens])
  const raw = useMemo(() => counts.map((c) => c.map((n) => n / state.tokens)), [counts, state.tokens])
  const smoothed = useMemo(() => {
    const perWord = WORDS.map((_, w) =>
      smoothWord(
        counts.map((c) => Math.log((c[w] + 0.5) / (state.tokens + V / 2))),
        counts.map((c) => 1 / (c[w] + 0.5)),
        state.drift * state.drift,
      ),
    )
    return SLICES.map((_, t) => softmax(perWord.map((b) => b[t])))
  }, [counts, state.tokens, state.drift])
  const pooled = useMemo(() => {
    const tot = WORDS.map((_, w) => counts.reduce((s, c) => s + c[w], 0))
    return tot.map((n) => n / (state.tokens * T))
  }, [counts, state.tokens])

  const mae = (est: (t: number, w: number) => number) => {
    let e = 0
    for (let t = 0; t < T; t++) for (let w = 0; w < V; w++) e += Math.abs(est(t, w) - truth[t][w])
    return e / (T * V)
  }
  const t0 = state.slice - 1
  const top = (p: number[]) =>
    p
      .map((v, w) => [v, w] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 3)
      .map(([, w]) => WORDS[w])
      .join(', ')

  const xAxis = useAxis({ label: 'time slice t', range: [1, T] })
  const yAxis = useAxis({ label: 'p(word | topic, t)', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'time slice t', range: [1, T] })
  const yAxis2 = useAxis({ label: `p(${state.word} | topic, t)`, range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="A topic drifting across time slices"
      state={state}
      caption="One topic over 12 time slices. Left: the true word probabilities, softmax of natural parameters that follow a random walk. Right, for one word: the per-slice frequency among N tokens (points), the Kalman-smoothed estimate that the dynamic topic model's variational update computes (dashed), and the single static topic that LDA would fit to the pooled corpus (grey). With few tokens per slice the raw frequencies are noisy and smoothing borrows strength from neighbouring slices. A very small drift variance flattens the topic towards the static fit; a very large one reproduces the raw frequencies. Drag the slice line on either chart, or step it with the arrows."

      readouts={
        <>
          <Readout label={`top words at t = ${state.slice} (true)`} value={top(truth[t0])} />
          <Readout label="top words (smoothed)" value={top(smoothed[t0])} />
          <Readout label="mean abs. error, raw" value={formatNumber(mae((t, w) => raw[t][w]))} />
          <Readout label="smoothed" value={formatNumber(mae((t, w) => smoothed[t][w]))} />
          <Readout label="static" value={formatNumber(mae((_, w) => pooled[w]))} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(
            WORDS.map((w, i) => ({
              name: w,
              type: 'line' as const,
              x: SLICES,
              y: truth.map((p) => p[i]),
              slot: i,
            })),
          )}
          <Handle kind={'x' as const} at={state.slice} onDrag={(v: number) => state.set('slice', v)} label="slice" />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve name="true" x={SLICES} y={truth.map((p) => p[wi])} emphasis />
          <Points name="per-slice frequency" x={SLICES} y={raw.map((p) => p[wi])} slot={0} />
          <Curve name="smoothed (DTM)" x={SLICES} y={smoothed.map((p) => p[wi])} slot={1} dashed />
          <Curve name="static (LDA)" x={SLICES} y={SLICES.map(() => pooled[wi])} muted />
          <Handle kind={'x' as const} at={state.slice} onDrag={(v: number) => state.set('slice', v)} label="slice" />
        </Plot>
      </div>
    </Figure>
  )
}
