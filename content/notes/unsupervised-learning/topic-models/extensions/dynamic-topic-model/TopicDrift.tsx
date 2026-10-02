import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, useParam } from 'aifn-render'
import { rng } from '@/lib/math'

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
  const r = rng(7)
  const walk = new Array<number>(V).fill(0)
  return SLICES.map((t) => {
    for (let w = 0; w < V; w++) walk[w] += 0.08 * r.normal()
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
  const r = rng(seed)
  return p.map((pt) => {
    const c = new Array<number>(V).fill(0)
    for (let i = 0; i < n; i++) {
      let u = r.uniform()
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
  const [word, setWord] = useState<Word>('transistor')
  const [tokens, setTokens] = useState(60)
  const [drift, setDrift] = useState(0.3)
  const slice = useParam(4, { min: 1, max: T, step: 1 })
  const wi = WORDS.indexOf(word)

  const truth = useMemo(() => trueTopic().map(softmax), [])
  const counts = useMemo(() => sampleCounts(truth, tokens, 3), [truth, tokens])
  const raw = useMemo(() => counts.map((c) => c.map((n) => n / tokens)), [counts, tokens])
  const smoothed = useMemo(() => {
    const perWord = WORDS.map((_, w) =>
      smoothWord(
        counts.map((c) => Math.log((c[w] + 0.5) / (tokens + V / 2))),
        counts.map((c) => 1 / (c[w] + 0.5)),
        drift * drift,
      ),
    )
    return SLICES.map((_, t) => softmax(perWord.map((b) => b[t])))
  }, [counts, tokens, drift])
  const pooled = useMemo(() => {
    const tot = WORDS.map((_, w) => counts.reduce((s, c) => s + c[w], 0))
    return tot.map((n) => n / (tokens * T))
  }, [counts, tokens])

  const mae = (est: (t: number, w: number) => number) => {
    let e = 0
    for (let t = 0; t < T; t++) for (let w = 0; w < V; w++) e += Math.abs(est(t, w) - truth[t][w])
    return e / (T * V)
  }
  const t0 = slice.value - 1
  const top = (p: number[]) =>
    p
      .map((v, w) => [v, w] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 3)
      .map(([, w]) => WORDS[w])
      .join(', ')
  const handle = [{ kind: 'x' as const, at: slice.value, onDrag: slice.set, label: 'slice' }]

  return (
    <Interactive
      title="A topic drifting across time slices"
      caption="One topic over 12 time slices. Left: the true word probabilities, softmax of natural parameters that follow a random walk. Right, for one word: the per-slice frequency among N tokens (points), the Kalman-smoothed estimate that the dynamic topic model's variational update computes (dashed), and the single static topic that LDA would fit to the pooled corpus (grey). With few tokens per slice the raw frequencies are noisy and smoothing borrows strength from neighbouring slices. A very small drift variance flattens the topic towards the static fit; a very large one reproduces the raw frequencies. Drag the slice line on either chart, or step it with the arrows."
      controls={
        <>
          <ParamSlider label="time slice t" param={slice} withArrows />
          <ParamSlider label="tokens per slice N" value={tokens} onChange={setTokens} min={10} max={600} step={10} />
          <ParamSlider
            label="drift standard deviation σ"
            value={drift}
            onChange={setDrift}
            min={0.02}
            max={2}
            step={0.02}
          />
          <ParamChoice
            label="word"
            value={word}
            onChange={setWord}
            options={WORDS.map((w) => ({ value: w, label: w }))}
          />
        </>
      }
      readout={
        <>
          <Readout label={`top words at t = ${slice.value} (true)`} value={top(truth[t0])} />
          <Readout label="top words (smoothed)" value={top(smoothed[t0])} />
          <Readout label="mean abs. error, raw" value={formatNumber(mae((t, w) => raw[t][w]))} />
          <Readout label="smoothed" value={formatNumber(mae((t, w) => smoothed[t][w]))} />
          <Readout label="static" value={formatNumber(mae((_, w) => pooled[w]))} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <XYChart
          height={300}
          xLabel="time slice t"
          yLabel="p(word | topic, t)"
          xRange={[1, T]}
          yRange={[0, undefined]}
          handles={handle}
          series={WORDS.map((w, i) => ({
            name: w,
            type: 'line' as const,
            x: SLICES,
            y: truth.map((p) => p[i]),
            slot: i,
          }))}
        />
        <XYChart
          height={300}
          xLabel="time slice t"
          yLabel={`p(${word} | topic, t)`}
          xRange={[1, T]}
          yRange={[0, undefined]}
          handles={handle}
          series={[
            { name: 'true', type: 'line', x: SLICES, y: truth.map((p) => p[wi]), emphasis: true },
            { name: 'per-slice frequency', type: 'scatter', x: SLICES, y: raw.map((p) => p[wi]), slot: 0 },
            { name: 'smoothed (DTM)', type: 'line', x: SLICES, y: smoothed.map((p) => p[wi]), slot: 1, dashed: true },
            { name: 'static (LDA)', type: 'line', x: SLICES, y: SLICES.map(() => pooled[wi]), muted: true },
          ]}
        />
      </div>
    </Interactive>
  )
}
