import { useMemo } from 'react'
import { Bars, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const V = 30
const RANKS = Array.from({ length: V }, (_, i) => i + 1)

/** Fixed next-token logits with a long tail: a Zipf-like decay plus a little seeded noise, sorted by size. */
const LOGITS = (() => {
  const r = stream(7)
  return RANKS.map((i) => 3 - 1.2 * Math.log(i) + 0.3 * normal(r)).sort((a, b) => b - a)
})()

function softmax(z: number[], t: number): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp((v - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

const entropyBits = (p: number[]) => -p.reduce((a, q) => (q > 0 ? a + q * Math.log2(q) : a), 0)

/** Temperature, then top-k, then top-p, applied to one next-token distribution. */
export function NextTokenSampler() {
  const state = useFigureState({
    logT: float(0, {
      min: -1,
      max: 0.5,
      step: 0.01,
      label: 'temperature T',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    k: float(V, { min: 1, max: V, step: 1, label: 'top-k', format: (v) => String(v) }),
    topP: float(1, { min: 0.05, max: 1, step: 0.01, label: 'top-p' }),
  })
  const t = 10 ** state.logT

  const r = useMemo(() => {
    const tempered = softmax(LOGITS, t)
    // Tokens are sorted by logit, and temperature preserves order, so both truncations keep a prefix.
    let cum = 0
    let nucleus = V
    for (let i = 0; i < V; i++) {
      cum += tempered[i]
      if (cum >= state.topP - 1e-12) {
        nucleus = i + 1
        break
      }
    }
    const kept = Math.min(state.k, nucleus)
    const mass = tempered.slice(0, kept).reduce((a, b) => a + b, 0)
    const final = tempered.map((q, i) => (i < kept ? q / mass : 0))
    return { tempered, final, kept, mass }
  }, [t, state.k, state.topP])

  const series = [
    { name: 'after temperature', x: RANKS, y: r.tempered, muted: true },
    { name: 'sampling distribution', x: RANKS, y: r.final, slot: 0 },
  ] as const

  const xAxis = useAxis({ label: 'token rank', range: [0.5, 30.5] })
  const yAxis = useAxis({ label: 'probability', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Temperature, top-k and top-p on one distribution"
      state={state}
      caption="Thirty candidate next tokens, ordered by the model's logit. Grey bars are the softmax at temperature T. Top-k keeps the k most probable tokens and top-p keeps the smallest set whose probability reaches p; the coloured bars are what remains after both, renormalised to sum to one. Low temperature concentrates mass on the first token; high temperature flattens the tail, and the truncations cut it off again."

      readouts={
        <>
          <Readout label="tokens kept" value={r.kept} />
          <Readout label="mass kept before renormalising" value={formatNumber(r.mass)} />
          <Readout label="entropy after temperature (bits)" value={formatNumber(entropyBits(r.tempered))} />
          <Readout label="entropy of sampling distribution (bits)" value={formatNumber(entropyBits(r.final))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...series[0]} />
        <Bars {...series[1]} />
      </Plot>
    </Figure>
  )
}
