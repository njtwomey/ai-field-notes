import { useMemo } from 'react'
import { Bars, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const V = 12
const TOKENS = Array.from({ length: V }, (_, i) => i + 1)

// The target model's logits, and a fixed direction in which the draft model's logits err.
const { TARGET, NOISE } = (() => {
  const r = stream(11)
  const target = TOKENS.map((i) => 2 - 0.9 * Math.log(i) + 0.2 * normal(r)).sort((a, b) => b - a)
  return { TARGET: target, NOISE: TOKENS.map(() => normal(r)) }
})()

function softmax(z: number[], t = 1): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp((v - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** Target p, draft q and the accepted mass min(p, q); the grey remainder of each p bar is the residual max(0, p − q). */
export function DraftAcceptance() {
  const state = useFigureState({
    error: float(0.8, { min: 0, max: 2.5, step: 0.05, label: 'draft error' }),
    logTemp: float(0, {
      min: -0.5,
      max: 0.5,
      step: 0.01,
      label: 'draft temperature',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    gamma: int(4, { min: 1, max: 10, step: 1, label: 'draft length γ', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const p = softmax(TARGET)
    const q = softmax(
      TARGET.map((z, i) => z + state.error * NOISE[i]),
      10 ** state.logTemp,
    )
    const accepted = p.map((pi, i) => Math.min(pi, q[i]))
    const alpha = accepted.reduce((a, b) => a + b, 0)
    return { p, q, accepted, alpha }
  }, [state.error, state.logTemp])

  const g = state.gamma
  const a = r.alpha
  // Expected tokens per target call when each draft token is accepted independently with probability α.
  const perCall = a >= 1 - 1e-9 ? g + 1 : (1 - a ** (g + 1)) / (1 - a)

  const series = [
    { name: 'target p', x: TOKENS, y: r.p, muted: true },
    { name: 'accepted min(p, q)', x: TOKENS, y: r.accepted, slot: 0 },
    { name: 'draft q', x: TOKENS, y: r.q, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'token', range: [0.5, 12.5] })
  const yAxis = useAxis({ label: 'probability', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="How much of the draft survives"
      state={state}
      caption="The target model's next-token distribution p (grey bars) and a draft model's q (points). A drafted token is kept with probability min(1, p/q), so the coloured part of each bar, min(p, q), is the probability that the step proposes and keeps that token. The grey remainder, max(0, p − q), is where the correction step resamples after a rejection. Together they rebuild p exactly. Draft error perturbs the draft's logits; draft temperature makes it over- or under-confident."

      readouts={
        <>
          <Readout label="acceptance rate α = Σ min(p, q)" value={formatNumber(a)} />
          <Readout label="total variation 1 − α" value={formatNumber(1 - a)} />
          <Readout label="expected tokens per target call" value={formatNumber(perCall)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...series[0]} />
        <Bars {...series[1]} />
        <Points {...series[2]} />
      </Plot>
    </Figure>
  )
}
