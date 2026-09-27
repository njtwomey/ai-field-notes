import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const V = 12
const TOKENS = Array.from({ length: V }, (_, i) => i + 1)

// The target model's logits, and a fixed direction in which the draft model's logits err.
const { TARGET, NOISE } = (() => {
  const r = rng(11)
  const target = TOKENS.map((i) => 2 - 0.9 * Math.log(i) + 0.2 * r.normal()).sort((a, b) => b - a)
  return { TARGET: target, NOISE: TOKENS.map(() => r.normal()) }
})()

function softmax(z: number[], t = 1): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp((v - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** Target p, draft q and the accepted mass min(p, q); the grey remainder of each p bar is the residual max(0, p − q). */
export function DraftAcceptance() {
  const error = useParam(0.8, { min: 0, max: 2.5, step: 0.05 })
  const logTemp = useParam(0, { min: -0.5, max: 0.5, step: 0.01 })
  const gamma = useParam(4, { min: 1, max: 10, step: 1 })

  const r = useMemo(() => {
    const p = softmax(TARGET)
    const q = softmax(
      TARGET.map((z, i) => z + error.value * NOISE[i]),
      10 ** logTemp.value,
    )
    const accepted = p.map((pi, i) => Math.min(pi, q[i]))
    const alpha = accepted.reduce((a, b) => a + b, 0)
    return { p, q, accepted, alpha }
  }, [error.value, logTemp.value])

  const g = gamma.value
  const a = r.alpha
  // Expected tokens per target call when each draft token is accepted independently with probability α.
  const perCall = a >= 1 - 1e-9 ? g + 1 : (1 - a ** (g + 1)) / (1 - a)

  const series: XYSeries[] = [
    { name: 'target p', type: 'bar', x: TOKENS, y: r.p, muted: true },
    { name: 'accepted min(p, q)', type: 'bar', x: TOKENS, y: r.accepted, slot: 0 },
    { name: 'draft q', type: 'scatter', x: TOKENS, y: r.q, slot: 1 },
  ]

  return (
    <Interactive
      title="How much of the draft survives"
      caption="The target model's next-token distribution p (grey bars) and a draft model's q (points). A drafted token is kept with probability min(1, p/q), so the coloured part of each bar, min(p, q), is the probability that the step proposes and keeps that token. The grey remainder, max(0, p − q), is where the correction step resamples after a rejection. Together they rebuild p exactly. Draft error perturbs the draft's logits; draft temperature makes it over- or under-confident."
      controls={
        <>
          <ParamSlider label="draft error" param={error} />
          <ParamSlider label="draft temperature" param={logTemp} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="draft length γ" param={gamma} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="acceptance rate α = Σ min(p, q)" value={formatNumber(a)} />
          <Readout label="total variation 1 − α" value={formatNumber(1 - a)} />
          <Readout label="expected tokens per target call" value={formatNumber(perCall)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="token"
        yLabel="probability"
        xRange={[0.5, 12.5]}
        yRange={[0, undefined]}
        height={300}
      />
    </Interactive>
  )
}
