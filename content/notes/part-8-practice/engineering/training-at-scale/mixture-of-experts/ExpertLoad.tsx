import { useMemo } from 'react'
import { Bars, choice, Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

const TOKENS = 1024
const EXPERTS = 8
const EXPERT_IDS = Array.from({ length: EXPERTS }, (_, i) => i + 1)
// Fixed per-token router noise, so that only the controls change the routing.
const NOISE = (() => {
  const r = stream(11)
  return Array.from({ length: TOKENS }, () => Array.from({ length: EXPERTS }, () => normal(r)))
})()

type K = '1' | '2'
const KS = [
  { value: '1', label: 'top-1' },
  { value: '2', label: 'top-2' },
] as const

/**
 * Router logits are a per-expert preference, scaled by `skew`, plus per-token noise. Expert 1 is the most preferred
 * and expert 8 the least. Returns loads, dropped assignments and the Switch auxiliary loss N Σ f_i P_i.
 */
function route(skew: number, k: number, capacityFactor: number) {
  const load = new Array<number>(EXPERTS).fill(0)
  const meanProb = new Array<number>(EXPERTS).fill(0)
  for (const noise of NOISE) {
    const logits = noise.map((n, i) => skew * (1 - (2 * i) / (EXPERTS - 1)) + n)
    const m = Math.max(...logits)
    const e = logits.map((v) => Math.exp(v - m))
    const s = e.reduce((a, b) => a + b, 0)
    e.forEach((v, i) => (meanProb[i] += v / s / TOKENS))
    const order = logits.map((_, i) => i).sort((a, b) => logits[b] - logits[a])
    for (let j = 0; j < k; j++) load[order[j]]++
  }
  const capacity = Math.floor((capacityFactor * k * TOKENS) / EXPERTS)
  const dropped = load.reduce((a, l) => a + Math.max(0, l - capacity), 0)
  const aux = EXPERTS * load.reduce((a, l, i) => a + (l / (k * TOKENS)) * meanProb[i], 0)
  return { load, capacity, dropped: dropped / (k * TOKENS), aux }
}

export function ExpertLoad() {
  const state = useFigureState({
    skew: float(1, { min: 0, max: 3, step: 0.05, label: 'router preference' }),
    k: choice<K>(KS, '1', { label: 'experts per token' }),
    cf: float(1.25, { min: 1, max: 2, step: 0.05, label: 'capacity factor' }),
  })
  const r = useMemo(() => route(state.skew, Number(state.k), state.cf), [state.skew, state.k, state.cf])

  const series = useMemo(
    () =>
      [
        { name: 'tokens routed', x: EXPERT_IDS, y: r.load, slot: 0 },
        { name: 'capacity', x: [0.5, EXPERTS + 0.5], y: [r.capacity, r.capacity], slot: 1, dashed: true },
      ] as const,
    [r],
  )

  const xAxis = useAxis({ label: 'expert', range: [0.5, EXPERTS + 0.5] })
  const yAxis = useAxis({ label: 'tokens', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Routing 1,024 tokens to 8 experts"
      purpose="Change the router's preference, the experts per token and the capacity factor to see expert load, dropped tokens and the auxiliary loss."
      state={state}
      caption="Each token's router scores are a shared preference for lower-numbered experts plus token-specific noise. With no preference the load is nearly even and the auxiliary loss is near its minimum of 1. As the preference grows, popular experts exceed their capacity and the excess tokens are dropped, while the auxiliary loss rises."

      readouts={
        <>
          <Readout label="capacity per expert" value={`${r.capacity} tokens`} />
          <Readout label="assignments dropped" value={`${formatNumber(100 * r.dropped)}%`} />
          <Readout label="N Σ f_i P_i" value={formatNumber(r.aux)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Bars {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
