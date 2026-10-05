import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const ITEMS = 40
const SLATE = 5
const USERS = 100
const ROUNDS = 40
const ROUND_AXIS = Array.from({ length: ROUNDS }, (_, t) => t + 1)

type Policy = 'clicks' | 'ctr' | 'explore'
type Trace = { gini: number[]; regret: number[]; distinct: number[] }

/** Gini coefficient of a non-negative vector: 0 when all equal, near 1 when one entry holds everything. */
function gini(x: number[]): number {
  const s = [...x].sort((a, b) => a - b)
  const n = s.length
  const total = s.reduce((a, b) => a + b, 0)
  if (total === 0) return 0
  const weighted = s.reduce((acc, v, i) => acc + (i + 1) * v, 0)
  return (2 * weighted) / (n * total) - (n + 1) / n
}

/**
 * Each round, USERS users see a slate of SLATE items ranked by the policy's score and click item i at rank k with
 * probability π(k)·a_i, π(k) = 1/k. Scores use only logged data. 'clicks' ranks by total clicks (popularity); 'ctr'
 * ranks by smoothed click rate per impression, ignoring position; 'explore' is 'ctr' with a share ε of slots filled at
 * random. Items start with a small random head start in clicks, as if from an earlier system.
 */
function simulate(policy: Policy, eps: number, seed: number): Trace {
  const r = stream(seed)
  const appeal = Array.from({ length: ITEMS }, () => 0.05 + 0.6 * uniform(r) ** 2)
  const best = [...appeal].sort((a, b) => b - a)
  const oracle = best.slice(0, SLATE).reduce((s, a, k) => s + a / (k + 1), 0)
  const clicks = Array.from({ length: ITEMS }, () => Math.floor(6 * uniform(r)))
  const shown = Array.from({ length: ITEMS }, () => 10)
  const exposure = new Array(ITEMS).fill(0)
  const trace: Trace = { gini: [], regret: [], distinct: [] }
  let cumRegret = 0
  for (let t = 0; t < ROUNDS; t++) {
    const score = clicks.map((c, i) => (policy === 'clicks' ? c : (c + 1) / (shown[i] + 10)))
    const ranked = score.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s)
    let expected = 0
    for (let u = 0; u < USERS; u++) {
      const slate: number[] = []
      let next = 0
      while (slate.length < SLATE) {
        let pick: number
        if (policy === 'explore' && uniform(r) < eps) pick = Math.floor(uniform(r) * ITEMS)
        else {
          while (slate.includes(ranked[next].i)) next++
          pick = ranked[next].i
        }
        if (!slate.includes(pick)) slate.push(pick)
      }
      slate.forEach((i, k) => {
        const p = appeal[i] / (k + 1)
        expected += p
        exposure[i] += 1 / (k + 1)
        shown[i] += 1
        if (uniform(r) < p) clicks[i]++
      })
    }
    cumRegret += oracle - expected / USERS
    trace.gini.push(gini(exposure))
    trace.regret.push(cumRegret)
    trace.distinct.push(exposure.filter((e) => e > 0).length)
  }
  return trace
}

/** Popularity-driven, click-rate-driven and exploring policies, round by round. */
export function LoopSimulation() {
  const state = useFigureState({
    round: float(ROUNDS, { min: 1, max: ROUNDS, step: 1, label: 'round', format: (v) => String(v) }),
    eps: float(0.2, { min: 0.05, max: 0.5, step: 0.05, label: 'exploration share ε' }),
    seed: int(3, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const traces = useMemo(
    () => ({
      clicks: simulate('clicks', 0, state.seed),
      ctr: simulate('ctr', 0, state.seed),
      explore: simulate('explore', state.eps, state.seed),
    }),
    [state.eps, state.seed],
  )
  const t = state.round
  const x = ROUND_AXIS.slice(0, t)
  const series = [
    { name: 'rank by total clicks', x, y: traces.clicks.gini.slice(0, t), slot: 0 },
    { name: 'rank by click rate', x, y: traces.ctr.gini.slice(0, t), slot: 1 },
    { name: `click rate + ε = ${state.eps} exploration`, x, y: traces.explore.gini.slice(0, t), slot: 2 },
  ] as const

  const xAxis = useAxis({ label: 'round', range: [1, ROUNDS] })
  const yAxis = useAxis({ label: 'Gini of cumulative exposure', range: [0, 1] })
  return (
    <Figure
      title="A feedback loop in simulation"
      state={state}
      caption="Forty items with hidden appeal; each round a hundred users see a five-item slate with position bias π(k) = 1/k, and the system re-ranks from the clicks it has logged. Ranking by total clicks locks in at once: the items with a head start are the only ones ever shown, they collect all the clicks, and exposure stays concentrated (Gini near 0.9). Ranking by click rate lets a shown item whose rate falls be replaced by an untried one, so more items get a chance. Random exploration shows every item; it lowers concentration and, depending on the seed, costs or saves reward. Regret is the expected clicks lost per user against always showing the five best items. Step through rounds with the arrows; change the seed to redraw the items."

      readouts={
        <>
          <Readout
            label="items ever shown (clicks / rate / explore)"
            value={`${traces.clicks.distinct[t - 1]} / ${traces.ctr.distinct[t - 1]} / ${traces.explore.distinct[t - 1]} of ${ITEMS}`}
          />
          <Readout label="cumulative regret per user, total clicks" value={formatNumber(traces.clicks.regret[t - 1])} />
          <Readout label="click rate" value={formatNumber(traces.ctr.regret[t - 1])} />
          <Readout label="with exploration" value={formatNumber(traces.explore.regret[t - 1])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
