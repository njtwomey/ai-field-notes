import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const ITEMS = 10
const IDX = Array.from({ length: ITEMS }, (_, i) => i + 1)
/** True click probability given examination, one per item (item 1 is the most relevant). */
const REL = [0.62, 0.55, 0.5, 0.42, 0.4, 0.33, 0.3, 0.24, 0.2, 0.15]
const REPS = 20

type Estimates = { naive: number[]; ips: number[]; clipped: number[] }

/**
 * Simulate `sessions` impressions of all ten items. The logging ranker sorts items by relevance plus Gaussian noise
 * (so relevant items are usually on top: rank is confounded with relevance). A click happens with probability
 * π(k)·r under the position-based model π(k) = k^(−η). Estimate each item's r by its click rate (naive), by the mean of
 * click/π(k) (IPS) and with weights capped at τ (clipped IPS).
 */
function simulate(seed: number, sessions: number, eta: number, noise: number, tau: number): Estimates {
  const r = stream(seed)
  const prop = IDX.map((k) => k ** -eta)
  const clicks = new Array(ITEMS).fill(0)
  const ips = new Array(ITEMS).fill(0)
  const clipped = new Array(ITEMS).fill(0)
  for (let s = 0; s < sessions; s++) {
    const order = REL.map((rel, i) => ({ i, score: rel + noise * normal(r) })).sort((a, b) => b.score - a.score)
    order.forEach(({ i }, k) => {
      if (uniform(r) < prop[k] * REL[i]) {
        clicks[i]++
        ips[i] += 1 / prop[k]
        clipped[i] += Math.min(1 / prop[k], tau)
      }
    })
  }
  return {
    naive: clicks.map((c) => c / sessions),
    ips: ips.map((c) => c / sessions),
    clipped: clipped.map((c) => c / sessions),
  }
}

const rmse = (est: number[]) => Math.sqrt(est.reduce((s, e, i) => s + (e - REL[i]) ** 2, 0) / ITEMS)

/** Naive click rates against IPS-corrected estimates of relevance under position bias. */
export function IpsClicks() {
  const state = useFigureState({
    eta: float(1, { min: 0, max: 2, step: 0.1, label: 'position bias η' }),
    sessions: int(500, { min: 50, max: 3000, step: 50, label: 'sessions logged', format: (v) => String(v) }),
    noise: float(0.1, { min: 0, max: 0.5, step: 0.02, label: 'logging-rank noise' }),
    tau: float(5, { min: 1, max: 20, step: 0.5, label: 'clipping cap τ' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const one = useMemo(
    () => simulate(state.seed, state.sessions, state.eta, state.noise, state.tau),
    [state.seed, state.sessions, state.eta, state.noise, state.tau],
  )
  // Spread across independent replications: the price of unbiasedness.
  const spread = useMemo(() => {
    const reps = Array.from({ length: REPS }, (_, k) =>
      simulate(1000 + k, state.sessions, state.eta, state.noise, state.tau),
    )
    const sd = (key: keyof Estimates) => {
      let total = 0
      for (let i = 0; i < ITEMS; i++) {
        const xs = reps.map((e) => e[key][i])
        const m = xs.reduce((a, b) => a + b, 0) / REPS
        total += xs.reduce((a, b) => a + (b - m) ** 2, 0) / (REPS - 1)
      }
      return Math.sqrt(total / ITEMS)
    }
    return { naive: sd('naive'), ips: sd('ips'), clipped: sd('clipped') }
  }, [state.sessions, state.eta, state.noise, state.tau])

  const series = [
    { name: 'true relevance r', x: IDX, y: REL, emphasis: true },
    { name: 'naive click rate', x: IDX, y: one.naive, slot: 0 },
    { name: 'IPS estimate', x: IDX, y: one.ips, slot: 1 },
    { name: `clipped IPS (τ = ${state.tau})`, x: IDX, y: one.clipped, slot: 2 },
  ] as const

  const xAxis = useAxis({ label: 'item (1 = most relevant)', range: [0.5, 10.5] })
  const yAxis = useAxis({ label: 'click probability if examined', range: [0, 1] })
  return (
    <Figure
      title="Correcting click rates for position"
      state={state}
      caption="Ten items are shown in every session, ranked by a logging model that orders them by true relevance plus noise, so the most relevant items usually sit at the top. Clicks follow the position-based model with examination π(k) = k^(−η). The naive click rate estimates π(k)·r, not r, and collapses for items that are usually low in the list. Weighting each click by 1/π(k) removes the bias, at the cost of spread: the readout gives the standard deviation of each estimator over 20 independent logs. Capping weights at τ trades some of the bias back for lower spread."

      readouts={
        <>
          <Readout label="RMSE naive" value={formatNumber(rmse(one.naive))} />
          <Readout label="RMSE IPS" value={formatNumber(rmse(one.ips))} />
          <Readout label="RMSE clipped" value={formatNumber(rmse(one.clipped))} />
          <Readout label="spread naive" value={formatNumber(spread.naive)} />
          <Readout label="spread IPS" value={formatNumber(spread.ips)} />
          <Readout label="spread clipped" value={formatNumber(spread.clipped)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
        <Points {...series[2]} />
        <Points {...series[3]} />
      </Plot>
    </Figure>
  )
}
