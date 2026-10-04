import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normalQuantile } from 'aifn/numerics/special'

const MAX_DAYS = 56
const K = normalQuantile(0.975) + normalQuantile(0.8)

const percent = (v: number) => `${(100 * v).toFixed(2)}%`
const count = (v: number) => Math.round(v).toLocaleString('en-GB')

/**
 * Relative minimum detectable effect of a conversion-rate A/B test (α = 0.05 two-sided, power 0.8) against the number
 * of days it runs. A population of P users each visits on any day with probability π, so D = Pπ users visit per day
 * and U(d) = P(1 − (1 − π)^d) distinct users have been seen after d days. The dashed curve assumes every visitor is
 * new, U(d) = D·d.
 */
export function MdeByDays() {
  const state = useFigureState({
    days: int(14, { min: 1, max: MAX_DAYS, step: 1, label: 'days' }),
    p: float(0.05, { min: 0.005, max: 0.3, step: 0.005, label: 'baseline conversion' }),
    daily: int(100, { min: 5, max: 500, step: 5, label: 'visitors per day (thousands)' }),
    visit: slider(0.02, 1, 0.2, { step: 0.01, label: 'daily visit probability π' }),
    q: float(0.5, { min: 0.05, max: 0.5, step: 0.05, label: 'share in treatment q' }),
  })

  const r = useMemo(() => {
    const D = state.daily * 1000
    const P = D / state.visit
    const unique = (d: number) => P * (1 - Math.pow(1 - state.visit, d))
    // Relative MDE with N users in total, a share q of them treated: (z + z)·(σ/μ)/√(N q (1 − q)), σ² = p(1 − p).
    const cv = Math.sqrt((1 - state.p) / state.p)
    const mde = (n: number) => (K * cv) / Math.sqrt(n * state.q * (1 - state.q))
    const xs = Array.from({ length: MAX_DAYS }, (_, i) => i + 1)
    const series = [
      { name: 'returning visitors', x: xs, y: xs.map((d) => 100 * mde(unique(d))), slot: 0 },
      { name: 'every visitor new', x: xs, y: xs.map((d) => 100 * mde(D * d)), slot: 1, dashed: true },
    ] as const
    return { series, unique, mde, D }
  }, [state.p, state.daily, state.visit, state.q])

  const users = r.unique(state.days)
  const mde = r.mde(users)

  const xAxis = useAxis({ label: 'days', range: [1, MAX_DAYS] })
  const yAxis = useAxis({ label: 'relative MDE (%)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Detectable lift against experiment duration"
      state={state}
      caption="Relative minimum detectable effect of a conversion-rate test at α = 0.05 (two-sided) and 80% power, as the experiment runs. Each user visits on any given day with the chosen probability, so later days bring mostly returning users and the number of distinct users grows more slowly than the traffic. The dashed curve assumes every visitor is new, which gives the familiar 1/√days decline. The model holds each user's conversion probability fixed; in practice it rises with exposure. Drag the line labelled days, or use its slider."

      readouts={
        <>
          <Readout label="distinct users" value={count(users)} />
          <Readout label="relative MDE" value={percent(mde)} />
          <Readout label="absolute MDE" value={`${formatNumber(100 * mde * state.p)} pp`} />
          <Readout label="if every visitor were new" value={percent(r.mde(r.D * state.days))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...r.series[0]} />
        <Curve {...r.series[1]} />
        <Handle kind="x" at={state.days} label="days" onDrag={(x) => state.set('days', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
