import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { normalQuantile } from '@/lib/math/special'

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
  const [p, setP] = useState(0.05)
  const [daily, setDaily] = useState(100)
  const [visit, setVisit] = useState(0.2)
  const [q, setQ] = useState(0.5)
  const days = useParam(14, { min: 1, max: MAX_DAYS, step: 1 })

  const r = useMemo(() => {
    const D = daily * 1000
    const P = D / visit
    const unique = (d: number) => P * (1 - Math.pow(1 - visit, d))
    // Relative MDE with N users in total, a share q of them treated: (z + z)·(σ/μ)/√(N q (1 − q)), σ² = p(1 − p).
    const cv = Math.sqrt((1 - p) / p)
    const mde = (n: number) => (K * cv) / Math.sqrt(n * q * (1 - q))
    const xs = Array.from({ length: MAX_DAYS }, (_, i) => i + 1)
    const series: XYSeries[] = [
      { name: 'returning visitors', type: 'line', x: xs, y: xs.map((d) => 100 * mde(unique(d))), slot: 0 },
      { name: 'every visitor new', type: 'line', x: xs, y: xs.map((d) => 100 * mde(D * d)), slot: 1, dashed: true },
    ]
    return { series, unique, mde, D }
  }, [p, daily, visit, q])

  const users = r.unique(days.value)
  const mde = r.mde(users)
  const handles: Handle[] = [{ kind: 'x', at: days.value, label: 'days', onDrag: (x) => days.set(Math.round(x)) }]

  return (
    <Interactive
      title="Detectable lift against experiment duration"
      caption="Relative minimum detectable effect of a conversion-rate test at α = 0.05 (two-sided) and 80% power, as the experiment runs. Each user visits on any given day with the chosen probability, so later days bring mostly returning users and the number of distinct users grows more slowly than the traffic. The dashed curve assumes every visitor is new, which gives the familiar 1/√days decline. The model holds each user's conversion probability fixed; in practice it rises with exposure. Drag the line labelled days, or use its slider."
      controls={
        <>
          <ParamSlider label="days" param={days} />
          <ParamSlider label="baseline conversion" value={p} onChange={setP} min={0.005} max={0.3} step={0.005} />
          <ParamSlider
            label="visitors per day (thousands)"
            value={daily}
            onChange={setDaily}
            min={5}
            max={500}
            step={5}
          />
          <ParamSlider
            label="daily visit probability π"
            value={visit}
            onChange={setVisit}
            min={0.02}
            max={1}
            step={0.01}
          />
          <ParamSlider label="share in treatment q" value={q} onChange={setQ} min={0.05} max={0.5} step={0.05} />
        </>
      }
      readout={
        <>
          <Readout label="distinct users" value={count(users)} />
          <Readout label="relative MDE" value={percent(mde)} />
          <Readout label="absolute MDE" value={`${formatNumber(100 * mde * p)} pp`} />
          <Readout label="if every visitor were new" value={percent(r.mde(r.D * days.value))} />
        </>
      }
    >
      <XYChart
        height={300}
        series={r.series}
        xRange={[1, MAX_DAYS]}
        yRange={[0, undefined]}
        xLabel="days"
        yLabel="relative MDE (%)"
        handles={handles}
      />
    </Interactive>
  )
}
