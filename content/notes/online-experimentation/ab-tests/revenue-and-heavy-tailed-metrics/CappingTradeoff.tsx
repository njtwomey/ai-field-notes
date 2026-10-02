import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { normalCdf, normalQuantile } from '@/lib/math/special'

type Lift = 'value' | 'rate'

const MU = 3.5
const LIFT = 0.05
const X_MAX = 30
const Y_MAX = 1.6
const LIFTS: { value: Lift; label: string }[] = [
  { value: 'value', label: 'purchase values +5%' },
  { value: 'rate', label: 'purchase rate +5%' },
]

/** E[min(R, c)^k] for R ~ LogNormal(mu, s²). */
function cappedMoment(c: number, mu: number, s: number, k: number): number {
  if (!Number.isFinite(c)) return Math.exp(k * mu + (k * k * s * s) / 2)
  const z = (Math.log(c) - mu) / s
  return Math.exp(k * mu + (k * k * s * s) / 2) * normalCdf(z - k * s) + c ** k * (1 - normalCdf(z))
}

/**
 * Revenue per user Y = B·R with B ~ Bern(p) and R ~ LogNormal(μ, σ²). Capping Y at c lowers the variance and also the
 * treatment effect. Users needed for fixed power scale as variance / effect², so the ratio to the uncapped metric is
 * (var_c / var) / (Δ_c / Δ)². Every ratio is independent of μ, which only sets the currency scale.
 */
function tradeoff(share: number, p: number, s: number, lift: Lift) {
  const c = share <= 0 ? Infinity : Math.exp(MU + s * normalQuantile(1 - share))
  const m1 = cappedMoment(c, MU, s, 1)
  const m2 = cappedMoment(c, MU, s, 2)
  const mean = p * m1
  const variance = p * m2 - mean * mean
  const full = Math.exp(MU + (s * s) / 2)
  const fullVariance = p * Math.exp(2 * MU + 2 * s * s) - (p * full) ** 2
  const fullEffect = LIFT * p * full
  const effect = lift === 'rate' ? LIFT * p * m1 : p * (cappedMoment(c, MU + Math.log(1 + LIFT), s, 1) - m1)
  const retained = effect / fullEffect
  const varianceRatio = variance / fullVariance
  return { c, varianceRatio, retained, users: varianceRatio / (retained * retained) }
}

export function CappingTradeoff() {
  const [p, setP] = useState(0.05)
  const [sigma, setSigma] = useState(1.2)
  const [lift, setLift] = useState<Lift>('value')
  const share = useParam(2, { min: 0, max: X_MAX, step: 0.25 })

  const series = useMemo<XYSeries[]>(() => {
    const xs = Array.from({ length: X_MAX * 4 + 1 }, (_, i) => i / 4)
    const rows = xs.map((x) => tradeoff(x / 100, p, sigma, lift))
    return [
      { name: 'users needed', type: 'line', x: xs, y: rows.map((r) => Math.min(r.users, Y_MAX)), slot: 0 },
      { name: 'variance', type: 'line', x: xs, y: rows.map((r) => r.varianceRatio), slot: 1 },
      { name: 'effect retained', type: 'line', x: xs, y: rows.map((r) => r.retained), slot: 2 },
    ]
  }, [p, sigma, lift])

  const r = tradeoff(share.value / 100, p, sigma, lift)
  const handles: Handle[] = [
    { kind: 'x', at: share.value, label: 'cap', onDrag: (x) => share.set(Math.round(x * 4) / 4) },
  ]

  return (
    <Interactive
      title="What capping revenue buys and costs"
      caption="Revenue per user is zero for non-buyers and log-normal for buyers. Capping each user's revenue at c cuts the variance, but a change in the capped mean is smaller than the change in the true mean. The users needed for fixed power scale as variance ÷ effect², shown relative to no cap (1 = no gain; below 1 = capping helps). When the treatment raises purchase values, much of the effect sits in the tail that the cap removes; when it raises the purchase rate, the capped metric keeps most of it. Drag the line labelled cap, or use its slider; 0% means no cap."
      controls={
        <>
          <ParamSlider label="buyers above the cap (%)" param={share} />
          <ParamSlider label="purchase rate p" value={p} onChange={setP} min={0.01} max={0.3} step={0.01} />
          <ParamSlider
            label="log-normal σ of purchase value"
            value={sigma}
            onChange={setSigma}
            min={0.4}
            max={2}
            step={0.05}
          />
          <ParamChoice label="the treatment raises" value={lift} onChange={setLift} options={LIFTS} />
        </>
      }
      readout={
        <>
          <Readout label="cap (median purchase = 33.1)" value={Number.isFinite(r.c) ? formatNumber(r.c) : 'none'} />
          <Readout label="variance ratio" value={formatNumber(r.varianceRatio)} />
          <Readout label="effect retained" value={formatNumber(r.retained)} />
          <Readout label="users needed, ratio" value={formatNumber(r.users)} />
        </>
      }
    >
      <XYChart
        height={320}
        series={series}
        xRange={[0, X_MAX]}
        yRange={[0, Y_MAX]}
        xLabel="buyers above the cap (%)"
        yLabel="ratio to the uncapped metric"
        handles={handles}
      />
    </Interactive>
  )
}
