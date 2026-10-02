import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { betaPdf, betaQuantile } from '../_shared/beta'

const GRID = linspace(0.001, 0.999, 400)

/**
 * Beta prior with mean m and strength κ = a + b, updated by k successes in n trials. The posterior is
 * Beta(a + k, b + n − k); its mean is a weighted average of the prior mean and k/n with weight n/(κ + n) on the data.
 */
export function BetaUpdate() {
  const [m, setM] = useState(0.3)
  const [kappa, setKappa] = useState(10)
  const [n, setN] = useState(20)
  const [rate, setRate] = useState(0.7)

  const r = useMemo(() => {
    const k = Math.round(rate * n)
    const a = m * kappa
    const b = (1 - m) * kappa
    const a1 = a + k
    const b1 = b + n - k
    const lo = betaQuantile(0.025, a1, b1)
    const hi = betaQuantile(0.975, a1, b1)
    const prior = GRID.map((x) => betaPdf(x, a, b))
    // The likelihood θ^k (1 − θ)^(n − k), normalised to unit area so that it shares the density axis.
    const like = n > 0 ? GRID.map((x) => betaPdf(x, k + 1, n - k + 1)) : GRID.map(() => 1)
    const post = GRID.map((x) => betaPdf(x, a1, b1))
    const inside = GRID.map((x, i) => (x >= lo && x <= hi ? post[i] : NaN))
    const cap = Math.min(Math.max(...post, ...like) * 1.08, 40)
    const series: XYSeries[] = [
      { name: 'prior', type: 'line', x: GRID, y: prior, slot: 0, dashed: true },
      { name: 'likelihood (normalised)', type: 'line', x: GRID, y: like, slot: 1 },
      { name: 'posterior', type: 'line', x: GRID, y: post, slot: 2 },
      { name: '95% credible interval', type: 'line', x: GRID, y: inside, slot: 2, area: true },
    ]
    return { k, a1, b1, lo, hi, series, cap, mean: a1 / (a1 + b1), w: n / (kappa + n) }
  }, [m, kappa, n, rate])

  return (
    <Interactive
      title="Beta–Binomial updating"
      caption="A Beta prior with mean m and strength κ = a + b (pseudo-observations), updated with k successes in n trials. The posterior Beta(a + k, b + n − k) sits between the prior and the likelihood; its mean gives weight n/(κ + n) to the data. The shaded band is the central 95% credible interval. A strong prior needs many trials to move; with n much larger than κ the prior hardly matters."
      controls={
        <>
          <ParamSlider label="prior mean m" value={m} onChange={setM} min={0.05} max={0.95} step={0.05} />
          <ParamSlider label="prior strength κ = a + b" value={kappa} onChange={setKappa} min={1} max={100} step={1} />
          <ParamSlider label="trials n" value={n} onChange={setN} min={0} max={200} step={1} withArrows />
          <ParamSlider
            label="observed success rate k/n"
            value={rate}
            onChange={setRate}
            min={0}
            max={1}
            step={0.01}
            format={(v) => `${formatNumber(v)} (k = ${Math.round(v * n)})`}
          />
        </>
      }
      readout={
        <>
          <Readout label="posterior" value={`Beta(${formatNumber(r.a1)}, ${formatNumber(r.b1)})`} />
          <Readout label="posterior mean" value={formatNumber(r.mean)} />
          <Readout label="95% interval" value={`[${formatNumber(r.lo)}, ${formatNumber(r.hi)}]`} />
          <Readout label="weight on data n/(κ + n)" value={formatNumber(r.w)} />
        </>
      }
    >
      <XYChart
        height={300}
        series={r.series}
        xRange={[0, 1]}
        yRange={[0, r.cap]}
        xLabel="success probability θ"
        yLabel="density"
      />
    </Interactive>
  )
}
