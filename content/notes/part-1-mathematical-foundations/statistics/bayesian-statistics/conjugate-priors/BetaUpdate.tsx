import { useMemo } from 'react'
import { Area, Curve, Figure, formatNumber, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { betaPdf, betaQuantile } from '../_shared/beta'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const GRID = toFlat(linspace(0.001, 0.999, 400))

/**
 * Beta prior with mean m and strength κ = a + b, updated by k successes in n trials. The posterior is
 * Beta(a + k, b + n − k); its mean is a weighted average of the prior mean and k/n with weight n/(κ + n) on the data.
 */
export function BetaUpdate() {
  const state = useFigureState({
    m: slider(0.05, 0.95, 0.3, { step: 0.05, label: 'prior mean m' }),
    kappa: int(10, { min: 1, max: 100, suggestions: [1, 2, 10, 50, 100], label: 'prior strength κ = a + b' }),
    n: int(20, { min: 0, max: 200, suggestions: [0, 5, 20, 100, 200], label: 'trials n' }),
    rate: slider(0, 1, 0.7, { step: 0.01, label: 'observed success rate k/n' }),
  })
  const { m, kappa, n, rate } = state

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
    const series = [
      { name: 'prior', x: GRID, y: prior, slot: 0, dashed: true },
      { name: 'likelihood (normalised)', x: GRID, y: like, slot: 1 },
      { name: 'posterior', x: GRID, y: post, slot: 2 },
      { name: '95% credible interval', x: GRID, y: inside, slot: 2 },
    ] as const
    return { k, a1, b1, lo, hi, series, cap, mean: a1 / (a1 + b1), w: n / (kappa + n) }
  }, [m, kappa, n, rate])

  const xAxis = useAxis({ label: 'success probability θ', range: [0, 1] })
  const yAxis = useAxis({ label: 'density', range: [0, r.cap] })
  return (
    <Figure
      title="Beta–Binomial updating"
      caption="A Beta prior with mean m and strength κ = a + b (pseudo-observations), updated with k successes in n trials. The posterior Beta(a + k, b + n − k) sits between the prior and the likelihood; its mean gives weight n/(κ + n) to the data. The shaded band is the central 95% credible interval. A strong prior needs many trials to move; with n much larger than κ the prior hardly matters."
      state={state}
      readouts={
        <>
          <Readout label="successes k" value={`${r.k} of ${n}`} />
          <Readout label="posterior" value={`Beta(${formatNumber(r.a1)}, ${formatNumber(r.b1)})`} />
          <Readout label="posterior mean" value={formatNumber(r.mean)} />
          <Readout label="95% interval" value={`[${formatNumber(r.lo)}, ${formatNumber(r.hi)}]`} />
          <Readout label="weight on data n/(κ + n)" value={formatNumber(r.w)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...r.series[0]} />
        <Curve {...r.series[1]} />
        <Curve {...r.series[2]} />
        <Area {...r.series[3]} />
      </Plot>
    </Figure>
  )
}
