import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn-compute/foundation/random'
import { requiredN, sampleBinomial, twoProportionZ, zTestPower } from '../_shared/ab'
import { normalPdf, normalQuantile } from 'aifn-compute/numerics/special'

const SIMULATIONS = 400
const pct = (v: number) => `${(100 * v).toFixed(2)}%`
const pp = (v: number) => `${(100 * v).toFixed(2)} pp`

/**
 * An A/B test on conversion: the sampling distribution of the observed difference under H₀ and under the true lift,
 * the rejection region, and one simulated experiment with its confidence interval. Power is computed by the normal
 * formula and checked by simulating many experiments.
 */
export function AbSimulator() {
  const state = useFigureState({
    baseline: float(0.1, { min: 0.01, max: 0.5, step: 0.005, label: 'baseline rate p_A' }),
    lift: int(1, { min: -2, max: 3, step: 0.05, label: 'true lift (percentage points)' }),
    n: int(14751, { min: 1000, max: 60000, step: 250, label: 'users per arm n' }),
    alpha: float(0.05, { min: 0.01, max: 0.2, step: 0.01, label: 'significance level α' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const p0 = state.baseline
  const p1 = Math.min(Math.max(state.baseline + state.lift / 100, 0.001), 0.999)

  const r = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    const one = twoProportionZ(
      sampleBinomial(state.n, p0, uniform),
      state.n,
      sampleBinomial(state.n, p1, uniform),
      state.n,
      state.alpha,
    )
    let rejections = 0
    for (let i = 0; i < SIMULATIONS; i++) {
      const t = twoProportionZ(
        sampleBinomial(state.n, p0, uniform),
        state.n,
        sampleBinomial(state.n, p1, uniform),
        state.n,
        state.alpha,
      )
      if (t.p < state.alpha) rejections++
    }
    return { one, simulatedPower: rejections / SIMULATIONS }
  }, [state.n, p0, p1, state.alpha, state.seed])

  const chart = useMemo(() => {
    const bar = (p0 + p1) / 2
    const s0 = Math.sqrt((2 * bar * (1 - bar)) / state.n)
    const s1 = Math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / state.n)
    const delta = p1 - p0
    const crit = normalQuantile(1 - state.alpha / 2) * s0
    const lo = Math.min(-4 * s0, delta - 4 * s1)
    const hi = Math.max(4 * s0, delta + 4 * s1)
    const grid = Array.from({ length: 241 }, (_, i) => lo + ((hi - lo) * i) / 240)
    const density = (d: number, mu: number, s: number) => normalPdf((d - mu) / s) / (s * 100)
    const tail = (from: number, to: number) => grid.filter((d) => d >= from && d <= to)
    const right = tail(crit, hi)
    const left = tail(lo, -crit)
    const series = [
      {
        name: 'H₀: no difference',
        x: grid.map((d) => 100 * d),
        y: grid.map((d) => density(d, 0, s0)),
        slot: 0,
      },
      {
        name: 'true lift',
        x: grid.map((d) => 100 * d),
        y: grid.map((d) => density(d, delta, s1)),
        slot: 1,
      },
      {
        name: 'reject H₀ (power)',
        x: right.map((d) => 100 * d),
        y: right.map((d) => density(d, delta, s1)),
        slot: 1,
      },
      {
        // Same name as the right tail, so the legend shows and toggles one entry.
        name: 'reject H₀ (power)',
        x: left.map((d) => 100 * d),
        y: left.map((d) => density(d, delta, s1)),
        slot: 1,
      },
      { name: 'one experiment', x: [100 * r.one.diff], y: [0], emphasis: true },
    ] as const
    const segments: Segment[] = [
      { from: [100 * r.one.ci[0], 0], to: [100 * r.one.ci[1], 0] },
      { from: [100 * crit, 0], to: [100 * crit, density(crit, 0, s0) * 6] },
      { from: [-100 * crit, 0], to: [-100 * crit, density(crit, 0, s0) * 6] },
    ]
    return { series, segments, crit }
  }, [state.n, p0, p1, state.alpha, r.one])

  const power = zTestPower(p0, p1, state.n, state.alpha)
  const needed = p1 !== p0 ? requiredN(p0, p1, state.alpha, 0.8) : Infinity
  const significant = r.one.p < state.alpha

  const xAxis = useAxis({ label: 'observed B − A (percentage points)', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="One A/B test, and the thousands it could have been"
      state={state}
      caption="Curves: the sampling distribution of the observed difference B − A when there is no effect, and when the true lift is as set. The test rejects H₀ when the observed difference lands beyond the short vertical lines; the shaded area under the true-lift curve is the power. The diamond is one simulated experiment, and the horizontal bar through it is its 95% confidence interval. Rerun to draw another experiment; power is also checked by simulating 400 of them."

      readouts={
        <>
          <Readout label="A, B observed" value={`${pct(r.one.pa)}, ${pct(r.one.pb)}`} />
          <Readout label="B − A" value={pp(r.one.diff)} />
          <Readout label="95% CI" value={`${pp(r.one.ci[0])} to ${pp(r.one.ci[1])}`} />
          <Readout label="z, p" value={`${formatNumber(r.one.z)}, ${formatNumber(r.one.p)}`} />
          <Readout label="decision" value={significant ? 'reject H₀' : 'no evidence of a difference'} />
          <Readout label="power, formula" value={`${(100 * power).toFixed(1)}%`} />
          <Readout label="power, simulated" value={`${(100 * r.simulatedPower).toFixed(1)}%`} />
          <Readout label="n per arm for 80% power" value={Number.isFinite(needed) ? needed.toLocaleString() : '—'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...chart.series[0]} />
        <Curve {...chart.series[1]} />
        <Area {...chart.series[2]} />
        <Area {...chart.series[3]} />
        <Points {...chart.series[4]} />
        <Segments segments={chart.segments} />
      </Plot>
    </Figure>
  )
}
