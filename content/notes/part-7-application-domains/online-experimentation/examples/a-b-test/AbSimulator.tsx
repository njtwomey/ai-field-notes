import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { normalPdf, normalQuantile } from '@/lib/math/special'
import { requiredN, sampleBinomial, twoProportionZ, zTestPower } from '../_shared/ab'

const SIMULATIONS = 400
const pct = (v: number) => `${(100 * v).toFixed(2)}%`
const pp = (v: number) => `${(100 * v).toFixed(2)} pp`

/**
 * An A/B test on conversion: the sampling distribution of the observed difference under H₀ and under the true lift,
 * the rejection region, and one simulated experiment with its confidence interval. Power is computed by the normal
 * formula and checked by simulating many experiments.
 */
export function AbSimulator() {
  const [baseline, setBaseline] = useState(0.1)
  const [lift, setLift] = useState(1)
  const [n, setN] = useState(14751)
  const [alpha, setAlpha] = useState(0.05)
  const [seed, setSeed] = useState(1)
  const p0 = baseline
  const p1 = Math.min(Math.max(baseline + lift / 100, 0.001), 0.999)

  const r = useMemo(() => {
    const { uniform } = rng(seed)
    const one = twoProportionZ(sampleBinomial(n, p0, uniform), n, sampleBinomial(n, p1, uniform), n, alpha)
    let rejections = 0
    for (let i = 0; i < SIMULATIONS; i++) {
      const t = twoProportionZ(sampleBinomial(n, p0, uniform), n, sampleBinomial(n, p1, uniform), n, alpha)
      if (t.p < alpha) rejections++
    }
    return { one, simulatedPower: rejections / SIMULATIONS }
  }, [n, p0, p1, alpha, seed])

  const chart = useMemo(() => {
    const bar = (p0 + p1) / 2
    const s0 = Math.sqrt((2 * bar * (1 - bar)) / n)
    const s1 = Math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
    const delta = p1 - p0
    const crit = normalQuantile(1 - alpha / 2) * s0
    const lo = Math.min(-4 * s0, delta - 4 * s1)
    const hi = Math.max(4 * s0, delta + 4 * s1)
    const grid = Array.from({ length: 241 }, (_, i) => lo + ((hi - lo) * i) / 240)
    const density = (d: number, mu: number, s: number) => normalPdf((d - mu) / s) / (s * 100)
    const tail = (from: number, to: number) => grid.filter((d) => d >= from && d <= to)
    const right = tail(crit, hi)
    const left = tail(lo, -crit)
    const series: XYSeries[] = [
      {
        name: 'H₀: no difference',
        type: 'line',
        x: grid.map((d) => 100 * d),
        y: grid.map((d) => density(d, 0, s0)),
        slot: 0,
      },
      {
        name: 'true lift',
        type: 'line',
        x: grid.map((d) => 100 * d),
        y: grid.map((d) => density(d, delta, s1)),
        slot: 1,
      },
      {
        name: 'reject H₀ (power)',
        type: 'line',
        area: true,
        x: right.map((d) => 100 * d),
        y: right.map((d) => density(d, delta, s1)),
        slot: 1,
      },
      {
        // Same name as the right tail, so the legend shows and toggles one entry.
        name: 'reject H₀ (power)',
        type: 'line',
        area: true,
        x: left.map((d) => 100 * d),
        y: left.map((d) => density(d, delta, s1)),
        slot: 1,
      },
      { name: 'one experiment', type: 'scatter', x: [100 * r.one.diff], y: [0], emphasis: true },
    ]
    const segments: Segment[] = [
      { from: [100 * r.one.ci[0], 0], to: [100 * r.one.ci[1], 0] },
      { from: [100 * crit, 0], to: [100 * crit, density(crit, 0, s0) * 6] },
      { from: [-100 * crit, 0], to: [-100 * crit, density(crit, 0, s0) * 6] },
    ]
    return { series, segments, crit }
  }, [n, p0, p1, alpha, r.one])

  const power = zTestPower(p0, p1, n, alpha)
  const needed = p1 !== p0 ? requiredN(p0, p1, alpha, 0.8) : Infinity
  const significant = r.one.p < alpha

  return (
    <Interactive
      title="One A/B test, and the thousands it could have been"
      caption="Curves: the sampling distribution of the observed difference B − A when there is no effect, and when the true lift is as set. The test rejects H₀ when the observed difference lands beyond the short vertical lines; the shaded area under the true-lift curve is the power. The diamond is one simulated experiment, and the horizontal bar through it is its 95% confidence interval. Rerun to draw another experiment; power is also checked by simulating 400 of them."
      controls={
        <>
          <ParamSlider
            label="baseline rate p_A"
            value={baseline}
            onChange={setBaseline}
            min={0.01}
            max={0.5}
            step={0.005}
          />
          <ParamSlider
            label="true lift (percentage points)"
            value={lift}
            onChange={setLift}
            min={-2}
            max={3}
            step={0.05}
          />
          <ParamSlider label="users per arm n" value={n} onChange={setN} min={1000} max={60000} step={250} />
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.01}
            max={0.2}
            step={0.01}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>Rerun</ParamButton>
        </>
      }
      readout={
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
      <XYChart
        height={320}
        series={chart.series}
        segments={chart.segments}
        yRange={[0, undefined]}
        xLabel="observed B − A (percentage points)"
        yLabel="density"
      />
    </Interactive>
  )
}
