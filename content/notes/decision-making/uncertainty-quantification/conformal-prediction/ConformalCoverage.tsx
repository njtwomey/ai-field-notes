import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { logGamma } from '@/lib/math/special'

const XS = linspace(0, 6, 121)
const TEST = 3000
const mean = (x: number) => Math.sin(x) + 0.3 * x
const noise = (x: number) => 0.15 + 0.12 * x
const COV = linspace(0.5, 1, 301)

/**
 * Split conformal prediction with the absolute residual score. The regressor is the true mean function (a stand-in
 * for any fitted model); noise grows with x. The interval is f(x) ± q̂, where q̂ is the ⌈(n + 1)(1 − α)⌉-th smallest
 * calibration residual. The right chart is the Beta(n + 1 − l, l) law of the coverage given the calibration set.
 */
export function ConformalCoverage() {
  const alpha = useParam(0.1, { min: 0.02, max: 0.3, step: 0.01 })
  const n = useParam(100, { min: 10, max: 2000, step: 10 })
  const seed = useParam(1, { min: 1, max: 50, step: 1 })

  const test = useMemo(() => {
    const g = rng(999)
    return Array.from({ length: TEST }, () => {
      const x = 6 * g.uniform()
      return { x, y: mean(x) + noise(x) * g.normal() }
    })
  }, [])

  const r = useMemo(() => {
    const g = rng(seed.value)
    const scores: number[] = []
    for (let i = 0; i < n.value; i++) {
      const x = 6 * g.uniform()
      scores.push(Math.abs(noise(x) * g.normal()))
    }
    scores.sort((a, b) => a - b)
    const k = Math.ceil((n.value + 1) * (1 - alpha.value))
    const q = k > n.value ? Infinity : scores[k - 1]
    const inside = (p: { x: number; y: number }) => Math.abs(p.y - mean(p.x)) <= q
    const low = test.filter((p) => p.x < 2)
    const high = test.filter((p) => p.x >= 4)
    const l = Math.floor((n.value + 1) * alpha.value)
    // Beta(n + 1 − l, l) density of the conditional coverage; undefined when l = 0 (the interval is infinite).
    const aB = n.value + 1 - l
    const bB = l
    const logB = logGamma(aB) + logGamma(bB) - logGamma(aB + bB)
    const density =
      l > 0 ? COV.map((c) => Math.exp((aB - 1) * Math.log(c) + (bB - 1) * Math.log(1 - c) - logB)) : COV.map(() => 0)
    return {
      q,
      k,
      l,
      coverage: test.filter(inside).length / TEST,
      coverLow: low.filter(inside).length / low.length,
      coverHigh: high.filter(inside).length / high.length,
      density: density.map((d) => (Number.isFinite(d) ? d : 0)),
    }
  }, [alpha.value, n.value, seed.value, test])

  const band = Number.isFinite(r.q) ? r.q : 10
  const top = Math.max(...r.density, 1) * 1.1

  return (
    <Interactive
      title="Split conformal intervals and their coverage"
      caption="Left: test points and the conformal interval f(x) ± q̂, where q̂ is the ⌈(n + 1)(1 − α)⌉-th smallest absolute residual on n calibration points. Right: the distribution of the coverage you get for a random calibration set, Beta(n + 1 − l, l) with l = ⌊(n + 1)α⌋; the dashed line is the target 1 − α and the dot the coverage of this calibration set on 3000 test points. Change the seed to redraw the calibration set. Coverage holds on average over x but not in each region: the noise grows with x."
      controls={
        <>
          <ParamSlider label="miscoverage α" param={alpha} />
          <ParamSlider label="calibration points n" param={n} format={(v) => String(v)} />
          <ParamSlider label="calibration seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="q̂" value={Number.isFinite(r.q) ? formatNumber(r.q) : '∞'} />
          <Readout label="rank used" value={`${r.k} of ${n.value}`} />
          <Readout label="test coverage" value={formatNumber(r.coverage)} />
          <Readout label="coverage, x < 2" value={formatNumber(r.coverLow)} />
          <Readout label="coverage, x ≥ 4" value={formatNumber(r.coverHigh)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          xLabel="x"
          yLabel="y"
          xRange={[0, 6]}
          yRange={[-2.5, 4.5]}
          series={[
            {
              name: 'test points',
              type: 'scatter',
              x: test.slice(0, 600).map((p) => p.x),
              y: test.slice(0, 600).map((p) => p.y),
              muted: true,
            },
            { name: 'model f(x)', type: 'line', x: XS, y: XS.map(mean), emphasis: true },
            { name: 'upper f(x) + q̂', type: 'line', x: XS, y: XS.map((x) => mean(x) + band), slot: 0 },
            { name: 'lower f(x) − q̂', type: 'line', x: XS, y: XS.map((x) => mean(x) - band), slot: 0 },
          ]}
        />
        <XYChart
          height={300}
          xLabel="coverage given the calibration set"
          yLabel="density"
          xRange={[0.5, 1]}
          yRange={[0, top]}
          series={[
            { name: 'Beta(n + 1 − l, l)', type: 'line', x: COV, y: r.density, slot: 1, area: true },
            {
              name: 'target 1 − α',
              type: 'line',
              x: [1 - alpha.value, 1 - alpha.value],
              y: [0, top],
              dashed: true,
              muted: true,
            },
            { name: 'this calibration set', type: 'scatter', x: [r.coverage], y: [0], emphasis: true },
          ]}
        />
      </div>
    </Interactive>
  )
}
