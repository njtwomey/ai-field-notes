import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

const T = 200
const TIMES = Array.from({ length: T }, (_, t) => t + 1)

/** Paths of x_t = δ + φx_{t−1} + ε_t from x_0 = 0: a stationary AR(1) for φ < 1, a random walk at φ = 1. */
export function RandomWalkFan() {
  const phi = useParam(1, { min: 0, max: 1, step: 0.01 })
  const drift = useParam(0, { min: -0.3, max: 0.3, step: 0.01 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const count = useParam(15, { min: 1, max: 50, step: 1 })

  const r = useMemo(() => {
    const many = count.value > 1
    // Path k has its own stream, so adding paths leaves the existing ones unchanged.
    const shocks = Array.from({ length: count.value }, (_, k) => {
      const g = rng(seed.value * 1000 + k)
      return TIMES.map(() => g.normal())
    })
    const paths = shocks.map((e) => {
      let x = 0
      return e.map((v) => (x = drift.value + phi.value * x + v))
    })
    // Mean and variance of x_t from x_0 = 0: sums of the geometric series in φ and φ².
    let m = 0
    let v = 0
    const mean: number[] = []
    const sd: number[] = []
    for (let t = 0; t < T; t++) {
      m = drift.value + phi.value * m
      v = phi.value * phi.value * v + 1
      mean.push(m)
      sd.push(Math.sqrt(v))
    }
    const series: XYSeries[] = [
      ...paths.map((y): XYSeries => ({
        name: many ? 'sample paths' : 'sample path',
        type: 'line',
        x: TIMES,
        y,
        slot: 0,
        thin: many,
      })),
      { name: 'mean', type: 'line', x: TIMES, y: mean, emphasis: true, dashed: true },
      { name: 'mean ± 2 sd', type: 'line', x: TIMES, y: mean.map((u, t) => u + 2 * sd[t]), slot: 1, dashed: true },
      { name: 'mean − 2 sd', type: 'line', x: TIMES, y: mean.map((u, t) => u - 2 * sd[t]), slot: 1, dashed: true },
    ]
    return { series, sdEnd: sd[T - 1], sd10: sd[9] }
  }, [phi.value, drift.value, seed.value, count.value])

  const limit = phi.value < 1 ? formatNumber(1 / Math.sqrt(1 - phi.value ** 2)) : '∞ (grows as √t)'
  return (
    <Interactive
      title="From AR(1) to a random walk"
      caption="Sample paths of x_t = δ + φx_{t−1} + ε_t, all started at x_0 = 0, with unit-variance Gaussian shocks, drawn as light lines; the paths slider sets how many. The dashed lines are the mean and the mean ± 2 standard deviations at each t. For φ < 1 the band stops widening at ±2/√(1 − φ²) and the paths keep returning to the mean: the process forgets its start. At φ = 1 the process is a random walk, the band widens as √t and never stabilises, and a drift δ adds a straight-line trend δt."
      controls={
        <>
          <ParamSlider label="φ" param={phi} />
          <ParamSlider label="drift δ" param={drift} />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="sd of x₁₀" value={formatNumber(r.sd10)} />
          <Readout label="sd of x₂₀₀" value={formatNumber(r.sdEnd)} />
          <Readout label="limiting sd 1/√(1 − φ²)" value={limit} />
        </>
      }
    >
      <XYChart series={r.series} xLabel="t" yLabel="x_t" height={320} />
    </Interactive>
  )
}
