import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'
import { histogramDensity } from '../_shared/ode'

const H = 0.02
const Z0 = linspace(-4, 4, 321)
const SAMPLES = (() => {
  const { normal } = rng(3)
  return Float64Array.from({ length: 4000 }, () => normal())
})()
const LO = -5
const HI = 5

/**
 * A one-dimensional continuous normalising flow ż = f(z) = a·tanh(2z) + c. Each grid point z₀ is integrated together
 * with its log-density, d log p / dt = −f′(z), which is the instantaneous change of variables.
 */
export function ContinuousFlow1d() {
  const time = useParam(1, { min: 0, max: 2, step: 0.1 })
  const a = useParam(1, { min: -1.5, max: 1.5, step: 0.1 })
  const c = useParam(0, { min: -1, max: 1, step: 0.1 })

  const result = useMemo(() => {
    const f = (z: number) => a.value * Math.tanh(2 * z) + c.value
    const df = (z: number) => (2 * a.value) / Math.cosh(2 * z) ** 2
    const steps = Math.round(time.value / H)
    // RK4 on the pair (z, log p): both right-hand sides depend on z only.
    const advance = (z: number, lp: number) => {
      for (let k = 0; k < steps; k++) {
        const k1 = f(z)
        const l1 = -df(z)
        const k2 = f(z + (H / 2) * k1)
        const l2 = -df(z + (H / 2) * k1)
        const k3 = f(z + (H / 2) * k2)
        const l3 = -df(z + (H / 2) * k2)
        const k4 = f(z + H * k3)
        const l4 = -df(z + H * k3)
        z += (H / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        lp += (H / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
      }
      return [z, lp] as const
    }
    const curve = Z0.map((z0) => advance(z0, Math.log(normalPdf(z0))))
    const pushed = Float64Array.from(SAMPLES, (z) => advance(z, 0)[0])
    // Mass under the ODE-predicted density, by the trapezoid rule on the moved grid.
    let mass = 0
    for (let i = 1; i < curve.length; i++)
      mass += 0.5 * (Math.exp(curve[i][1]) + Math.exp(curve[i - 1][1])) * (curve[i][0] - curve[i - 1][0])
    return { curve, hist: histogramDensity(pushed, LO, HI, 60), mass }
  }, [time.value, a.value, c.value])

  const series = useMemo<XYSeries[]>(
    () => [
      { name: 'pushed samples (histogram)', type: 'bar', x: result.hist.x, y: result.hist.y, muted: true },
      { name: 'base density N(0, 1)', type: 'line', x: Z0, y: Z0.map(normalPdf), slot: 1, dashed: true },
      {
        name: 'p_t from d log p/dt = −f′(z)',
        type: 'line',
        x: result.curve.map((p) => p[0]),
        y: result.curve.map((p) => Math.exp(p[1])),
        slot: 0,
      },
    ],
    [result],
  )

  return (
    <Interactive
      title="Instantaneous change of variables in one dimension"
      caption="Samples from N(0, 1) follow ż = a·tanh(2z) + c. The histogram shows where they end up. The curve is computed without samples: each grid point is moved by the ODE while its log-density falls at rate f′(z). The two agree. With a > 0 the field pushes mass away from 0 and splits the density in two; with a < 0 it squeezes the density into a spike."
      controls={
        <>
          <ParamSlider label="time t" param={time} withArrows />
          <ParamSlider label="strength a" param={a} />
          <ParamSlider label="drift c" param={c} />
        </>
      }
      readout={<Readout label="mass under the curve" value={formatNumber(result.mass)} />}
    >
      <XYChart height={300} xLabel="z" yLabel="density" series={series} xRange={[LO, HI]} yRange={[0, undefined]} />
    </Interactive>
  )
}
