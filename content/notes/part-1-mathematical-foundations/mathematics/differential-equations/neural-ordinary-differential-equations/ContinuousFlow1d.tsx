import { useMemo } from 'react'
import { Bars, Curve, Figure, float, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal as drawNormal, stream } from 'aifn-compute/foundation/random'
import { histogramDensity } from '../_shared/ode'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalPdf } from 'aifn-compute/numerics/special'

const H = 0.02
const Z0 = toFlat(linspace(-4, 4, 321))
const SAMPLES = (() => {
  const rs = stream(3)
  const normal = () => drawNormal(rs)
  return Float64Array.from({ length: 4000 }, () => normal())
})()
const LO = -5
const HI = 5

/**
 * A one-dimensional continuous normalising flow ż = f(z) = a·tanh(2z) + c. Each grid point z₀ is integrated together
 * with its log-density, d log p / dt = −f′(z), which is the instantaneous change of variables.
 */
export function ContinuousFlow1d() {
  const state = useFigureState({
    time: slider(0, 2, 1, { step: 0.1, label: 'time t' }),
    a: float(1, { min: -1.5, max: 1.5, step: 0.1, label: 'strength a' }),
    c: float(0, { min: -1, max: 1, step: 0.1, label: 'drift c' }),
  })

  const result = useMemo(() => {
    const f = (z: number) => state.a * Math.tanh(2 * z) + state.c
    const df = (z: number) => (2 * state.a) / Math.cosh(2 * z) ** 2
    const steps = Math.round(state.time / H)
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
  }, [state.time, state.a, state.c])

  const series = useMemo(
    () =>
      [
        { name: 'pushed samples (histogram)', x: result.hist.x, y: result.hist.y, muted: true },
        { name: 'base density N(0, 1)', x: Z0, y: Z0.map((v: number) => normalPdf(v)), slot: 1, dashed: true },
        {
          name: 'p_t from d log p/dt = −f′(z)',
          x: result.curve.map((p) => p[0]),
          y: result.curve.map((p) => Math.exp(p[1])),
          slot: 0,
        },
      ] as const,
    [result],
  )

  const xAxis = useAxis({ label: 'z', range: [LO, HI] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Instantaneous change of variables in one dimension"
      state={state}
      caption="Samples from N(0, 1) follow ż = a·tanh(2z) + c. The histogram shows where they end up. The curve is computed without samples: each grid point is moved by the ODE while its log-density falls at rate f′(z). The two agree. With a > 0 the field pushes mass away from 0 and splits the density in two; with a < 0 it squeezes the density into a spike."

      readouts={<Readout label="mass under the curve" value={formatNumber(result.mass)} />}
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
