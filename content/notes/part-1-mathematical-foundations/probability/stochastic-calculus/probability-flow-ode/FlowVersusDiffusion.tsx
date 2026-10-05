import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { T_END, histogram, joinPaths, normals, reverseParticles, vpMixture } from '../_shared/sde'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const STEPS = 250
const PARTICLES = 2000
const H = T_END / STEPS
const XS = toFlat(linspace(-4.5, 4.5, 181))

/**
 * The reverse-time SDE and the probability-flow ODE for the same variance-preserving diffusion, started from the same
 * N(0, 1) draws at t = 5 and driven by the exact score. Paths differ (rough against smooth); marginals agree.
 */
export function FlowVersusDiffusion() {
  const state = useFigureState({
    t: slider(0, T_END, 0, { step: H * 5, label: 'time t' }),
    count: int(12, { min: 1, max: 50, step: 1, label: 'paths', format: (v) => String(v) }),
  })
  const start = useMemo(() => normals(PARTICLES, 41), [])
  const sde = useMemo(() => reverseParticles('sde', start, STEPS, 42), [start])
  const ode = useMemo(() => reverseParticles('ode', start, STEPS, 42), [start])

  const pathSeries = useMemo<SeriesSpec[]>(() => {
    const times = sde.map((_, k) => T_END - k * H)
    // The drawn paths are the first particles of the 2,000, so raising the count adds paths and keeps the others.
    const paths = (states: Float64Array[]) =>
      joinPaths(Array.from({ length: state.count }, (_, i) => ({ x: times, y: states.map((s) => s[i]) })))
    // The ODE paths are not random, but many of them still form a solid fan, so they are drawn light as well.
    const thin = state.count > 1
    return [
      { name: 'reverse SDE', type: 'line', ...paths(sde), slot: 0, thin },
      { name: 'probability-flow ODE', type: 'line', ...paths(ode), slot: 1, thin },
    ]
  }, [sde, ode, state.count])

  const k = Math.round((T_END - state.t) / H)
  const densitySeries = useMemo(() => {
    const hs = histogram(sde[k], -4.5, 4.5, 60)
    const ho = histogram(ode[k], -4.5, 4.5, 60)
    return [
      { name: 'SDE particles', x: hs.x, y: hs.y, slot: 0 },
      { name: 'ODE particles', x: ho.x, y: ho.y, slot: 1 },
      {
        name: 'exact p_t',
        x: XS,
        y: XS.map((x) => vpMixture(x, state.t).p),
        emphasis: true,
        dashed: true,
      },
    ] as const
  }, [sde, ode, k, state.t])

  const share = (xs: Float64Array) => xs.reduce((s, x) => s + (x > -0.25 ? 1 : 0), 0) / xs.length
  const xAxis = useAxis({ label: 't', range: [0, T_END] })
  const yAxis = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const xAxis2 = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const yAxis2 = useAxis({ label: 'density', range: [0, 0.9] })
  return (
    <Figure
      title="Same marginals, different paths"
      state={state}
      caption="Particles start from the same N(0, 1) draws at t = 5 and move back to t = 0 with the exact score of the noised two-mode density. The reverse-time SDE (drift f − g²∇log p_t plus noise) gives rough paths that cross one another; the probability-flow ODE (drift f − ½g²∇log p_t, no noise) gives smooth paths that never cross, so each starting point is mapped to one data point. The paths slider sets how many particles of each kind are drawn, as light lines when there are several. The lower panel shows that 2,000 particles of each kind have the same density p_t at every time. Drag the vertical line to move in time."

      readouts={
        <>
          <Readout label="SDE share right of −0.25" value={formatNumber(share(sde[k]))} />
          <Readout label="ODE share right of −0.25" value={formatNumber(share(ode[k]))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        {seriesLayers(pathSeries)}
        <Handle {...state.handle('t', { label: 't' })} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={200}>
        <Bars {...densitySeries[0]} />
        <Curve {...densitySeries[1]} />
        <Curve {...densitySeries[2]} />
      </Plot>
    </Figure>
  )
}
