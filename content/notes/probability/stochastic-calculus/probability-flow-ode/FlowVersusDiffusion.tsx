import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { T_END, histogram, joinPaths, normals, reverseParticles, vpMixture } from '../_shared/sde'

const STEPS = 250
const PARTICLES = 2000
const SHOWN = 12
const H = T_END / STEPS
const XS = linspace(-4.5, 4.5, 181)

/**
 * The reverse-time SDE and the probability-flow ODE for the same variance-preserving diffusion, started from the same
 * N(0, 1) draws at t = 5 and driven by the exact score. Paths differ (rough against smooth); marginals agree.
 */
export function FlowVersusDiffusion() {
  const t = useParam(0, { min: 0, max: T_END, step: H * 5 })
  const start = useMemo(() => normals(PARTICLES, 41), [])
  const sde = useMemo(() => reverseParticles('sde', start, STEPS, 42), [start])
  const ode = useMemo(() => reverseParticles('ode', start, STEPS, 42), [start])

  const pathSeries = useMemo<XYSeries[]>(() => {
    const times = sde.map((_, k) => T_END - k * H)
    const paths = (states: Float64Array[]) =>
      joinPaths(Array.from({ length: SHOWN }, (_, i) => ({ x: times, y: states.map((s) => s[i]) })))
    return [
      { name: 'reverse SDE', type: 'line', ...paths(sde), slot: 0 },
      { name: 'probability-flow ODE', type: 'line', ...paths(ode), slot: 1 },
    ]
  }, [sde, ode])

  const k = Math.round((T_END - t.value) / H)
  const densitySeries = useMemo<XYSeries[]>(() => {
    const hs = histogram(sde[k], -4.5, 4.5, 60)
    const ho = histogram(ode[k], -4.5, 4.5, 60)
    return [
      { name: 'SDE particles', type: 'bar', x: hs.x, y: hs.y, slot: 0 },
      { name: 'ODE particles', type: 'line', x: ho.x, y: ho.y, slot: 1 },
      {
        name: 'exact p_t',
        type: 'line',
        x: XS,
        y: XS.map((x) => vpMixture(x, t.value).p),
        emphasis: true,
        dashed: true,
      },
    ]
  }, [sde, ode, k, t.value])

  const share = (xs: Float64Array) => xs.reduce((s, x) => s + (x > -0.25 ? 1 : 0), 0) / xs.length
  return (
    <Interactive
      title="Same marginals, different paths"
      caption="Twelve particles start from the same N(0, 1) draws at t = 5 and move back to t = 0 with the exact score of the noised two-mode density. The reverse-time SDE (drift f − g²∇log p_t plus noise) gives rough paths that cross one another; the probability-flow ODE (drift f − ½g²∇log p_t, no noise) gives smooth paths that never cross, so each starting point is mapped to one data point. The lower panel shows that 2,000 particles of each kind have the same density p_t at every time. Drag the vertical line to move in time."
      controls={<ParamSlider label="time t" param={t} withArrows />}
      readout={
        <>
          <Readout label="SDE share right of −0.25" value={formatNumber(share(sde[k]))} />
          <Readout label="ODE share right of −0.25" value={formatNumber(share(ode[k]))} />
        </>
      }
    >
      <XYChart
        height={260}
        xLabel="t"
        yLabel="x"
        series={pathSeries}
        xRange={[0, T_END]}
        yRange={[-4.5, 4.5]}
        handles={[{ kind: 'x', at: t.value, label: 't', onDrag: (x) => t.set(x) }]}
      />
      <XYChart height={200} xLabel="x" yLabel="density" series={densitySeries} xRange={[-4.5, 4.5]} yRange={[0, 0.9]} />
    </Interactive>
  )
}
