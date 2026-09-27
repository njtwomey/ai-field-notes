import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { autocorrelation, effectiveSampleSize } from '../../_shared/mcmc'

const PATH_SWEEPS = 25
const ANGLES = linspace(0, 2 * Math.PI, 121)
const R = 3.6

/** Points on the ellipse x^T Σ^{-1} x = c² for Σ = [[1, ρ], [ρ, 1]], via the eigenvectors (1, ±1)/√2. */
function ellipse(rho: number, c: number) {
  const a = c * Math.sqrt(1 + rho)
  const b = c * Math.sqrt(1 - rho)
  return {
    x: ANGLES.map((t) => (a * Math.cos(t) - b * Math.sin(t)) / Math.SQRT2),
    y: ANGLES.map((t) => (a * Math.cos(t) + b * Math.sin(t)) / Math.SQRT2),
  }
}

/**
 * Gibbs sampling a standard bivariate Gaussian with correlation ρ. Each coordinate is drawn from its full conditional
 * N(ρ × other, 1 − ρ²). The x₁ chain is then AR(1) with coefficient ρ², so τ = (1 + ρ²)/(1 − ρ²).
 */
export function GibbsGaussian() {
  const rho = useParam(0.9, { min: -0.99, max: 0.99, step: 0.01 })
  const sweeps = useParam(1000, { min: 100, max: 5000, step: 100 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const sx = useParam(-3, { min: -R, max: R, step: 0.05 })
  const sy = useParam(3, { min: -R, max: R, step: 0.05 })
  const [blocked, setBlocked] = useState(false)

  const run = useMemo(() => {
    const g = rng(seed.value)
    const r = rho.value
    const sd = Math.sqrt(1 - r * r)
    let x = sx.value
    let y = sy.value
    const xs: number[] = []
    const ys: number[] = []
    const px = [x]
    const py = [y]
    for (let t = 0; t < sweeps.value; t++) {
      if (blocked) {
        // Joint draw: x₁ ~ N(0, 1), then x₂ | x₁. Independent of the previous state.
        x = g.normal()
        y = r * x + sd * g.normal()
        if (t < PATH_SWEEPS) {
          px.push(x)
          py.push(y)
        }
      } else {
        x = r * y + sd * g.normal()
        if (t < PATH_SWEEPS) {
          px.push(x)
          py.push(y)
        }
        y = r * x + sd * g.normal()
        if (t < PATH_SWEEPS) {
          px.push(x)
          py.push(y)
        }
      }
      xs.push(x)
      ys.push(y)
    }
    const lag1 = autocorrelation(xs, 1)[1] ?? 0
    return { xs, ys, px, py, lag1, ess: effectiveSampleSize(xs) }
  }, [rho.value, sweeps.value, seed.value, sx.value, sy.value, blocked])

  const series: XYSeries[] = useMemo(() => {
    const e1 = ellipse(rho.value, 1)
    const e2 = ellipse(rho.value, 2)
    return [
      { name: 'samples', type: 'scatter', x: run.xs.slice(0, 1500), y: run.ys.slice(0, 1500), muted: true },
      { name: '1 standard deviation', type: 'line', x: e1.x, y: e1.y, slot: 0 },
      { name: '2 standard deviations', type: 'line', x: e2.x, y: e2.y, slot: 0, dashed: true },
      { name: `path, first ${PATH_SWEEPS} sweeps`, type: 'line', x: run.px, y: run.py, slot: 1 },
    ]
  }, [run, rho.value])

  const tau = blocked ? 1 : (1 + rho.value ** 2) / (1 - rho.value ** 2)
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [sx.value, sy.value],
      label: 'start',
      onDrag: ([x, y]) => {
        sx.set(x)
        sy.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="Gibbs sampling a correlated Gaussian"
      caption="The target is a bivariate Gaussian with unit variances and correlation ρ. Each sweep draws x₁ from its conditional given x₂, then x₂ given x₁, so the path moves in axis-parallel steps. When |ρ| is near 1 the conditionals are narrow and the steps are short compared with the length of the ellipse, so the chain creeps along it. Drag the start point. The blocked sampler draws both coordinates jointly and gives independent samples."
      controls={
        <>
          <ParamSlider label="correlation ρ" param={rho} />
          <ParamSlider label="sweeps" param={sweeps} format={(v) => String(v)} withArrows />
          <ParamSlider label="random seed" param={seed} withArrows />
          <ParamSwitch label="block both coordinates" checked={blocked} onChange={setBlocked} />
        </>
      }
      readout={
        <>
          <Readout label="lag-1 autocorrelation of x₁" value={formatNumber(run.lag1)} />
          <Readout label="theory" value={formatNumber(blocked ? 0 : rho.value ** 2)} />
          <Readout label="τ in theory" value={formatNumber(tau)} />
          <Readout label="ESS of x₁" value={formatNumber(run.ess)} />
          <Readout label="sweeps / τ" value={formatNumber(sweeps.value / tau)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="x₁"
        yLabel="x₂"
        xRange={[-R, R]}
        yRange={[-R, R]}
        equalAspect
        handles={handles}
      />
    </Interactive>
  )
}
