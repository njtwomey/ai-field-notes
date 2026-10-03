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
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { autocorrelation, effectiveSampleSize, splitRhat } from '../../_shared/mcmc'

const PATH_SWEEPS = 25
/** Sample points drawn in all, shared evenly between the chains. */
const SHOWN = 1500
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
  const chains = useParam(4, { min: 1, max: 20, step: 1 })
  const rho = useParam(0.9, { min: -0.99, max: 0.99, step: 0.01 })
  const sweeps = useParam(1000, { min: 100, max: 5000, step: 100 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const sx = useParam(-3, { min: -R, max: R, step: 0.05 })
  const sy = useParam(3, { min: -R, max: R, step: 0.05 })
  const [blocked, setBlocked] = useState(false)

  // Each chain has its own random stream and starts at the common start point.
  const runs = useMemo(
    () =>
      Array.from({ length: chains.value }, (_, k) => {
        const g = rng(seed.value * 1000 + k)
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
        return { xs, ys, px, py }
      }),
    [chains.value, rho.value, sweeps.value, seed.value, sx.value, sy.value, blocked],
  )

  const stats = useMemo(() => {
    const half = Math.floor(sweeps.value / 2)
    return {
      lag1: runs.reduce((a, r) => a + (autocorrelation(r.xs, 1)[1] ?? 0), 0) / runs.length,
      ess: runs.reduce((a, r) => a + effectiveSampleSize(r.xs), 0),
      rhat: Math.max(splitRhat(runs.map((r) => r.xs.slice(half))), splitRhat(runs.map((r) => r.ys.slice(half)))),
    }
  }, [runs, sweeps.value])

  const series: XYSeries[] = useMemo(() => {
    const e1 = ellipse(rho.value, 1)
    const e2 = ellipse(rho.value, 2)
    const many = runs.length > 1
    // About SHOWN sample points in all, taken evenly from every chain.
    const perChain = Math.max(1, Math.floor(SHOWN / runs.length))
    const stride = Math.max(1, Math.ceil(sweeps.value / perChain))
    const sxs: number[] = []
    const sys: number[] = []
    for (const r of runs)
      for (let i = 0; i < r.xs.length; i += stride) {
        sxs.push(r.xs[i])
        sys.push(r.ys[i])
      }
    return [
      { name: 'samples', type: 'scatter', x: sxs, y: sys, muted: true },
      { name: '1 standard deviation', type: 'line', x: e1.x, y: e1.y, slot: 0 },
      { name: '2 standard deviations', type: 'line', x: e2.x, y: e2.y, slot: 0, dashed: true },
      ...runs.map((r): XYSeries => ({
        name: many ? `paths, first ${PATH_SWEEPS} sweeps` : `path, first ${PATH_SWEEPS} sweeps`,
        type: 'line',
        x: r.px,
        y: r.py,
        slot: 1,
        thin: many,
      })),
    ]
  }, [runs, rho.value, sweeps.value])

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
      caption="The target is a bivariate Gaussian with unit variances and correlation ρ. Each sweep draws x₁ from its conditional given x₂, then x₂ given x₁, so the path moves in axis-parallel steps. Several chains run from the same start, each with its own random stream; the light lines are their first 25 sweeps, and the chains slider sets how many run. When |ρ| is near 1 the conditionals are narrow and the steps are short compared with the length of the ellipse, so the chain creeps along it. Drag the start point. The effective sample size is summed over chains and compared with chains × sweeps/τ; split R̂ uses the second half of every chain. The blocked sampler draws both coordinates jointly and gives independent samples."
      controls={
        <>
          <ParamSlider label="correlation ρ" param={rho} />
          <ParamSlider label="chains" param={chains} format={(v) => String(v)} withArrows />
          <ParamSlider label="sweeps" param={sweeps} format={(v) => String(v)} withArrows />
          <ParamSlider label="random seed" param={seed} withArrows />
          <ParamSwitch label="block both coordinates" checked={blocked} onChange={setBlocked} />
        </>
      }
      readout={
        <>
          <Readout label="lag-1 autocorrelation of x₁ (chain mean)" value={formatNumber(stats.lag1)} />
          <Readout label="theory" value={formatNumber(blocked ? 0 : rho.value ** 2)} />
          <Readout label="τ in theory" value={formatNumber(tau)} />
          <Readout label="ESS of x₁" value={formatNumber(stats.ess)} />
          <Readout label="chains × sweeps / τ" value={formatNumber((chains.value * sweeps.value) / tau)} />
          <Readout label="split R̂ (worse coordinate)" value={formatNumber(stats.rhat)} />
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
