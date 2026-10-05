import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { autocorrelation, effectiveSampleSize, splitRhat } from '../../_shared/mcmc'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const PATH_SWEEPS = 25
/** Sample points drawn in all, shared evenly between the chains. */
const SHOWN = 1500
const ANGLES = toFlat(linspace(0, 2 * Math.PI, 121))
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
  const state = useFigureState({
    rho: slider(-0.99, 0.99, 0.9, { step: 0.01, label: 'correlation ρ' }),
    chains: int(4, { min: 1, max: 20, step: 1, label: 'chains', format: (v) => String(v) }),
    sweeps: int(1000, { min: 100, max: 5000, step: 100, label: 'sweeps', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed' }),
    blocked: setting(false, 'block both coordinates'),
    sx: slider(-R, R, -3, { step: 0.05, onChart: true }),
    sy: slider(-R, R, 3, { step: 0.05, onChart: true }),
  })

  // Each chain has its own random stream and starts at the common start point.
  const runs = useMemo(
    () =>
      Array.from({ length: state.chains }, (_, k) => {
        const g = stream(state.seed * 1000 + k)
        const r = state.rho
        const sd = Math.sqrt(1 - r * r)
        let x = state.sx
        let y = state.sy
        const xs: number[] = []
        const ys: number[] = []
        const px = [x]
        const py = [y]
        for (let t = 0; t < state.sweeps; t++) {
          if (state.blocked) {
            // Joint draw: x₁ ~ N(0, 1), then x₂ | x₁. Independent of the previous state.
            x = normal(g)
            y = r * x + sd * normal(g)
            if (t < PATH_SWEEPS) {
              px.push(x)
              py.push(y)
            }
          } else {
            x = r * y + sd * normal(g)
            if (t < PATH_SWEEPS) {
              px.push(x)
              py.push(y)
            }
            y = r * x + sd * normal(g)
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
    [state.chains, state.rho, state.sweeps, state.seed, state.sx, state.sy, state.blocked],
  )

  const stats = useMemo(() => {
    const half = Math.floor(state.sweeps / 2)
    return {
      lag1: runs.reduce((a, r) => a + (autocorrelation(r.xs, 1)[1] ?? 0), 0) / runs.length,
      ess: runs.reduce((a, r) => a + effectiveSampleSize(r.xs), 0),
      rhat: Math.max(splitRhat(runs.map((r) => r.xs.slice(half))), splitRhat(runs.map((r) => r.ys.slice(half)))),
    }
  }, [runs, state.sweeps])

  const series: SeriesSpec[] = useMemo(() => {
    const e1 = ellipse(state.rho, 1)
    const e2 = ellipse(state.rho, 2)
    const many = runs.length > 1
    // About SHOWN sample points in all, taken evenly from every chain.
    const perChain = Math.max(1, Math.floor(SHOWN / runs.length))
    const stride = Math.max(1, Math.ceil(state.sweeps / perChain))
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
      ...runs.map((r): SeriesSpec => ({
        name: many ? `paths, first ${PATH_SWEEPS} sweeps` : `path, first ${PATH_SWEEPS} sweeps`,
        type: 'line',
        x: r.px,
        y: r.py,
        slot: 1,
        thin: many,
      })),
    ]
  }, [runs, state.rho, state.sweeps])

  const tau = state.blocked ? 1 : (1 + state.rho ** 2) / (1 - state.rho ** 2)

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Gibbs sampling a correlated Gaussian"
      state={state}
      caption="The target is a bivariate Gaussian with unit variances and correlation ρ. Each sweep draws x₁ from its conditional given x₂, then x₂ given x₁, so the path moves in axis-parallel steps. Several chains run from the same start, each with its own random stream; the light lines are their first 25 sweeps, and the chains slider sets how many run. When |ρ| is near 1 the conditionals are narrow and the steps are short compared with the length of the ellipse, so the chain creeps along it. Drag the start point. The effective sample size is summed over chains and compared with chains × sweeps/τ; split R̂ uses the second half of every chain. The blocked sampler draws both coordinates jointly and gives independent samples."

      readouts={
        <>
          <Readout label="lag-1 autocorrelation of x₁ (chain mean)" value={formatNumber(stats.lag1)} />
          <Readout label="theory" value={formatNumber(state.blocked ? 0 : state.rho ** 2)} />
          <Readout label="τ in theory" value={formatNumber(tau)} />
          <Readout label="ESS of x₁" value={formatNumber(stats.ess)} />
          <Readout label="chains × sweeps / τ" value={formatNumber((state.chains * state.sweeps) / tau)} />
          <Readout label="split R̂ (worse coordinate)" value={formatNumber(stats.rhat)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
        <Handle
          kind="point"
          at={[state.sx, state.sy]}
          label="start"
          onDrag={([x, y]) => {
            state.set('sx', x)
            state.set('sy', y)
          }}
        />
      </Plot>
    </Figure>
  )
}
