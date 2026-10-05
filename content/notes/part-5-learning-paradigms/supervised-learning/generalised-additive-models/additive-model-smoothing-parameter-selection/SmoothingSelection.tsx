import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { evaluate, logDet, makeBasis, penalise, smooth } from '../../regression/nonlinear-regression/_shared/splines'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const N = 100
const KNOTS = Array.from({ length: 16 }, (_, i) => (i + 1) / 17)
const LOG_LAMBDAS = toFlat(linspace(-8, 1, 37))
const GRID = toFlat(linspace(0, 1, 101))
const truth = (x: number) => Math.sin(2 * Math.PI * x)

/** Rescale a criterion to [0, 1] over the grid so that GCV and REML share one axis. */
const unit = (v: number[]) => {
  const lo = Math.min(...v)
  const hi = Math.max(...v)
  return v.map((x) => (x - lo) / (hi - lo || 1))
}

export function SmoothingSelection() {
  const state = useFigureState({
    logLambda: float(-3, { min: -8, max: 1, step: 0.05, label: 'log₁₀ λ', format: (v) => v.toFixed(2) }),
    noise: float(0.3, { min: 0.05, max: 1.5, step: 0.05, label: 'noise σ' }),
    seed: int(5, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    const eps = Array.from({ length: N }, () => normal(r))
    return { x, y: x.map((xi, i) => truth(xi) + state.noise * eps[i]) }
  }, [state.seed, state.noise])
  const basis = useMemo(() => makeBasis(data.x, KNOTS, 0, 1), [data])
  const p = basis.BtB.length

  // Both criteria over the λ grid. The penalty's null space (straight lines) has dimension 2.
  const curves = useMemo(() => {
    const gcv: number[] = []
    const reml: number[] = []
    for (const lg of LOG_LAMBDAS) {
      const lambda = 10 ** lg
      const s = penalise(basis, lambda)
      const f = smooth(s, data.y)
      const rss = data.y.reduce((acc, y, i) => acc + (y - f.fitted[i]) ** 2, 0)
      const pen =
        lambda * f.coef.reduce((acc, c, j) => acc + c * f.coef.reduce((q, d, k) => q + basis.omega[j][k] * d, 0), 0)
      gcv.push((N * rss) / (N - s.edf) ** 2)
      reml.push((N - 2) * Math.log(rss + pen) + logDet(s.A) - (p - 2) * Math.log(lambda))
    }
    const best = (v: number[]) => LOG_LAMBDAS[v.indexOf(Math.min(...v))]
    return { gcv: unit(gcv), reml: unit(reml), gcvBest: best(gcv), remlBest: best(reml) }
  }, [basis, data, p])

  const smoother = useMemo(() => penalise(basis, 10 ** state.logLambda), [basis, state.logLambda])
  const fit = useMemo(() => smooth(smoother, data.y), [smoother, data])

  const criteria = [
    { name: 'GCV', x: LOG_LAMBDAS, y: curves.gcv, slot: 0 },
    { name: 'REML', x: LOG_LAMBDAS, y: curves.reml, slot: 1 },
  ] as const
  const fitSeries = [
    { name: 'data', x: data.x, y: data.y, muted: true },
    { name: 'true curve', x: GRID, y: GRID.map(truth), slot: 2, dashed: true },
    { name: 'fit at chosen λ', x: GRID, y: evaluate(smoother, fit.coef, GRID), slot: 3 },
  ] as const

  const xAxis = useAxis({ label: 'log₁₀ λ', range: [-8, 1] })
  const yAxis = useAxis({ label: 'criterion (rescaled)', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'x', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'y', range: [-3, 3] })
  return (
    <Figure
      title="Choosing λ by GCV and REML"
      state={state}
      caption={
        <>
          A penalised cubic spline with 20 basis functions fitted to 100 noisy points. The left panel plots both
          criteria against log₁₀ λ, each rescaled to run from 0 at its minimum to 1. Drag the vertical line, or use the
          slider, to fit at any λ. Draw new samples: the GCV choice jumps around more than the REML choice, and now and
          then lands on a much smaller λ that follows the noise.
        </>
      }

      readouts={
        <>
          <Readout label="effective df" value={formatNumber(smoother.edf)} />
          <Readout label="GCV best log₁₀ λ" value={curves.gcvBest.toFixed(2)} />
          <Readout label="REML best log₁₀ λ" value={curves.remlBest.toFixed(2)} />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...criteria[0]} />
          <Curve {...criteria[1]} />
          <Handle {...state.handle('logLambda', { label: 'λ' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={260}>
          <Points {...fitSeries[0]} />
          <Curve {...fitSeries[1]} />
          <Curve {...fitSeries[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
