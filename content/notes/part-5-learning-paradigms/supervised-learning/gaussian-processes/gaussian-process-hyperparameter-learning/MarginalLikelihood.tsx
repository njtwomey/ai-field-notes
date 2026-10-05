import { useMemo } from 'react'
import {
  Button,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { addDiagonal, cholesky, gram, logMarginal, makeKernel, posterior, gridMaximum } from '../_shared/gp'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const N = 20
const TRUE_ELL = 1
const TRUE_NOISE = 0.25
const LOG_ELL = toFlat(linspace(-1, 1.3, 40))
const LOG_NOISE = toFlat(linspace(-2, 0.2, 40))
const GRID = toFlat(linspace(-5, 5, 121))
const X_RANGE: [number, number] = [-5, 5]
const Y_RANGE: [number | undefined, number | undefined] = [-3, 3]
/** Colour only the top 30 nats, so the ridge and its maxima stand out from the steep walls. */
const DEPTH = 30

/** A draw from a GP with squared exponential covariance, σ_f = 1 and ℓ = 1, observed with noise sd 0.25. */
function makeData(seed: number) {
  const g = stream(seed)
  const x = Array.from({ length: N }, () => -4.5 + 9 * uniform(g)).sort((a, b) => a - b)
  const k = makeKernel('se', { ell: TRUE_ELL, sf: 1 })
  const L = cholesky(addDiagonal(gram(k, x, x), 1e-8))
  const z = x.map(() => normal(g))
  const y = x.map((_, i) => L[i].reduce((s, v, j) => s + v * z[j], 0) + TRUE_NOISE * normal(g))
  return { x, y }
}

/** The log marginal likelihood over length-scale and noise level, with the fit at the chosen point. */
export function MarginalLikelihood() {
  const fmtPow = (v: number) => formatNumber(10 ** v)
  const state = useFigureState({
    logEll: slider(-1, 1.3, -0.6, { step: 0.01, label: 'length-scale ℓ', format: fmtPow }),
    logNoise: slider(-2, 0.2, -1.5, { step: 0.01, label: 'noise sd σ_n', format: fmtPow }),
    seed: int(4, { min: 1, max: 20, label: 'data seed' }),
  })
  const seed = { value: state.seed }
  const logEll = { value: state.logEll, set: (v: number) => state.set('logEll', v) }
  const logNoise = { value: state.logNoise, set: (v: number) => state.set('logNoise', v) }
  const data = useMemo(() => makeData(seed.value), [seed.value])

  const surface = useMemo(() => {
    const z = LOG_NOISE.map((ln) =>
      LOG_ELL.map(
        (le) => logMarginal(makeKernel('se', { ell: 10 ** le, sf: 1 }), data.x, data.y, 10 ** (2 * ln)).value,
      ),
    )
    const best = gridMaximum(z)
    const range: [number, number] = [best.value - DEPTH, best.value]
    return { z: z.map((row) => row.map((v) => Math.max(v, range[0]))), best, range }
  }, [data])

  const fit = useMemo(() => {
    const k = makeKernel('se', { ell: 10 ** logEll.value, sf: 1 })
    const noiseVar = 10 ** (2 * logNoise.value)
    const post = posterior(k, data.x, data.y, noiseVar, GRID)
    return { post, lml: logMarginal(k, data.x, data.y, noiseVar) }
  }, [data, logEll.value, logNoise.value])

  const sd = fit.post.variance.map(Math.sqrt)
  const series = [
    {
      name: 'mean + 2 sd',
      x: GRID,
      y: fit.post.mean.map((m, i) => m + 2 * sd[i]),
      slot: 0,
      dashed: true,
    },
    {
      name: 'mean − 2 sd',
      x: GRID,
      y: fit.post.mean.map((m, i) => m - 2 * sd[i]),
      slot: 0,
      dashed: true,
    },
    { name: 'posterior mean', x: GRID, y: fit.post.mean, slot: 0 },
    { name: 'data', x: data.x, y: data.y, slot: 1 },
  ] as const
  const toBest = () => {
    logEll.set(LOG_ELL[surface.best.j])
    logNoise.set(LOG_NOISE[surface.best.i])
  }

  const xAxis = useAxis({ label: 'log₁₀ ℓ', hold: 'union' })
  const yAxis = useAxis({ label: 'log₁₀ σ_n', hold: 'union' })
  const xAxis2 = useAxis({ label: 'x', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'y', range: Y_RANGE })
  return (
    <Figure
      title="The marginal likelihood as a function of length-scale and noise"
      state={state}
      caption="Left: ln p(y | ℓ, σ_n) for 20 points drawn from a GP with ℓ = 1 and noise sd 0.25, over log₁₀ ℓ (horizontal) and log₁₀ σ_n (vertical), with σ_f = 1. Drag the point to choose the hyperparameters; the right panel shows the resulting posterior. The bottom left explains the data as a rapidly varying function with no noise; the right side explains it as a flat function with large noise. The marginal likelihood prefers the setting in between, where the data-fit and complexity terms balance. Some seeds give two local maxima, one for each explanation."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={toBest}>
            Go to the maximum
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="ln p(y | θ)" value={formatNumber(fit.lml.value)} />
          <Readout label="data fit −½yᵀK⁻¹y" value={formatNumber(fit.lml.dataFit)} />
          <Readout label="complexity −½ln|K|" value={formatNumber(fit.lml.complexity)} />
          <Readout
            label="grid maximum at ℓ, σ_n"
            value={`${fmtPow(LOG_ELL[surface.best.j])}, ${fmtPow(LOG_NOISE[surface.best.i])}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster
            x={LOG_ELL}
            y={LOG_NOISE}
            z={surface.z}
            scale={'sequential'}
            range={surface.range}
            valueLabel={'ln p(y | θ)'}
          />
          <Handle {...state.handle(['logEll', 'logNoise'], { label: '(ℓ, σ_n)' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          <Points {...series[3]} />
        </Plot>
      </div>
    </Figure>
  )
}
