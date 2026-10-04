import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { erf, logGamma } from 'aifn/numerics/special'

const GRID = toFlat(linspace(0.0005, 0.9995, 600))
const erfc = (x: number) => 1 - erf(x)
const logBeta = (a: number, b: number) => logGamma(a) + logGamma(b) - logGamma(a + b)

/**
 * A coin's exact beta posterior against its Laplace approximation, a Gaussian at the MAP estimate with variance equal to
 * the inverse curvature of the negative log posterior there.
 */
export function LaplacePosterior() {
  const state = useFigureState({
    heads: int(7, { min: 0, max: 200, step: 1, label: 'heads' }),
    tails: int(3, { min: 0, max: 200, step: 1, label: 'tails' }),
    alpha: float(1, { min: 0.5, max: 10, step: 0.5, label: 'prior α' }),
    beta: float(1, { min: 0.5, max: 10, step: 0.5, label: 'prior β' }),
  })

  const r = useMemo(() => {
    const a = state.alpha + state.heads
    const b = state.beta + state.tails
    const exact = GRID.map((t) => Math.exp((a - 1) * Math.log(t) + (b - 1) * Math.log1p(-t) - logBeta(a, b)))
    const mean = a / (a + b)
    const sd = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
    // The mode is interior only when both exponents are positive; otherwise the maximum sits on the boundary.
    if (a <= 1 || b <= 1) return { exact, mean, sd, laplace: null }
    const mode = (a - 1) / (a + b - 2)
    const curvature = (a - 1) / mode ** 2 + (b - 1) / (1 - mode) ** 2
    const lsd = 1 / Math.sqrt(curvature)
    const gauss = GRID.map((t) => Math.exp(-0.5 * ((t - mode) / lsd) ** 2) / (lsd * Math.sqrt(2 * Math.PI)))
    // Evidence p(D) = B(a, b) / B(α, β), exactly and by Laplace: ℓ(mode) + ½ log(2π σ²) − log B(α, β).
    const logEvidence = logBeta(a, b) - logBeta(state.alpha, state.beta)
    const logLaplace =
      (a - 1) * Math.log(mode) +
      (b - 1) * Math.log1p(-mode) +
      0.5 * Math.log(2 * Math.PI * lsd * lsd) -
      logBeta(state.alpha, state.beta)
    const massOutside = 0.5 * (erfc(mode / (lsd * Math.SQRT2)) + erfc((1 - mode) / (lsd * Math.SQRT2)))
    return { exact, mean, sd, laplace: { gauss, mode, lsd, ratio: Math.exp(logLaplace - logEvidence), massOutside } }
  }, [state.heads, state.tails, state.alpha, state.beta])

  const series: SeriesSpec[] = [
    { name: 'exact posterior', type: 'line', x: GRID, y: r.exact, slot: 0, area: true },
    ...(r.laplace
      ? [{ name: 'Laplace approximation', type: 'line' as const, x: GRID, y: r.laplace.gauss, slot: 1, dashed: true }]
      : []),
  ]

  const xAxis = useAxis({ label: 'θ', range: [0, 1] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Exact posterior and its Laplace approximation"
      state={state}
      caption="A coin's bias θ after the given heads and tails, under a beta prior. The shaded curve is the exact beta posterior; the dashed curve is the Gaussian at its mode with variance from the curvature there. With few flips near 0 or 1 the posterior is skewed and the Gaussian spills outside [0, 1]; as the flips grow the two curves merge and the evidence ratio approaches 1. With no heads or no tails and a flat prior, the mode is on the boundary and there is no Laplace approximation."

      readouts={
        <>
          <Readout label="exact mean, sd" value={`${formatNumber(r.mean)}, ${formatNumber(r.sd)}`} />
          <Readout
            label="Laplace mean, sd"
            value={r.laplace ? `${formatNumber(r.laplace.mode)}, ${formatNumber(r.laplace.lsd)}` : 'mode on boundary'}
          />
          <Readout label="evidence, Laplace / exact" value={r.laplace ? formatNumber(r.laplace.ratio) : '—'} />
          <Readout
            label="Laplace mass outside [0, 1]"
            value={r.laplace ? `${(100 * r.laplace.massOutside).toFixed(2)}%` : '—'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
