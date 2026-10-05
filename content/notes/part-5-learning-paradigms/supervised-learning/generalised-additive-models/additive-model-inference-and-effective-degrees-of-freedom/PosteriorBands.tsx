import { useMemo } from 'react'
import {
  Button,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { logDet } from '../../regression/nonlinear-regression/_shared/splines'
import {
  cholesky,
  crossProduct,
  derivativePenalty,
  designMatrix,
  penalisedFit,
  times,
  uniformKnots,
} from '../_shared/core-smoothing'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const N = 100
const K = 17
const MAX_DRAWS = 50
const GRID = toFlat(linspace(0, 1, 121))
const LOG_LAMBDAS = toFlat(linspace(-9, 1, 51))
const truth = (x: number) => Math.sin(2 * Math.PI * x) * Math.exp(-x) + 0.5 * x

export function PosteriorBands() {
  const state = useFigureState({
    logLambda: float(-5, { min: -9, max: 1, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ' }),
    showDraws: setting(true, 'posterior draws'),
    count: int(20, { min: 1, max: MAX_DRAWS, step: 1, label: 'draws', format: (v) => String(v) }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    const eps = Array.from({ length: N }, () => normal(r))
    return { x, y: x.map((xi, i) => truth(xi) + state.noise * eps[i]) }
  }, [state.seed, state.noise])
  const t = useMemo(() => uniformKnots(K, 0, 1), [])
  const S = useMemo(() => derivativePenalty(t, 3, 2, 0, 1), [t])
  const B = useMemo(() => designMatrix(data.x, t), [data, t])
  const BtB = useMemo(() => crossProduct(B), [B])
  const gridB = useMemo(() => designMatrix(GRID, t), [t])
  const p = BtB.length

  // REML over a grid of λ, with σ² profiled out; the penalty's null space (straight lines) has dimension 2.
  const remlBest = useMemo(() => {
    let best = { score: Infinity, logLambda: 0 }
    for (const lg of LOG_LAMBDAS) {
      const lambda = 10 ** lg
      const f = penalisedFit(B, BtB, data.y, S, lambda)
      const A = BtB.map((row, j) => row.map((v, k) => v + lambda * S[j][k]))
      const score = (N - 2) * Math.log(f.rss + lambda * f.penalty) + logDet(A) - (p - 2) * Math.log(lambda)
      if (score < best.score) best = { score, logLambda: lg }
    }
    return best.logLambda
  }, [B, BtB, S, data, p])

  const fit = useMemo(() => penalisedFit(B, BtB, data.y, S, 10 ** state.logLambda), [B, BtB, S, data, state.logLambda])
  const sigma2 = fit.rss / (N - fit.edf)
  // Bayesian posterior covariance V_β = (BᵀB + λS)⁻¹ σ̂²; pointwise standard error of f̂(x) is √(b(x)ᵀ V_β b(x)).
  const curve = times(gridB, fit.coef)
  const se = gridB.map((b) => Math.sqrt(sigma2 * times(fit.inv, b).reduce((acc, v, j) => acc + v * b[j], 0)))
  const upper = curve.map((c, i) => c + 1.96 * se[i])
  const lower = curve.map((c, i) => c - 1.96 * se[i])
  const covered = GRID.filter((x, i) => Math.abs(truth(x) - curve[i]) <= 1.96 * se[i]).length / GRID.length

  const factor = useMemo(() => cholesky(fit.inv.map((row) => row.map((v) => v * sigma2))), [fit, sigma2])
  // Draw k uses its own stream, so raising the count adds curves without redrawing the earlier ones.
  const draws = useMemo(() => {
    const L = factor
    return Array.from({ length: state.count }, (_, k) => {
      const r = stream((1000 + state.seed) * 1000 + k)
      const z = Array.from({ length: p }, () => normal(r))
      const beta = fit.coef.map((c, i) => c + L[i].reduce((acc, v, j) => acc + v * z[j], 0))
      return times(gridB, beta)
    })
  }, [factor, fit, gridB, p, state.seed, state.count])

  const series: SeriesSpec[] = [
    ...(state.showDraws
      ? draws.map((y): SeriesSpec => ({
          name: 'posterior draws',
          type: 'line',
          x: GRID,
          y,
          slot: 3,
          thin: draws.length > 1,
        }))
      : []),
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    { name: 'true f', type: 'line', x: GRID, y: GRID.map(truth), slot: 2, dashed: true },
    { name: '95% band', type: 'line', x: GRID, y: upper, slot: 1 },
    { name: '95% band', type: 'line', x: GRID, y: lower, slot: 1 },
    { name: 'posterior mean f̂', type: 'line', x: GRID, y: curve, slot: 0 },
  ]

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'f(x)', range: [-2, 2.5] })
  return (
    <Figure
      title="Posterior bands and draws for a penalised spline"
      state={state}
      caption={
        <>
          A cubic spline with 20 B-splines and the penalty λ∫f″², fitted to 100 points. The band is f̂(x) ± 1.96 standard
          errors from the posterior covariance (BᵀB + λS)⁻¹σ̂², and the light curves are draws of f from that posterior;
          the draws slider sets how many. At small λ the band is wide and the draws wiggle; at large λ the band narrows
          around a straight line and misses the truth. The readout reports the fraction of the x grid where the band
          contains the true curve: its average over many data sets is close to 95% near the REML choice of λ.
        </>
      }
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => state.set('logLambda', remlBest)}>
            Set λ by REML
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="edf" value={formatNumber(fit.edf)} />
          <Readout label="σ̂" value={formatNumber(Math.sqrt(sigma2))} />
          <Readout label="REML log₁₀ λ" value={remlBest.toFixed(1)} />
          <Readout label="truth inside band" value={`${Math.round(100 * covered)}% of x`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
