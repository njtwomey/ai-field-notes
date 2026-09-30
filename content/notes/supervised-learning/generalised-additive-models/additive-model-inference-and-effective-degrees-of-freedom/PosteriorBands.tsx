import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
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

const N = 100
const K = 17
const MAX_DRAWS = 50
const GRID = linspace(0, 1, 121)
const LOG_LAMBDAS = linspace(-9, 1, 51)
const truth = (x: number) => Math.sin(2 * Math.PI * x) * Math.exp(-x) + 0.5 * x

export function PosteriorBands() {
  const [logLambda, setLogLambda] = useState(-5)
  const [noise, setNoise] = useState(0.3)
  const [seed, setSeed] = useState(1)
  const [showDraws, setShowDraws] = useState(true)
  const count = useParam(20, { min: 1, max: MAX_DRAWS, step: 1 })

  const data = useMemo(() => {
    const r = rng(seed)
    const x = Array.from({ length: N }, () => r.uniform())
    const eps = Array.from({ length: N }, () => r.normal())
    return { x, y: x.map((xi, i) => truth(xi) + noise * eps[i]) }
  }, [seed, noise])
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

  const fit = useMemo(() => penalisedFit(B, BtB, data.y, S, 10 ** logLambda), [B, BtB, S, data, logLambda])
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
    return Array.from({ length: count.value }, (_, k) => {
      const r = rng((1000 + seed) * 1000 + k)
      const z = Array.from({ length: p }, () => r.normal())
      const beta = fit.coef.map((c, i) => c + L[i].reduce((acc, v, j) => acc + v * z[j], 0))
      return times(gridB, beta)
    })
  }, [factor, fit, gridB, p, seed, count.value])

  const series: XYSeries[] = [
    ...(showDraws
      ? draws.map((y): XYSeries => ({
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

  return (
    <Interactive
      title="Posterior bands and draws for a penalised spline"
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
          <ParamSlider
            label="log₁₀ λ"
            value={logLambda}
            onChange={setLogLambda}
            min={-9}
            max={1}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamSlider label="noise σ" value={noise} onChange={setNoise} min={0.05} max={1} step={0.05} />
          <ParamSwitch label="posterior draws" checked={showDraws} onChange={setShowDraws} />
          <ParamSlider label="draws" param={count} withArrows format={(v) => String(v)} />
          <ParamButton onClick={() => setLogLambda(remlBest)}>Set λ by REML</ParamButton>
          <ParamButton onClick={() => setSeed((v) => v + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="edf" value={formatNumber(fit.edf)} />
          <Readout label="σ̂" value={formatNumber(Math.sqrt(sigma2))} />
          <Readout label="REML log₁₀ λ" value={remlBest.toFixed(1)} />
          <Readout label="truth inside band" value={`${Math.round(100 * covered)}% of x`} />
        </>
      }
    >
      <XYChart series={series} xRange={[0, 1]} yRange={[-2, 2.5]} xLabel="x" yLabel="f(x)" height={340} />
    </Interactive>
  )
}
