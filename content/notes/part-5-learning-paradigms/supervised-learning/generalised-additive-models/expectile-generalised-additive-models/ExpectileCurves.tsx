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
import {
  addScaled,
  cholesky,
  cholSolve,
  crossprod,
  crossprodY,
  diffMatrix,
  dot,
  gram,
  psplineRow,
  type Matrix,
} from '../_shared/terms-psplines'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'
import { normalCdf, normalPdf } from 'aifn/numerics/special'

const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

const N = 300
const K = 15
const GRID = toFlat(linspace(0, 1, 121))
const BACKGROUND = [0.05, 0.25, 0.5, 0.75, 0.95]
const S = gram(diffMatrix(K, 2))
const location = (x: number) => 1 + 0.8 * Math.sin(2 * Math.PI * x)
const scale = (x: number) => 0.15 + 0.6 * x

/** The τ-expectile of the standard normal: the root of E(m − Z)⁺ / E|Z − m| = τ, found by bisection. */
function normalExpectile(tau: number): number {
  const lower = (m: number) => m * normalCdf(m) + normalPdf(m)
  const upper = (m: number) => normalPdf(m) - m * (1 - normalCdf(m))
  let [lo, hi] = [-6, 6]
  for (let i = 0; i < 60; i++) {
    const m = (lo + hi) / 2
    if (lower(m) / (lower(m) + upper(m)) < tau) lo = m
    else hi = m
  }
  return (lo + hi) / 2
}

/**
 * Asymmetric least squares with a P-spline: weight τ above the curve and 1 − τ below it, refit, and repeat until no
 * point changes side.
 */
function laws(B: Matrix, y: number[], tau: number, lambda: number) {
  let w = y.map(() => 0.5)
  let beta: number[] = []
  let iterations = 0
  while (iterations < 50) {
    beta = cholSolve(cholesky(addScaled(crossprod(B, w), [lambda, S])), crossprodY(B, y, w))
    iterations++
    const next = B.map((row, i) => (y[i] > dot(row, beta) ? tau : 1 - tau))
    if (next.every((v, i) => v === w[i])) break
    w = next
  }
  return { beta, iterations }
}

export function ExpectileCurves() {
  const state = useFigureState({
    tau: float(0.9, { min: 0.01, max: 0.99, step: 0.01, label: 'τ' }),
    logLambda: float(0, { min: -2, max: 4, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    seed: int(11, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    const y = x.map((xi) => location(xi) + scale(xi) * normal(r))
    return { x, y, B: x.map((xi) => psplineRow(xi, 0, 1, K)) }
  }, [state.seed])
  const Bgrid = useMemo(() => GRID.map((g) => psplineRow(g, 0, 1, K)), [])

  const lambda = 10 ** state.logLambda
  const background = useMemo(() => BACKGROUND.map((t) => laws(data.B, data.y, t, lambda).beta), [data, lambda])
  const chosen = useMemo(() => laws(data.B, data.y, state.tau, lambda), [data, state.tau, lambda])
  const e = normalExpectile(state.tau)
  const below = mean(data.B.map((row, i) => (data.y[i] < dot(row, chosen.beta) ? 1 : 0)))

  const series: SeriesSpec[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    ...BACKGROUND.map((t, j): SeriesSpec => ({
      name: `τ = ${t}`,
      type: 'line',
      x: GRID,
      y: Bgrid.map((row) => dot(row, background[j])),
      muted: true,
    })),
    {
      name: 'true τ-expectile',
      type: 'line',
      x: GRID,
      y: GRID.map((g) => location(g) + scale(g) * e),
      slot: 2,
      dashed: true,
    },
    { name: `fitted τ-expectile`, type: 'line', x: GRID, y: Bgrid.map((row) => dot(row, chosen.beta)), slot: 1 },
  ]

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-1.5, 3.5] })
  return (
    <Figure
      title="Expectile curves by asymmetric least squares"
      state={state}
      caption={
        <>
          Three hundred points with a sinusoidal mean and a spread that grows from left to right. Grey curves are
          P-spline expectile fits at τ = 0.05, 0.25, 0.5, 0.75 and 0.95; the τ = 0.5 curve is the ordinary mean fit. The
          coloured curve is the fit at the chosen τ, and the dashed curve is the true τ-expectile of the Gaussian
          conditional distribution. The curves fan out where the noise is large. The readouts show the share of points
          below the fitted curve and the quantile level that the true τ-expectile corresponds to: for Gaussian noise the
          0.9-expectile is only about the 0.81-quantile.
        </>
      }

      readouts={
        <>
          <Readout label="share of y below the fit" value={formatNumber(below)} />
          <Readout label="quantile level of the true τ-expectile" value={formatNumber(normalCdf(e))} />
          <Readout label="reweighting iterations" value={String(chosen.iterations)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
