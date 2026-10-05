import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

// Rubin's (1981) eight schools: estimated coaching effects y_j and their standard errors σ_j, in SAT points.
const Y = [28, 8, -3, 7, -1, 1, 18, 12]
const SIGMA = [15, 10, 16, 11, 9, 11, 10, 18]
const NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
const TAU_MAX = 30
const TAUS = toFlat(linspace(0, TAU_MAX, 151))

/** Precision-weighted grand mean μ̂(τ), its variance, and the conditional posterior means of the eight effects. */
function atTau(tau: number) {
  const w = SIGMA.map((s) => 1 / (s * s + tau * tau))
  const sw = w.reduce((a, b) => a + b, 0)
  const mu = w.reduce((a, wj, j) => a + wj * Y[j], 0) / sw
  const shrink = SIGMA.map((s) => (s * s) / (s * s + tau * tau))
  const theta = Y.map((y, j) => shrink[j] * mu + (1 - shrink[j]) * y)
  // log p(τ | y) up to a constant, under a uniform prior on τ.
  const logPost =
    0.5 * Math.log(1 / sw) +
    SIGMA.reduce((a, s, j) => a - 0.5 * Math.log(s * s + tau * tau) - (Y[j] - mu) ** 2 / (2 * (s * s + tau * tau)), 0)
  return { mu, shrink, theta, logPost }
}

const PATHS = TAUS.map(atTau)
const LOG_MAX = Math.max(...PATHS.map((p) => p.logPost))

/**
 * Conditional posterior means E[θ_j | τ, y] = B_j μ̂(τ) + (1 − B_j) y_j as the between-school standard deviation τ
 * varies. At τ = 0 every school gets the pooled mean; as τ grows each estimate returns to its own y_j.
 */
export function Shrinkage() {
  const state = useFigureState({
    tau: float(8, { min: 0, max: TAU_MAX, step: 0.5, label: 'between-school sd τ' }),
  })
  const now = useMemo(() => atTau(state.tau), [state.tau])

  const series = useMemo<SeriesSpec[]>(
    () => [
      ...NAMES.map((name, j) => ({
        name: `school ${name}`,
        type: 'line' as const,
        x: TAUS,
        y: PATHS.map((p) => p.theta[j]),
        slot: j,
      })),
      { name: 'grand mean μ̂(τ)', type: 'line', x: TAUS, y: PATHS.map((p) => p.mu), emphasis: true, dashed: true },
    ],
    [],
  )

  const xAxis = useAxis({ label: 'between-school standard deviation τ', range: [0, TAU_MAX] })
  const yAxis = useAxis({ label: 'E[θ_j | τ, y]', range: [-5, 30] })
  return (
    <Figure
      title="Partial pooling in the eight schools"
      state={state}
      caption="Each line is one school's conditional posterior mean effect as a function of the between-school standard deviation τ. At τ = 0 the model pools completely and every school gets the precision-weighted mean 7.7. As τ grows the shrinkage factor B_j = σ_j²/(σ_j² + τ²) falls and each estimate returns to the school's own result. Schools with large standard errors (A, C, H) are pulled hardest. Drag the τ line or use the slider."

      readouts={
        <>
          <Readout label="grand mean μ̂(τ)" value={formatNumber(now.mu)} />
          <Readout label="school A: y = 28 →" value={formatNumber(now.theta[0])} />
          <Readout label="shrinkage B_A" value={formatNumber(now.shrink[0])} />
          <Readout label="p(τ | y) relative to its maximum" value={formatNumber(Math.exp(now.logPost - LOG_MAX))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
        <Handle {...state.handle('tau', { label: 'τ' })} />
      </Plot>
    </Figure>
  )
}
