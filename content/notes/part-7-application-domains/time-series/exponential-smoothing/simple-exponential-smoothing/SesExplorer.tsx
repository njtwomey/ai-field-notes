import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { rmse, ses } from '../_shared/smoothing'
import { normal, stream } from 'aifn-compute/foundation/random'

const T = 120
const H = 20
const TIMES = Array.from({ length: T }, (_, t) => t + 1)
const AHEAD = Array.from({ length: H }, (_, h) => T + h + 1)
const WEIGHT_LAGS = Array.from({ length: 16 }, (_, j) => j)
const GRID = Array.from({ length: 99 }, (_, i) => (i + 1) / 100)

/** Steady-state Kalman gain of the local-level model with signal-to-noise ratio q = σ²_η / σ²_ε. */
const kalmanAlpha = (q: number) => {
  const p = (q + Math.sqrt(q * q + 4 * q)) / 2
  return p / (p + 1)
}

/** SES on data from a local-level model: slide α and compare it with the optimum and with the Kalman gain. */
export function SesExplorer() {
  const state = useFigureState({
    alpha: float(0.3, { min: 0.01, max: 1, step: 0.01, label: 'smoothing α' }),
    logQ: float(-1, {
      min: -3,
      max: 1,
      step: 0.1,
      label: 'signal-to-noise q',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const q = 10 ** state.logQ

  const data = useMemo(() => {
    const g = stream(state.seed)
    let level = 10
    const mu: number[] = []
    const y: number[] = []
    for (let t = 0; t < T; t++) {
      level += Math.sqrt(q) * normal(g)
      mu.push(level)
      y.push(level + normal(g))
    }
    // Score each α by its one-step forecast errors after a short burn-in, so the start value ℓ_0 = y_1 matters less.
    const score = (a: number) => rmse(ses(y, a).fitted.slice(10), y.slice(10))
    const best = GRID.reduce((b, a) => (score(a) < score(b) ? a : b), GRID[0])
    return { mu, y, score, best }
  }, [q, state.seed])

  const r = useMemo(() => {
    const fit = ses(data.y, state.alpha)
    const series = [
      { name: 'observations y_t', x: TIMES, y: data.y, muted: true },
      { name: 'true level μ_t', x: TIMES, y: data.mu, emphasis: true },
      { name: 'SES forecast ŷ_{t|t−1}', x: TIMES, y: fit.fitted, slot: 0 },
      { name: 'forecast ŷ_{T+h|T}', x: AHEAD, y: AHEAD.map(() => fit.level), slot: 0, dashed: true },
    ] as const
    const weights = [
      {
        name: 'weight on y_{T−j}',
        x: WEIGHT_LAGS,
        y: WEIGHT_LAGS.map((j) => state.alpha * (1 - state.alpha) ** j),
        slot: 1,
      },
    ] as const
    return { series, weights, error: data.score(state.alpha) }
  }, [data, state.alpha])

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ label: 'y_t', hold: 'union' })
  const xAxis2 = useAxis({ label: 'lag j', hold: 'union' })
  const yAxis2 = useAxis({ label: 'weight', range: [0, 1] })
  return (
    <Figure
      title="Simple exponential smoothing"
      state={state}
      caption="Data from a local-level model: a random-walk level μ_t with step variance q, observed with unit-variance noise. The solid line is the one-step-ahead SES forecast, which is the previous smoothed level; the dashed line is the flat forecast beyond the data. A small α averages over a long window and lags the level; α near 1 copies the last observation and follows the noise. The lower chart shows the weights α(1 − α)^j that the current level puts on past observations. The error-minimising α scatters around the steady-state Kalman gain for the same q; with 120 points the scatter is wide, and it narrows for longer series."

      readouts={
        <>
          <Readout label="one-step RMSE at α" value={formatNumber(r.error)} />
          <Readout label="best α on this series" value={formatNumber(data.best)} />
          <Readout label="Kalman steady-state α for q" value={formatNumber(kalmanAlpha(q))} />
        </>
      }
    >
      <div className="space-y-3">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Points {...r.series[0]} />
          <Curve {...r.series[1]} />
          <Curve {...r.series[2]} />
          <Curve {...r.series[3]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={130}>
          <Bars {...r.weights[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
