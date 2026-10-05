import { useMemo } from 'react'
import {
  choice,
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
import { rmse, smooth, type Seasonality } from '../_shared/smoothing'
import { normal, stream } from 'aifn-compute/foundation/random'

const M = 12
const TRAIN = 6 * M
const TEST = 2 * M
const TRAIN_T = Array.from({ length: TRAIN }, (_, t) => t + 1)
const TEST_T = Array.from({ length: TEST }, (_, h) => TRAIN + h + 1)
/** Seasonal shape with mean 0: a summer peak and a smaller winter bump. */
const SHAPE = Array.from(
  { length: M },
  (_, j) => Math.sin((2 * Math.PI * (j - 2)) / M) + 0.3 * Math.cos((4 * Math.PI * j) / M),
)

/** Holt–Winters on monthly data whose seasonal swing grows with the level; fit on six years, forecast two. */
export function HoltWintersExplorer() {
  const state = useFigureState({
    seasonality: choice<Seasonality>(
      [
        { value: 'additive', label: 'additive' },
        { value: 'multiplicative', label: 'multiplicative' },
        { value: 'none', label: 'none (Holt)' },
      ],
      'multiplicative',
      { label: 'seasonality' },
    ),
    alpha: float(0.3, { min: 0.01, max: 1, step: 0.01, label: 'level α' }),
    beta: float(0.05, { min: 0, max: 1, step: 0.01, label: 'slope β*' }),
    gamma: float(0.2, { min: 0, max: 1, step: 0.01, label: 'season γ' }),
    phi: float(1, { min: 0.8, max: 1, step: 0.01, label: 'damping φ' }),
    seed: int(5, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const data = useMemo(() => {
    const g = stream(state.seed)
    const y = Array.from({ length: TRAIN + TEST }, (_, t) => {
      const level = 100 + 2 * t
      return level * (1 + 0.25 * SHAPE[t % M]) * (1 + 0.03 * normal(g))
    })
    return { train: y.slice(0, TRAIN), test: y.slice(TRAIN) }
  }, [state.seed])

  const r = useMemo(() => {
    const fit = smooth(
      data.train,
      {
        alpha: state.alpha,
        beta: state.beta,
        gamma: state.gamma,
        phi: state.phi,
        trend: true,
        seasonality: state.seasonality,
        m: M,
      },
      TEST,
    )
    const series = [
      { name: 'data (training)', x: TRAIN_T, y: data.train, muted: true },
      { name: 'data (held out)', x: TEST_T, y: data.test, emphasis: true },
      { name: 'one-step forecast', x: TRAIN_T.slice(M), y: fit.fitted.slice(M), slot: 0 },
      { name: 'forecast from t = 72', x: TEST_T, y: fit.forecast, slot: 1 },
    ] as const
    return {
      series,
      trainError: rmse(fit.fitted.slice(M), data.train.slice(M)),
      testError: rmse(fit.forecast, data.test),
    }
  }, [data, state.alpha, state.beta, state.gamma, state.phi, state.seasonality])

  const xAxis = useAxis({ label: 'month t', hold: 'union' })
  const yAxis = useAxis({ label: 'y_t', hold: 'union' })
  return (
    <Figure
      title="Holt–Winters forecasting"
      state={state}
      caption="Six years of monthly data with a linear trend and a seasonal swing proportional to the level (grey), and two held-out years (large points). The blue line is the one-step-ahead forecast over the training years, shown after the first year, which only initialises the states. The orange line forecasts the held-out years from the end of the training data. α, β* and γ set how fast the level, slope and seasonal states adapt; φ < 1 damps the trend. Additive seasonality underestimates the swing at the end of the series, where the level is highest; multiplicative seasonality scales it."

      readouts={
        <>
          <Readout label="one-step RMSE, training" value={formatNumber(r.trainError)} />
          <Readout label="RMSE, held-out forecast" value={formatNumber(r.testError)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Points {...r.series[0]} />
        <Points {...r.series[1]} />
        <Curve {...r.series[2]} />
        <Curve {...r.series[3]} />
      </Plot>
    </Figure>
  )
}
