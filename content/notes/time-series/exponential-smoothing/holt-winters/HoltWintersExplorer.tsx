import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { rmse, smooth, type Seasonality } from '../_shared/smoothing'

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
  const [seasonality, setSeasonality] = useState<Seasonality>('multiplicative')
  const alpha = useParam(0.3, { min: 0.01, max: 1, step: 0.01 })
  const beta = useParam(0.05, { min: 0, max: 1, step: 0.01 })
  const gamma = useParam(0.2, { min: 0, max: 1, step: 0.01 })
  const phi = useParam(1, { min: 0.8, max: 1, step: 0.01 })
  const seed = useParam(5, { min: 1, max: 30, step: 1 })

  const data = useMemo(() => {
    const g = rng(seed.value)
    const y = Array.from({ length: TRAIN + TEST }, (_, t) => {
      const level = 100 + 2 * t
      return level * (1 + 0.25 * SHAPE[t % M]) * (1 + 0.03 * g.normal())
    })
    return { train: y.slice(0, TRAIN), test: y.slice(TRAIN) }
  }, [seed.value])

  const r = useMemo(() => {
    const fit = smooth(
      data.train,
      {
        alpha: alpha.value,
        beta: beta.value,
        gamma: gamma.value,
        phi: phi.value,
        trend: true,
        seasonality,
        m: M,
      },
      TEST,
    )
    const series: XYSeries[] = [
      { name: 'data (training)', type: 'scatter', x: TRAIN_T, y: data.train, muted: true },
      { name: 'data (held out)', type: 'scatter', x: TEST_T, y: data.test, emphasis: true },
      { name: 'one-step forecast', type: 'line', x: TRAIN_T.slice(M), y: fit.fitted.slice(M), slot: 0 },
      { name: 'forecast from t = 72', type: 'line', x: TEST_T, y: fit.forecast, slot: 1 },
    ]
    return {
      series,
      trainError: rmse(fit.fitted.slice(M), data.train.slice(M)),
      testError: rmse(fit.forecast, data.test),
    }
  }, [data, alpha.value, beta.value, gamma.value, phi.value, seasonality])

  return (
    <Interactive
      title="Holt–Winters forecasting"
      caption="Six years of monthly data with a linear trend and a seasonal swing proportional to the level (grey), and two held-out years (large points). The blue line is the one-step-ahead forecast over the training years, shown after the first year, which only initialises the states. The orange line forecasts the held-out years from the end of the training data. α, β* and γ set how fast the level, slope and seasonal states adapt; φ < 1 damps the trend. Additive seasonality underestimates the swing at the end of the series, where the level is highest; multiplicative seasonality scales it."
      controls={
        <>
          <ParamChoice
            label="seasonality"
            value={seasonality}
            onChange={setSeasonality}
            options={[
              { value: 'additive', label: 'additive' },
              { value: 'multiplicative', label: 'multiplicative' },
              { value: 'none', label: 'none (Holt)' },
            ]}
          />
          <ParamSlider label="level α" param={alpha} />
          <ParamSlider label="slope β*" param={beta} />
          <ParamSlider label="season γ" param={gamma} />
          <ParamSlider label="damping φ" param={phi} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="one-step RMSE, training" value={formatNumber(r.trainError)} />
          <Readout label="RMSE, held-out forecast" value={formatNumber(r.testError)} />
        </>
      }
    >
      <XYChart series={r.series} xLabel="month t" yLabel="y_t" height={320} />
    </Interactive>
  )
}
