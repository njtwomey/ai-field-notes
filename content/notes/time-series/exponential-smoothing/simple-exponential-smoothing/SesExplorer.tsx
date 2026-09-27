import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'
import { rmse, ses } from '../_shared/smoothing'

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
  const alpha = useParam(0.3, { min: 0.01, max: 1, step: 0.01 })
  const logQ = useParam(-1, { min: -3, max: 1, step: 0.1 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const q = 10 ** logQ.value

  const data = useMemo(() => {
    const g = rng(seed.value)
    let level = 10
    const mu: number[] = []
    const y: number[] = []
    for (let t = 0; t < T; t++) {
      level += Math.sqrt(q) * g.normal()
      mu.push(level)
      y.push(level + g.normal())
    }
    // Score each α by its one-step forecast errors after a short burn-in, so the start value ℓ_0 = y_1 matters less.
    const score = (a: number) => rmse(ses(y, a).fitted.slice(10), y.slice(10))
    const best = GRID.reduce((b, a) => (score(a) < score(b) ? a : b), GRID[0])
    return { mu, y, score, best }
  }, [q, seed.value])

  const r = useMemo(() => {
    const fit = ses(data.y, alpha.value)
    const series: XYSeries[] = [
      { name: 'observations y_t', type: 'scatter', x: TIMES, y: data.y, muted: true },
      { name: 'true level μ_t', type: 'line', x: TIMES, y: data.mu, emphasis: true },
      { name: 'SES forecast ŷ_{t|t−1}', type: 'line', x: TIMES, y: fit.fitted, slot: 0 },
      { name: 'forecast ŷ_{T+h|T}', type: 'line', x: AHEAD, y: AHEAD.map(() => fit.level), slot: 0, dashed: true },
    ]
    const weights: XYSeries[] = [
      {
        name: 'weight on y_{T−j}',
        type: 'bar',
        x: WEIGHT_LAGS,
        y: WEIGHT_LAGS.map((j) => alpha.value * (1 - alpha.value) ** j),
        slot: 1,
      },
    ]
    return { series, weights, error: data.score(alpha.value) }
  }, [data, alpha.value])

  return (
    <Interactive
      title="Simple exponential smoothing"
      caption="Data from a local-level model: a random-walk level μ_t with step variance q, observed with unit-variance noise. The solid line is the one-step-ahead SES forecast, which is the previous smoothed level; the dashed line is the flat forecast beyond the data. A small α averages over a long window and lags the level; α near 1 copies the last observation and follows the noise. The lower chart shows the weights α(1 − α)^j that the current level puts on past observations. The error-minimising α scatters around the steady-state Kalman gain for the same q; with 120 points the scatter is wide, and it narrows for longer series."
      controls={
        <>
          <ParamSlider label="smoothing α" param={alpha} />
          <ParamSlider label="signal-to-noise q" param={logQ} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="one-step RMSE at α" value={formatNumber(r.error)} />
          <Readout label="best α on this series" value={formatNumber(data.best)} />
          <Readout label="Kalman steady-state α for q" value={formatNumber(kalmanAlpha(q))} />
        </>
      }
    >
      <div className="space-y-3">
        <XYChart series={r.series} xLabel="t" yLabel="y_t" height={280} />
        <XYChart series={r.weights} xLabel="lag j" yLabel="weight" yRange={[0, 1]} height={130} />
      </div>
    </Interactive>
  )
}
