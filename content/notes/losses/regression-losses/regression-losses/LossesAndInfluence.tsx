import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'

const R = linspace(-4, 4, 321)
const DATA = [1, 2, 3, 6, 18]

const clip = (r: number, d: number) => Math.max(-d, Math.min(d, r))
const huber = (r: number, d: number) => (Math.abs(r) <= d ? 0.5 * r * r : d * (Math.abs(r) - 0.5 * d))
const pinball = (r: number, tau: number) => r * (tau - (r < 0 ? 1 : 0))
const logCosh = (r: number) => Math.abs(r) + Math.log1p(Math.exp(-2 * Math.abs(r))) - Math.LN2

/** Root of a decreasing function of θ on [lo, hi] by bisection: the M-estimate solving Σψ(yᵢ − θ) = 0. */
function solve(psiSum: (theta: number) => number, lo = -100, hi = 100) {
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2
    if (psiSum(mid) > 0) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/** Regression losses as functions of the residual, and their derivatives (influence functions). */
export function LossesAndInfluence() {
  const delta = useParam(1, { min: 0.2, max: 3, step: 0.1 })
  const tau = useParam(0.7, { min: 0.05, max: 0.95, step: 0.05 })

  const [losses, influence] = useMemo((): [XYSeries[], XYSeries[]] => {
    const d = delta.value
    const t = tau.value
    return [
      [
        { name: 'squared ½r²', type: 'line', x: R, y: R.map((r) => 0.5 * r * r), slot: 0 },
        { name: 'absolute |r|', type: 'line', x: R, y: R.map(Math.abs), slot: 1 },
        { name: 'Huber', type: 'line', x: R, y: R.map((r) => huber(r, d)), slot: 2 },
        { name: 'log-cosh', type: 'line', x: R, y: R.map(logCosh), slot: 3 },
        { name: 'quantile (pinball)', type: 'line', x: R, y: R.map((r) => pinball(r, t)), slot: 4 },
      ],
      [
        { name: 'squared ½r²', type: 'line', x: R, y: R, slot: 0 },
        { name: 'absolute |r|', type: 'line', x: R, y: R.map(Math.sign), slot: 1 },
        { name: 'Huber', type: 'line', x: R, y: R.map((r) => clip(r, d)), slot: 2 },
        { name: 'log-cosh', type: 'line', x: R, y: R.map(Math.tanh), slot: 3 },
        { name: 'quantile (pinball)', type: 'line', x: R, y: R.map((r) => (r < 0 ? t - 1 : t)), slot: 4 },
      ],
    ]
  }, [delta.value, tau.value])

  const sorted = [...DATA].sort((a, b) => a - b)
  const mean = DATA.reduce((a, b) => a + b, 0) / DATA.length
  const huberFit = solve((th) => DATA.reduce((a, y) => a + clip(y - th, delta.value), 0))
  const logCoshFit = solve((th) => DATA.reduce((a, y) => a + Math.tanh(y - th), 0))
  const quantile = sorted[Math.max(0, Math.ceil(DATA.length * tau.value - 1e-9) - 1)]

  return (
    <Interactive
      title="Regression losses and their influence functions"
      caption="Left: each loss as a function of the residual r = y − ŷ. Right: its derivative ψ(r), the pull that one residual exerts on the fit. The squared loss pulls in proportion to the residual, so one outlier can drag the fit anywhere. The absolute, Huber, log-cosh and quantile losses have bounded pull. The readout gives the constant that minimises each loss on the five values 1, 2, 3, 6, 18."
      controls={
        <>
          <ParamSlider label="Huber threshold δ" param={delta} />
          <ParamSlider label="quantile level τ" param={tau} />
        </>
      }
      readout={
        <>
          <Readout label="squared → mean" value={formatNumber(mean)} />
          <Readout label="absolute → median" value={formatNumber(sorted[2])} />
          <Readout label="Huber" value={formatNumber(huberFit)} />
          <Readout label="log-cosh" value={formatNumber(logCoshFit)} />
          <Readout label="quantile τ" value={formatNumber(quantile)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={losses} xLabel="residual r" yLabel="loss" xRange={[-4, 4]} yRange={[0, 4]} height={320} />
        <XYChart
          series={influence}
          xLabel="residual r"
          yLabel="ψ(r) = dloss/dr"
          xRange={[-4, 4]}
          yRange={[-3, 3]}
          height={320}
        />
      </div>
    </Interactive>
  )
}
