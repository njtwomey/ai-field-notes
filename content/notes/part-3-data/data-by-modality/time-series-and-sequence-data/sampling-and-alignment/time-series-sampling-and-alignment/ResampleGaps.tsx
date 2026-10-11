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
  Rug,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { rootMeanSquaredError } from 'aifn-compute/learning/metrics'
import { evaluatePiecewise, linearInterpolant, naturalCubicSpline, pchip } from 'aifn-compute/numerics/interpolate'
import type { SpectralTruth } from 'aifn-methods/data'
import { unevenSinusoids } from 'aifn-methods/data/signals'

const SPAN = 20
const METHODS = [
  { value: 'linear', label: 'linear' },
  { value: 'pchip', label: 'monotone cubic (PCHIP)' },
  { value: 'spline', label: 'natural cubic spline' },
] as const
const INTERPOLANT = { linear: linearInterpolant, pchip, spline: naturalCubicSpline }

/** Irregular samples with gaps, resampled onto a regular grid by three interpolants. */
export function ResampleGaps() {
  const state = useFigureState({
    method: choice(METHODS, 'linear', { label: 'interpolation' }),
    step: float(0.25, { ge: 0.05, le: 2, suggestions: [0.1, 0.25, 0.5, 1], label: 'grid step Δ' }),
    n: int(40, { ge: 8, le: 200, suggestions: [20, 40, 100], label: 'samples' }),
    noise: slider(0, 0.5, 0.05, { step: 0.01, label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { method, step, n, noise, seed } = state
  const view = useMemo(() => {
    const d = unevenSinusoids(stream(`resample-gaps/${seed}`), {
      sampling: 'gaps',
      n,
      span: SPAN,
      frequency: 0.15,
      gapFraction: 0.35,
      noise,
    })
    const t = toFlat(d.x)
    const y = toFlat(d.y!)
    const line = (d.meta.truth as SpectralTruth).lines[0]
    const truth = (u: number) => line.amplitude * Math.sin(2 * Math.PI * line.frequency * u + line.phase)
    const grid: number[] = []
    for (let u = t[0]; u <= t[t.length - 1] + 1e-9; u += step) grid.push(u)
    const fitted = toFlat(evaluatePiecewise(INTERPOLANT[method](tensor(t), tensor(y)), tensor(grid)))
    const dense = Array.from({ length: 401 }, (_, i) => (i / 400) * SPAN)
    const longestGap = Math.max(...t.slice(1).map((u, i) => u - t[i]))
    return {
      t,
      y,
      grid,
      fitted,
      dense,
      truth: dense.map(truth),
      rmse: rootMeanSquaredError(grid.map(truth), fitted),
      longestGap,
    }
  }, [method, step, n, noise, seed])
  const tAxis = useAxis({ label: 'time t', range: [0, SPAN] })
  const yAxis = useAxis({ label: 'y', range: [-2, 2] })
  return (
    <Figure
      title="Resampling across gaps"
      hoverReadout={false}
      state={state}
      caption="Dots: a sinusoid sampled at irregular times, with three gaps that cover about a third of the span. Squares: the series resampled onto a regular grid of step Δ by the chosen interpolant. Dashed: the true curve. Inside the observed stretches every interpolant is close to the truth. Inside a gap each one invents a curve that the data do not support: linear interpolation draws a straight chord, the monotone cubic never overshoots the neighbouring samples, and the natural spline can swing past them. The error readout compares the grid values with the true curve."
      readouts={
        <>
          <Readout label="grid points" value={String(view.grid.length)} />
          <Readout label="longest gap" value={formatNumber(view.longestGap)} />
          <Readout label="RMSE on the grid" value={formatNumber(view.rmse)} />
        </>
      }
    >
      <Plot x={tAxis} y={yAxis}>
        <Curve name="true curve" x={view.dense} y={view.truth} muted dashed silent />
        <Curve name="resampled" x={view.grid} y={view.fitted} slot={1} thin />
        <Points name="resampled grid" x={view.grid} y={view.fitted} slot={1} shape={1} size={5} />
        <Points name="observed samples" x={view.t} y={view.y} slot={0} />
        <Rug name="sample times" values={view.t} muted />
      </Plot>
    </Figure>
  )
}
