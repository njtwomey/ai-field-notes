import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const ALPHA = toFlat(linspace(0, 1, 201))

/**
 * Estimating a target mean from n_t target and n_s source observations whose mean is off by δ. The combined estimate
 * α x̄_s + (1 − α) x̄_t has MSE α²(δ² + σ²/n_s) + (1 − α)² σ²/n_t; everything is shown relative to target only (α = 0).
 */
export function SourceTargetTradeoff() {
  const state = useFigureState({
    delta: float(0.3, { min: 0, max: 1.5, step: 0.01, label: 'bias δ / σ' }),
    nt: int(10, { min: 1, max: 100, step: 1, label: 'target sample size n_t' }),
    ns: int(500, { min: 10, max: 2000, step: 10, label: 'source sample size n_s' }),
    alpha: float(0.5, { min: 0, max: 1, step: 0.005, label: 'weight on source α' }),
  })

  const r = useMemo(() => {
    const vt = 1 / state.nt
    const vs = 1 / state.ns
    const rel = (a: number) => (a * a * (state.delta * state.delta + vs) + (1 - a) * (1 - a) * vt) / vt
    const star = vt / (state.delta * state.delta + vs + vt)
    const pooled = state.ns / (state.ns + state.nt)
    const series = [
      { name: 'combined estimate', x: ALPHA, y: ALPHA.map(rel), slot: 0 },
      { name: 'target only', x: [0, 1], y: [1, 1], dashed: true, muted: true },
      { name: 'source only', x: [1], y: [rel(1)], slot: 1 },
      { name: 'pooled, α = n_s/(n_s + n_t)', x: [pooled], y: [rel(pooled)], slot: 2 },
      { name: 'optimum α*', x: [star], y: [rel(star)], emphasis: true },
    ] as const
    return { series, star, pooled, rel }
  }, [state.delta, state.nt, state.ns])

  const xAxis = useAxis({ label: 'weight on source α', range: [0, 1] })
  const yAxis = useAxis({ label: 'MSE relative to target only', range: [0, 3] })
  return (
    <Figure
      title="Borrowing from a biased source"
      state={state}
      caption="Mean squared error of α·(source mean) + (1 − α)·(target mean), relative to using the target alone, for unit noise variance. Drag the vertical line or use the α slider. Below the dashed line the source helps; above it the source hurts (the axis stops at 3). Every α between 0 and 2α* helps, and pooling the two samples hurts once the bias δ is large compared with the target's standard error."

      readouts={
        <>
          <Readout label="relative MSE at α" value={formatNumber(r.rel(state.alpha))} />
          <Readout label="optimal α*" value={formatNumber(r.star)} />
          <Readout label="break-even 2α*" value={formatNumber(Math.min(2 * r.star, 1))} />
          <Readout label="relative MSE, pooled" value={formatNumber(r.rel(r.pooled))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...r.series[0]} />
        <Curve {...r.series[1]} />
        <Points {...r.series[2]} />
        <Points {...r.series[3]} />
        <Points {...r.series[4]} />
        <Handle {...state.handle('alpha', { label: 'α' })} />
      </Plot>
    </Figure>
  )
}
