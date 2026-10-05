import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const XS = toFlat(linspace(-4, 4, 300))
/** τ·log(e^{a/τ} + e^{b/τ}), computed stably by factoring out the larger term. */
const lse = (a: number, b: number, tau: number) => {
  const m = Math.max(a, b)
  return m + tau * Math.log(Math.exp((a - m) / tau) + Math.exp((b - m) / tau))
}

/**
 * Two logits, z₁ = x (moved along the axis) and z₂ = 0. Log-sum-exp with temperature τ is a smooth maximum; its
 * derivative in x is the softmax probability of the first logit, a smoothed step.
 */
export function SmoothMax() {
  const state = useFigureState({
    tau: float(1, { min: 0.05, max: 3, step: 0.05, label: 'temperature τ' }),
    x: float(1, { min: -4, max: 4, step: 0.05, label: 'logit z₁' }),
  })

  const series = useMemo(() => {
    const t = state.tau
    return [
      { name: 'max(x, 0)', x: XS, y: XS.map((v) => Math.max(v, 0)), slot: 0, dashed: true },
      { name: 'τ · LSE(x/τ, 0)', x: XS, y: XS.map((v) => lse(v, 0, t)), slot: 1 },
      { name: 'softmax p₁ = ∂ LSE / ∂x', x: XS, y: XS.map((v) => 1 / (1 + Math.exp(-v / t))), slot: 2 },
    ] as const
  }, [state.tau])

  const p1 = 1 / (1 + Math.exp(-state.x / state.tau))
  const value = lse(state.x, 0, state.tau)

  const xAxis = useAxis({ label: 'z₁', range: [-4, 4] })
  const yAxis = useAxis({ label: 'value', range: [-0.2, 4.4] })
  return (
    <Figure
      title="Log-sum-exp is a smooth maximum"
      state={state}
      caption="Two logits, z₁ = x and z₂ = 0. The orange curve is log-sum-exp at temperature τ; it lies above max(x, 0) by at most τ log 2, and it approaches the maximum as τ shrinks. Its slope is the softmax probability of the first logit (green), a smoothed step that sharpens into an indicator as τ → 0. Drag the line labelled z₁ to read the values."

      readouts={
        <>
          <Readout label="max(z₁, 0)" value={formatNumber(Math.max(state.x, 0))} />
          <Readout label="τ · LSE" value={formatNumber(value)} />
          <Readout label="gap (≤ τ log 2)" value={formatNumber(value - Math.max(state.x, 0))} />
          <Readout label="softmax (p₁, p₂)" value={`(${formatNumber(p1)}, ${formatNumber(1 - p1)})`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('x', { label: 'z₁' })} />
      </Plot>
    </Figure>
  )
}
