import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

const XS = linspace(-4, 4, 300)
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
  const tau = useParam(1, { min: 0.05, max: 3, step: 0.05 })
  const x = useParam(1, { min: -4, max: 4, step: 0.05 })

  const series = useMemo((): XYSeries[] => {
    const t = tau.value
    return [
      { name: 'max(x, 0)', type: 'line', x: XS, y: XS.map((v) => Math.max(v, 0)), slot: 0, dashed: true },
      { name: 'τ · LSE(x/τ, 0)', type: 'line', x: XS, y: XS.map((v) => lse(v, 0, t)), slot: 1 },
      { name: 'softmax p₁ = ∂ LSE / ∂x', type: 'line', x: XS, y: XS.map((v) => 1 / (1 + Math.exp(-v / t))), slot: 2 },
    ]
  }, [tau.value])

  const p1 = 1 / (1 + Math.exp(-x.value / tau.value))
  const value = lse(x.value, 0, tau.value)
  const handles: Handle[] = [{ kind: 'x', at: x.value, label: 'z₁', onDrag: x.set }]

  return (
    <Interactive
      title="Log-sum-exp is a smooth maximum"
      caption="Two logits, z₁ = x and z₂ = 0. The orange curve is log-sum-exp at temperature τ; it lies above max(x, 0) by at most τ log 2, and it approaches the maximum as τ shrinks. Its slope is the softmax probability of the first logit (green), a smoothed step that sharpens into an indicator as τ → 0. Drag the line labelled z₁ to read the values."
      controls={
        <>
          <ParamSlider label="temperature τ" param={tau} />
          <ParamSlider label="logit z₁" param={x} />
        </>
      }
      readout={
        <>
          <Readout label="max(z₁, 0)" value={formatNumber(Math.max(x.value, 0))} />
          <Readout label="τ · LSE" value={formatNumber(value)} />
          <Readout label="gap (≤ τ log 2)" value={formatNumber(value - Math.max(x.value, 0))} />
          <Readout label="softmax (p₁, p₂)" value={`(${formatNumber(p1)}, ${formatNumber(1 - p1)})`} />
        </>
      }
    >
      <XYChart
        series={series}
        xRange={[-4, 4]}
        yRange={[-0.2, 4.4]}
        xLabel="z₁"
        yLabel="value"
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
