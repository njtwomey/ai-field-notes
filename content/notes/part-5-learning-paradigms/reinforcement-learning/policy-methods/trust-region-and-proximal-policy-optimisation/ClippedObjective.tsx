import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'

const R = linspace(0, 2, 401)
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const objective = (r: number, adv: number, eps: number) => Math.min(r * adv, clip(r, 1 - eps, 1 + eps) * adv)

/** PPO's clipped surrogate for one sample as a function of the probability ratio, for a positive and a negative advantage. */
export function ClippedObjective() {
  const eps = useParam(0.2, { min: 0.05, max: 0.5, step: 0.01 })
  const ratio = useParam(1.3, { min: 0, max: 2, step: 0.01 })

  const series: XYSeries[] = useMemo(
    () => [
      { name: 'clipped, A = +1', type: 'line', x: R, y: R.map((r) => objective(r, 1, eps.value)), slot: 0 },
      { name: 'unclipped r·A, A = +1', type: 'line', x: R, y: R, slot: 0, dashed: true },
      { name: 'clipped, A = −1', type: 'line', x: R, y: R.map((r) => objective(r, -1, eps.value)), slot: 1 },
      { name: 'unclipped r·A, A = −1', type: 'line', x: R, y: R.map((r) => -r), slot: 1, dashed: true },
    ],
    [eps.value],
  )
  const r = ratio.value
  // The gradient with respect to r is A where the unclipped term is the minimum, and 0 where the clipped term is.
  const gradient = (adv: number) => (r * adv <= clip(r, 1 - eps.value, 1 + eps.value) * adv ? adv : 0)

  return (
    <Interactive
      title="The clipped surrogate objective"
      caption="One sample's contribution to PPO's objective, min(r A, clip(r, 1 − ε, 1 + ε) A), as a function of the probability ratio r = π_θ(a | s) / π_old(a | s). With a positive advantage the objective rises with r only up to 1 + ε and is flat beyond it, so the gradient stops pushing the action's probability up. With a negative advantage it is flat below 1 − ε. In the opposite directions the unclipped term is the minimum, so a step that made things worse is always corrected. Drag the vertical line to read the gradient at a ratio."
      controls={
        <>
          <ParamSlider label="clip range ε" param={eps} />
          <ParamSlider label="ratio r" param={ratio} />
        </>
      }
      readout={
        <>
          <Readout
            label="objective at r (A = +1, A = −1)"
            value={`${formatNumber(objective(r, 1, eps.value))}, ${formatNumber(objective(r, -1, eps.value))}`}
          />
          <Readout label="gradient in r (A = +1, A = −1)" value={`${gradient(1)}, ${gradient(-1)}`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="probability ratio r"
        yLabel="objective"
        xRange={[0, 2]}
        yRange={[-2, 2]}
        handles={[{ kind: 'x', at: r, label: 'r', onDrag: (x) => ratio.set(x) }]}
      />
    </Interactive>
  )
}
