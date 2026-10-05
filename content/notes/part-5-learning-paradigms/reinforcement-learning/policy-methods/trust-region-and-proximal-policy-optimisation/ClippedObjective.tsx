import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const R = toFlat(linspace(0, 2, 401))
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const objective = (r: number, adv: number, eps: number) => Math.min(r * adv, clip(r, 1 - eps, 1 + eps) * adv)

/** PPO's clipped surrogate for one sample as a function of the probability ratio, for a positive and a negative advantage. */
export function ClippedObjective() {
  const state = useFigureState({
    eps: float(0.2, { min: 0.05, max: 0.5, step: 0.01, label: 'clip range ε' }),
    ratio: slider(0, 2, 1.3, { step: 0.01, label: 'ratio r' }),
  })

  const series = useMemo(
    () =>
      [
        { name: 'clipped, A = +1', x: R, y: R.map((r) => objective(r, 1, state.eps)), slot: 0 },
        { name: 'unclipped r·A, A = +1', x: R, y: R, slot: 0, dashed: true },
        { name: 'clipped, A = −1', x: R, y: R.map((r) => objective(r, -1, state.eps)), slot: 1 },
        { name: 'unclipped r·A, A = −1', x: R, y: R.map((r) => -r), slot: 1, dashed: true },
      ] as const,
    [state.eps],
  )
  const r = state.ratio
  // The gradient with respect to r is A where the unclipped term is the minimum, and 0 where the clipped term is.
  const gradient = (adv: number) => (r * adv <= clip(r, 1 - state.eps, 1 + state.eps) * adv ? adv : 0)

  const xAxis = useAxis({ label: 'probability ratio r', range: [0, 2] })
  const yAxis = useAxis({ label: 'objective', range: [-2, 2] })
  return (
    <Figure
      title="The clipped surrogate objective"
      state={state}
      caption="One sample's contribution to PPO's objective, min(r A, clip(r, 1 − ε, 1 + ε) A), as a function of the probability ratio r = π_θ(a | s) / π_old(a | s). With a positive advantage the objective rises with r only up to 1 + ε and is flat beyond it, so the gradient stops pushing the action's probability up. With a negative advantage it is flat below 1 − ε. In the opposite directions the unclipped term is the minimum, so a step that made things worse is always corrected. Drag the vertical line to read the gradient at a ratio."

      readouts={
        <>
          <Readout
            label="objective at r (A = +1, A = −1)"
            value={`${formatNumber(objective(r, 1, state.eps))}, ${formatNumber(objective(r, -1, state.eps))}`}
          />
          <Readout label="gradient in r (A = +1, A = −1)" value={`${gradient(1)}, ${gradient(-1)}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Handle kind="x" at={r} label="r" onDrag={(x) => state.set('ratio', x)} />
      </Plot>
    </Figure>
  )
}
