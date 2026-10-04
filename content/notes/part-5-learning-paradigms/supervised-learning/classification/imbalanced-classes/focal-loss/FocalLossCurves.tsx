import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const PT = toFlat(linspace(0.005, 0.995, 200))

/** Focal loss −(1 − p)^γ log p for the probability p of the true class. γ = 0 is cross-entropy. */
const focal = (p: number, gamma: number) => -((1 - p) ** gamma) * Math.log(p)
/** Magnitude of its derivative with respect to the logit: (1 − p)^γ ((1 − p) − γ p log p). */
const focalGrad = (p: number, gamma: number) => (1 - p) ** gamma * (1 - p - gamma * p * Math.log(p))

/** Cross-entropy and focal loss, and their gradients with respect to the logit, against the true-class probability. */
export function FocalLossCurves() {
  const state = useFigureState({
    gamma: float(2, { min: 0, max: 5, step: 0.1, label: 'focusing parameter γ' }),
    pt: slider(0.01, 0.99, 0.9, { step: 0.001, label: 'true-class probability p_t' }),
  })

  const loss = useMemo(
    () =>
      [
        { name: 'cross-entropy (γ = 0)', x: PT, y: PT.map((p) => focal(p, 0)), slot: 0 },
        { name: 'focal loss', x: PT, y: PT.map((p) => focal(p, state.gamma)), slot: 1 },
      ] as const,
    [state.gamma],
  )
  const grad = useMemo(
    () =>
      [
        { name: 'cross-entropy (γ = 0)', x: PT, y: PT.map((p) => focalGrad(p, 0)), slot: 0 },
        { name: 'focal loss', x: PT, y: PT.map((p) => focalGrad(p, state.gamma)), slot: 1 },
      ] as const,
    [state.gamma],
  )

  const lossRatio = focal(state.pt, state.gamma) / focal(state.pt, 0)
  const gradRatio = focalGrad(state.pt, state.gamma) / focalGrad(state.pt, 0)

  const xAxis = useAxis({ label: 'p_t', range: [0, 1] })
  const yAxis = useAxis({ label: 'loss', range: [0, 5] })
  const xAxis2 = useAxis({ label: 'p_t', range: [0, 1] })
  const yAxis2 = useAxis({ label: '|∂ loss / ∂ logit|', range: [0, 1.2] })
  return (
    <Figure
      title="Focal loss down-weights easy examples"
      state={state}
      caption="Left: loss against p_t, the probability the model gives the true class. Right: the size of the gradient with respect to the logit. Drag the vertical line to an example's p_t; the readout gives the factor by which focal loss scales its loss and its gradient relative to cross-entropy. Well-classified examples (p_t near 1) are suppressed by roughly (1 − p_t)^γ; badly classified ones keep most of their gradient."

      readouts={
        <>
          <Readout label="loss factor (1 − p_t)^γ" value={formatNumber(lossRatio)} />
          <Readout label="loss reduced by" value={`${formatNumber(1 / lossRatio)}×`} />
          <Readout label="gradient factor" value={formatNumber(gradRatio)} />
          <Readout label="gradient reduced by" value={`${formatNumber(1 / gradRatio)}×`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...loss[0]} />
          <Curve {...loss[1]} />
          <Handle {...state.handle('pt', { label: 'p_t' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...grad[0]} />
          <Curve {...grad[1]} />
          <Handle {...state.handle('pt', { label: 'p_t' })} />
        </Plot>
      </div>
    </Figure>
  )
}
