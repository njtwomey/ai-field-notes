/** The training curves of the mixture density figures: NLL and squared error of both networks, with a step marker. */
import type { MdnSnapshot } from 'aifn-methods/learning/mixture-density'
import { Curve, Handle, Plot, Plots, useAxis } from '@lab/viz'
import { SLOT } from './shared'

/** The NLL and MSE curves of both networks against the step, with the checkpoint's step as a draggable marker. */
export function TrainingCurves({
  snap,
  step,
  onStep,
  runKey,
}: {
  snap: MdnSnapshot | undefined
  step: number | undefined
  onStep: (step: number) => void
  runKey: unknown
}) {
  const stepAxis = useAxis({ label: 'step', range: [0, snap?.steps ?? 1], key: snap?.steps, integer: true })
  const nllAxis = useAxis({ label: 'NLL per point', hold: 'union', key: runKey })
  const mseAxis = useAxis({ label: 'squared error', hold: 'union', key: runKey, log: true })
  const h = snap?.history
  const marker = step !== undefined ? <Handle kind="x" at={step} onDrag={onStep} label={`step ${step}`} /> : null
  return (
    <Plots cols={2} scale={0.7}>
      <Plot x={stepAxis} y={nllAxis} title="negative log-likelihood (training data)">
        {h && <Curve name="MDN" x={h.step} y={h.nll} slot={SLOT.mdn} />}
        {h && <Curve name="squared error (Gaussian, residual σ)" x={h.step} y={h.meanNll} slot={SLOT.mean} />}
        {marker}
      </Plot>
      <Plot x={stepAxis} y={mseAxis} title="squared error of E[y | x]">
        {h && <Curve name="MDN mean" x={h.step} y={h.mse} slot={SLOT.mdn} />}
        {h && <Curve name="squared-error network" x={h.step} y={h.meanMse} slot={SLOT.mean} />}
        {marker}
      </Plot>
    </Plots>
  )
}
