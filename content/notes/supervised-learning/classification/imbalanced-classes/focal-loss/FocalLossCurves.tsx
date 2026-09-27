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
} from '@/components/viz'
import { linspace } from '@/lib/math'

const PT = linspace(0.005, 0.995, 200)

/** Focal loss −(1 − p)^γ log p for the probability p of the true class. γ = 0 is cross-entropy. */
const focal = (p: number, gamma: number) => -((1 - p) ** gamma) * Math.log(p)
/** Magnitude of its derivative with respect to the logit: (1 − p)^γ ((1 − p) − γ p log p). */
const focalGrad = (p: number, gamma: number) => (1 - p) ** gamma * (1 - p - gamma * p * Math.log(p))

/** Cross-entropy and focal loss, and their gradients with respect to the logit, against the true-class probability. */
export function FocalLossCurves() {
  const gamma = useParam(2, { min: 0, max: 5, step: 0.1 })
  const pt = useParam(0.9, { min: 0.01, max: 0.99, step: 0.001 })

  const loss = useMemo<XYSeries[]>(
    () => [
      { name: 'cross-entropy (γ = 0)', type: 'line', x: PT, y: PT.map((p) => focal(p, 0)), slot: 0 },
      { name: 'focal loss', type: 'line', x: PT, y: PT.map((p) => focal(p, gamma.value)), slot: 1 },
    ],
    [gamma.value],
  )
  const grad = useMemo<XYSeries[]>(
    () => [
      { name: 'cross-entropy (γ = 0)', type: 'line', x: PT, y: PT.map((p) => focalGrad(p, 0)), slot: 0 },
      { name: 'focal loss', type: 'line', x: PT, y: PT.map((p) => focalGrad(p, gamma.value)), slot: 1 },
    ],
    [gamma.value],
  )

  const handles: Handle[] = [{ kind: 'x', at: pt.value, label: 'p_t', onDrag: (x) => pt.set(x) }]
  const lossRatio = focal(pt.value, gamma.value) / focal(pt.value, 0)
  const gradRatio = focalGrad(pt.value, gamma.value) / focalGrad(pt.value, 0)

  return (
    <Interactive
      title="Focal loss down-weights easy examples"
      caption="Left: loss against p_t, the probability the model gives the true class. Right: the size of the gradient with respect to the logit. Drag the vertical line to an example's p_t; the readout gives the factor by which focal loss scales its loss and its gradient relative to cross-entropy. Well-classified examples (p_t near 1) are suppressed by roughly (1 − p_t)^γ; badly classified ones keep most of their gradient."
      controls={
        <>
          <ParamSlider label="focusing parameter γ" param={gamma} />
          <ParamSlider label="true-class probability p_t" param={pt} />
        </>
      }
      readout={
        <>
          <Readout label="loss factor (1 − p_t)^γ" value={formatNumber(lossRatio)} />
          <Readout label="loss reduced by" value={`${formatNumber(1 / lossRatio)}×`} />
          <Readout label="gradient factor" value={formatNumber(gradRatio)} />
          <Readout label="gradient reduced by" value={`${formatNumber(1 / gradRatio)}×`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={loss} xLabel="p_t" yLabel="loss" xRange={[0, 1]} yRange={[0, 5]} handles={handles} />
        <XYChart
          series={grad}
          xLabel="p_t"
          yLabel="|∂ loss / ∂ logit|"
          xRange={[0, 1]}
          yRange={[0, 1.2]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
