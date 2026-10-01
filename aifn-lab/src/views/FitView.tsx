import { useMemo } from 'react'
import {
  asTensor,
  capabilities,
  hasDecide,
  hasExpect,
  hasPredictive,
  isClassDistribution,
  isUnivariate,
  type Distribution,
} from 'aifn/learning/estimators'
import { fromData, linspace, reshape, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { PanelSlot } from '@lab/layout'
import { Curve, Plot, Points, Readout, useAxis } from '@lab/viz'
import { StateTree } from './StateTree'

export type FitPanelProps = {
  /** A fitted model from `aifn/learning/estimators`, `aifn/learning/compose` or any module with the same capabilities. */
  model: object
  /** The training data: inputs [n, d] and targets [n]. */
  data: { x: Tensor; y: Tensor }
  /** One-feature inputs only: the x range of the prediction curves (default: the data's range, padded by 5%). */
  xRange?: [number, number]
  /** Central probability of the predictive band (default 0.9). */
  interval?: number
  /** Build the model's input from a column of grid values [m, 1] (default: the column itself). */
  input?: (grid: Tensor) => unknown
  xLabel?: string
  yLabel?: string
}

/** The fitted state of a model: its fields that are not functions. */
function fittedState(model: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(model).filter(([, v]) => typeof v !== 'function'))
}

/**
 * A fitted model's predictions and fitted state. For one-feature inputs, the data with the model's point prediction
 * (`decide`), its expectation (`expect`; P(y = 1) for a Bernoulli) and a central band of its predictive; otherwise
 * predicted against observed targets. Under the chart, the fitted state as a tree (every field that is not a method).
 */
export function FitPanel({ model, data, xRange, interval = 0.9, input, xLabel = 'x', yLabel = 'y' }: FitPanelProps) {
  const x = toFlat(data.x)
  const y = toFlat(data.y)
  const oneFeature = data.x.shape.length === 1 || data.x.shape[1] === 1
  const range = useMemo((): [number, number] => {
    if (xRange) return xRange
    const lo = Math.min(...x)
    const hi = Math.max(...x)
    const pad = 0.05 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }, [x, xRange])
  const layers = useMemo(() => {
    if (!oneFeature) {
      const predict = hasExpect(model) ? model.expect : hasDecide(model) ? model.decide : null
      if (!predict) return null
      const p = toFlat(asTensor(predict(data.x) as Tensor))
      const lo = Math.min(...y, ...p)
      const hi = Math.max(...y, ...p)
      return (
        <>
          <Curve name="y = ŷ" x={[lo, hi]} y={[lo, hi]} muted dashed />
          <Points name="training rows" x={y} y={p} slot={0} />
        </>
      )
    }
    const g = toFlat(linspace(range[0], range[1], 200))
    const gridColumn = reshape(fromData(Float64Array.from(g)), [g.length, 1])
    const at = input ? input(gridColumn) : gridColumn
    const d: Distribution | null = hasPredictive(model) ? model.predictive(at) : null
    let band: { label: string; lower: number[]; upper: number[] } | null = null
    if (d && isUnivariate(d) && !isClassDistribution(d))
      band = {
        label: `${Math.round(interval * 100)}% predictive interval`,
        lower: toFlat(asTensor(d.quantile(fromData(Float64Array.of((1 - interval) / 2), [])))),
        upper: toFlat(asTensor(d.quantile(fromData(Float64Array.of((1 + interval) / 2), [])))),
      }
    return (
      <>
        <Points name="training rows" x={x} y={y} muted />
        {band && <Curve name={band.label} x={g} y={band.lower} slot={2} dashed />}
        {band && <Curve name={band.label} x={g} y={band.upper} slot={2} dashed />}
        {hasDecide(model) && <Curve name="decide(x)" x={g} y={toFlat(asTensor(model.decide(at) as Tensor))} slot={0} />}
        {hasExpect(model) && (
          <Curve
            name={d && isClassDistribution(d) ? 'P(y = 1 | x)' : 'E[y | x]'}
            x={g}
            y={toFlat(model.expect(at))}
            slot={1}
          />
        )}
      </>
    )
  }, [model, data, oneFeature, range, input, interval, x, y])
  const xAxis = useAxis({ label: oneFeature ? xLabel : `observed ${yLabel}`, range: oneFeature ? range : undefined })
  const yAxis = useAxis({ label: oneFeature ? yLabel : `predicted ${yLabel}` })
  const state = useMemo(() => fittedState(model), [model])
  return (
    <>
      <PanelSlot slot="readouts">
        <Readout label="capabilities" value={capabilities(model).join(', ') || 'none'} />
      </PanelSlot>
      <div className="flex flex-col gap-3">
        {/* Three quarters of the frame's height; the fitted state sits under it. */}
        <Plot x={xAxis} y={yAxis} scale={0.75}>
          {layers}
        </Plot>
        <div className="max-h-72 overflow-auto rounded-md border p-2">
          <StateTree value={state} name="fitted state" />
        </div>
      </div>
    </>
  )
}
