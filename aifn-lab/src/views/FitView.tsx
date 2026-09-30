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
import { Figure } from '@lab/layout'
import { ChartSize, Readout, XYChart, type XYSeries } from '@lab/viz'
import type { FrameProps } from './frame'
import { StateTree } from './StateTree'

export type FitViewProps = FrameProps & {
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
export function FitView({
  model,
  data,
  xRange,
  interval = 0.9,
  input,
  xLabel = 'x',
  yLabel = 'y',
  title,
  controls,
  readouts,
  ...frame
}: FitViewProps) {
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
  const series = useMemo((): XYSeries[] => {
    if (!oneFeature) {
      const predict = hasExpect(model) ? model.expect : hasDecide(model) ? model.decide : null
      if (!predict) return []
      const p = toFlat(asTensor(predict(data.x) as Tensor))
      const lo = Math.min(...y, ...p)
      const hi = Math.max(...y, ...p)
      return [
        { name: 'y = ŷ', type: 'line', x: [lo, hi], y: [lo, hi], muted: true, dashed: true },
        { name: 'training rows', type: 'scatter', x: y, y: p, slot: 0 },
      ]
    }
    const g = toFlat(linspace(range[0], range[1], 200))
    const gridColumn = reshape(fromData(Float64Array.from(g)), [g.length, 1])
    const at = input ? input(gridColumn) : gridColumn
    const out: XYSeries[] = [{ name: 'training rows', type: 'scatter', x, y, muted: true }]
    const d: Distribution | null = hasPredictive(model) ? model.predictive(at) : null
    if (d && isUnivariate(d) && !isClassDistribution(d)) {
      const lower = toFlat(asTensor(d.quantile(fromData(Float64Array.of((1 - interval) / 2), []))))
      const upper = toFlat(asTensor(d.quantile(fromData(Float64Array.of((1 + interval) / 2), []))))
      const label = `${Math.round(interval * 100)}% predictive interval`
      out.push({ name: label, type: 'line', x: g, y: lower, slot: 2, dashed: true })
      out.push({ name: label, type: 'line', x: g, y: upper, slot: 2, dashed: true })
    }
    if (hasDecide(model))
      out.push({ name: 'decide(x)', type: 'line', x: g, y: toFlat(asTensor(model.decide(at) as Tensor)), slot: 0 })
    if (hasExpect(model)) {
      const name = d && isClassDistribution(d) ? 'P(y = 1 | x)' : 'E[y | x]'
      out.push({ name, type: 'line', x: g, y: toFlat(model.expect(at)), slot: 1 })
    }
    return out
  }, [model, data, oneFeature, range, input, interval, x, y])
  const state = useMemo(() => fittedState(model), [model])
  return (
    <Figure
      title={title ?? 'Fitted model'}
      defaultSize="L"
      {...frame}
      controls={controls}
      readouts={
        <>
          {readouts}
          <Readout label="capabilities" value={capabilities(model).join(', ') || 'none'} />
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <ChartSize scale={0.75}>
          <XYChart
            series={series}
            xLabel={oneFeature ? xLabel : `observed ${yLabel}`}
            yLabel={oneFeature ? yLabel : `predicted ${yLabel}`}
            xRange={oneFeature ? range : undefined}
          />
        </ChartSize>
        <div className="max-h-72 overflow-auto rounded-md border p-2">
          <StateTree value={state} name="fitted state" />
        </div>
      </div>
    </Figure>
  )
}
