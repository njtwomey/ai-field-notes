import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  type Vec2,
} from 'aifn-render'
import { expectedCost, lineInSquare, rocExample } from '../../_shared/rocExample'

/**
 * ROC space with an empirical ROC curve, its convex hull, and an iso-cost line whose slope is
 * m = (1 − π) c_FP / (π c_FN). The reader drags the line (it passes through the handle); the optimal iso-cost line
 * touches the hull at the vertex with the least expected cost.
 */
export function HullIsoCost() {
  const state = useFigureState({
    pi: slider(0.05, 0.95, 0.3, { step: 0.01, label: 'prevalence π' }),
    logRatio: float(1, { min: -4, max: 4, step: 0.1, label: 'log₂(c_FN / c_FP)' }),
    at: slider(0, 1, 0.5, { step: 0.005, onChart: true }),
    atY: slider(0, 1, 0.5, { step: 0.005, onChart: true }),
  })

  const data = useMemo(() => rocExample(), [])
  const cfp = 1
  const cfn = 2 ** state.logRatio
  const m = ((1 - state.pi) * cfp) / (state.pi * cfn)

  const best = useMemo(() => {
    let k = 0
    data.hull.forEach((p, i) => {
      if (expectedCost(p, state.pi, cfp, cfn) < expectedCost(data.hull[k], state.pi, cfp, cfn)) k = i
    })
    return data.hull[k]
  }, [data, state.pi, cfn])

  const point: Vec2 = [state.at, state.atY]
  const dragged = lineInSquare(point, m)
  const optimal = lineInSquare(best, m)

  const xAxis = useAxis({ label: 'false-positive rate', range: [0, 1] })
  const yAxis = useAxis({ label: 'true-positive rate', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="Iso-cost lines touch the ROC convex hull"
      state={state}
      caption="The stepped line is the empirical ROC curve of 25 positives and 25 negatives; the dashed polygon is its convex hull. Every point on a straight line of slope m = (1 − π)·c_FP / (π·c_FN) has the same expected cost. Drag the handle to slide your iso-cost line; costs fall towards the top left. The lowest-cost line that still touches an achievable point meets the hull at a vertex (ringed). Change the prevalence π or the cost ratio and the slope, and the optimal vertex, change."

      readouts={
        <>
          <Readout label="slope m" value={formatNumber(m)} />
          <Readout label="cost on your line" value={formatNumber(expectedCost(point, state.pi, cfp, cfn))} />
          <Readout label="optimal vertex (FPR, TPR)" value={`(${formatNumber(best[0])}, ${formatNumber(best[1])})`} />
          <Readout label="least cost" value={formatNumber(expectedCost(best, state.pi, cfp, cfn))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="chance" x={[0, 1]} y={[0, 1]} dashed muted />
        <Curve name="ROC curve" x={data.curve.map((p) => p[0])} y={data.curve.map((p) => p[1])} slot={0} />
        <Curve name="convex hull" x={data.hull.map((p) => p[0])} y={data.hull.map((p) => p[1])} slot={1} dashed />
        <Curve name="your iso-cost line" x={dragged.x} y={dragged.y} slot={2} />
        <Curve name="optimal iso-cost line" x={optimal.x} y={optimal.y} emphasis />
        <Points name="optimal vertex" x={[best[0]]} y={[best[1]]} emphasis />
        <Handle
          kind="point"
          at={point}
          label="iso-cost line"
          onDrag={([x, y]) => {
            state.set('at', x)
            state.set('atY', y)
          }}
        />
      </Plot>
    </Figure>
  )
}
