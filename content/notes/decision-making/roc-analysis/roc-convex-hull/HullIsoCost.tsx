import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Vec2,
} from '@/components/viz'
import { expectedCost, lineInSquare, rocExample } from '../../_shared/rocExample'

/**
 * ROC space with an empirical ROC curve, its convex hull, and an iso-cost line whose slope is
 * m = (1 − π) c_FP / (π c_FN). The reader drags the line (it passes through the handle); the optimal iso-cost line
 * touches the hull at the vertex with the least expected cost.
 */
export function HullIsoCost() {
  const pi = useParam(0.3, { min: 0.05, max: 0.95, step: 0.01 })
  const logRatio = useParam(1, { min: -4, max: 4, step: 0.1 })
  const at = useParam(0.5, { min: 0, max: 1, step: 0.005 })
  const atY = useParam(0.5, { min: 0, max: 1, step: 0.005 })

  const data = useMemo(() => rocExample(), [])
  const cfp = 1
  const cfn = 2 ** logRatio.value
  const m = ((1 - pi.value) * cfp) / (pi.value * cfn)

  const best = useMemo(() => {
    let k = 0
    data.hull.forEach((p, i) => {
      if (expectedCost(p, pi.value, cfp, cfn) < expectedCost(data.hull[k], pi.value, cfp, cfn)) k = i
    })
    return data.hull[k]
  }, [data, pi.value, cfn])

  const point: Vec2 = [at.value, atY.value]
  const dragged = lineInSquare(point, m)
  const optimal = lineInSquare(best, m)
  const handles: Handle[] = [
    {
      kind: 'point',
      at: point,
      label: 'iso-cost line',
      onDrag: ([x, y]) => {
        at.set(x)
        atY.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="Iso-cost lines touch the ROC convex hull"
      caption="The stepped line is the empirical ROC curve of 25 positives and 25 negatives; the dashed polygon is its convex hull. Every point on a straight line of slope m = (1 − π)·c_FP / (π·c_FN) has the same expected cost. Drag the handle to slide your iso-cost line; costs fall towards the top left. The lowest-cost line that still touches an achievable point meets the hull at a vertex (ringed). Change the prevalence π or the cost ratio and the slope, and the optimal vertex, change."
      controls={
        <>
          <ParamSlider label="prevalence π" param={pi} />
          <ParamSlider label="log₂(c_FN / c_FP)" param={logRatio} />
        </>
      }
      readout={
        <>
          <Readout label="slope m" value={formatNumber(m)} />
          <Readout label="cost on your line" value={formatNumber(expectedCost(point, pi.value, cfp, cfn))} />
          <Readout label="optimal vertex (FPR, TPR)" value={`(${formatNumber(best[0])}, ${formatNumber(best[1])})`} />
          <Readout label="least cost" value={formatNumber(expectedCost(best, pi.value, cfp, cfn))} />
        </>
      }
    >
      <XYChart
        equalAspect
        xLabel="false-positive rate"
        yLabel="true-positive rate"
        xRange={[0, 1]}
        yRange={[0, 1]}
        handles={handles}
        series={[
          { name: 'chance', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
          { name: 'ROC curve', type: 'line', x: data.curve.map((p) => p[0]), y: data.curve.map((p) => p[1]), slot: 0 },
          {
            name: 'convex hull',
            type: 'line',
            x: data.hull.map((p) => p[0]),
            y: data.hull.map((p) => p[1]),
            slot: 1,
            dashed: true,
          },
          { name: 'your iso-cost line', type: 'line', x: dragged.x, y: dragged.y, slot: 2 },
          { name: 'optimal iso-cost line', type: 'line', x: optimal.x, y: optimal.y, emphasis: true },
          { name: 'optimal vertex', type: 'scatter', x: [best[0]], y: [best[1]], emphasis: true },
        ]}
      />
    </Interactive>
  )
}
