import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type Vec2,
  type XYSeries,
} from 'aifn-render'

const RANGE: [number, number] = [-3, 3]
const ANCHOR: Vec2 = [0, 0]
const CIRCLE = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
// Arrow length per unit of gradient, so that the descent directions stay inside the plot.
const ARROW_SCALE = 0.25

const clamp = (v: number) => Math.min(Math.max(v, RANGE[0]), RANGE[1])
const d2 = (u: Vec2, v: Vec2) => (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2

function circle(r: number) {
  return { x: CIRCLE.map((t) => r * Math.cos(t)), y: CIRCLE.map((t) => r * Math.sin(t)) }
}

/**
 * One triplet in a 2-D embedding. The anchor sits at the origin; the positive and the negative are draggable. The two
 * circles split the plane into the hard, semi-hard and easy regions for the negative.
 */
export function TripletMargin() {
  const [pos, setPos] = useState<Vec2>([1, 0.4])
  const [neg, setNeg] = useState<Vec2>([0.4, 1])
  const alpha = useParam(1, { min: 0, max: 3, step: 0.1 })

  const r = useMemo(() => {
    const dap = d2(ANCHOR, pos)
    const dan = d2(ANCHOR, neg)
    const loss = Math.max(0, dap - dan + alpha.value)
    const kind = dan < dap ? 'hard' : dan < dap + alpha.value ? 'semi-hard' : 'easy'
    const inner = circle(Math.sqrt(dap))
    const outer = circle(Math.sqrt(dap + alpha.value))
    const series: XYSeries[] = [
      { name: 'hard boundary: d(a, n)² = d(a, p)²', type: 'line', ...inner, slot: 0, dashed: true },
      { name: 'margin boundary: d(a, n)² = d(a, p)² + α', type: 'line', ...outer, slot: 1, dashed: true },
      { name: 'anchor', type: 'scatter', x: [ANCHOR[0]], y: [ANCHOR[1]], emphasis: true },
      { name: 'positive', type: 'scatter', x: [pos[0]], y: [pos[1]], slot: 0 },
      { name: 'negative', type: 'scatter', x: [neg[0]], y: [neg[1]], slot: 1 },
    ]
    // Gradient descent moves p along −∂L/∂p = 2(a − p) and n along −∂L/∂n = 2(n − a) while the triplet is active.
    const vectors: Segment[] =
      loss > 0
        ? [
            {
              from: pos,
              to: [pos[0] + ARROW_SCALE * 2 * (ANCHOR[0] - pos[0]), pos[1] + ARROW_SCALE * 2 * (ANCHOR[1] - pos[1])],
            },
            {
              from: neg,
              to: [neg[0] + ARROW_SCALE * 2 * (neg[0] - ANCHOR[0]), neg[1] + ARROW_SCALE * 2 * (neg[1] - ANCHOR[1])],
            },
          ]
        : []
    return { dap, dan, loss, kind, series, vectors }
  }, [pos, neg, alpha.value])

  const handles: Handle[] = [
    { kind: 'point', at: pos, label: 'positive', onDrag: ([x, y]) => setPos([clamp(x), clamp(y)]) },
    { kind: 'point', at: neg, label: 'negative', onDrag: ([x, y]) => setNeg([clamp(x), clamp(y)]) },
  ]

  return (
    <Interactive
      title="Triplet loss: hard, semi-hard and easy negatives"
      caption="Drag the positive and the negative; the anchor stays at the origin. Inside the inner circle the negative is closer to the anchor than the positive is (hard). Between the circles it is farther, but by less than the margin α (semi-hard). Outside the outer circle the triplet is satisfied and contributes no loss or gradient (easy). While the loss is positive, the arrows show the gradient-descent step: the positive moves toward the anchor and the negative moves directly away from it."
      controls={<ParamSlider label="margin α" param={alpha} />}
      readout={
        <>
          <Readout label="d(a, p)²" value={formatNumber(r.dap)} />
          <Readout label="d(a, n)²" value={formatNumber(r.dan)} />
          <Readout label="loss" value={formatNumber(r.loss)} />
          <Readout label="negative is" value={r.kind} />
        </>
      }
    >
      <XYChart
        series={r.series}
        vectors={r.vectors}
        handles={handles}
        xRange={RANGE}
        yRange={RANGE}
        equalAspect
        xLabel="embedding dimension 1"
        yLabel="embedding dimension 2"
      />
    </Interactive>
  )
}
