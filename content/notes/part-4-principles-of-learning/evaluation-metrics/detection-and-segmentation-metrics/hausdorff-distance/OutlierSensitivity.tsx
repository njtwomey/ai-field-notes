import { useMemo, useState } from 'react'
import { Interactive, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'

type Vec = [number, number]
const N = 40
const VIEW: [number, number] = [0, 12]

const ring = (cx: number, cy: number, rx: number, ry: number): Vec[] =>
  Array.from({ length: N }, (_, i) => {
    const t = (2 * Math.PI * i) / N
    return [cx + rx * Math.cos(t), cy + ry * Math.sin(t)]
  })

const TRUTH = ring(6, 6, 3, 3)
const PRED = ring(6.3, 5.8, 3.2, 2.8)

const dist = (a: Vec, b: Vec) => Math.hypot(a[0] - b[0], a[1] - b[1])
/** Distance from each point of A to its nearest point of B. */
const nearest = (a: Vec[], b: Vec[]) => a.map((p) => Math.min(...b.map((q) => dist(p, q))))
const percentile = (xs: number[], q: number) => {
  const s = [...xs].sort((u, v) => u - v)
  const pos = (s.length - 1) * q
  const lo = Math.floor(pos)
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo)
}

/**
 * Ground-truth and predicted boundaries, with one extra predicted point that can be dragged away. The Hausdorff
 * distance follows the worst point; the 95th-percentile distance and the average surface distance barely move.
 */
export function OutlierSensitivity() {
  const [stray, setStray] = useState<Vec>([6.3, 8.6])

  const r = useMemo(() => {
    const pred = [...PRED, stray]
    const ab = nearest(TRUTH, pred)
    const ba = nearest(pred, TRUTH)
    const both = [...ab, ...ba]
    return {
      hAB: Math.max(...ab),
      hBA: Math.max(...ba),
      hd95: percentile(both, 0.95),
      assd: both.reduce((s, v) => s + v, 0) / both.length,
      pred,
    }
  }, [stray])

  const pts = (name: string, p: Vec[], slot: number): XYSeries => ({
    name,
    type: 'scatter',
    x: p.map((v) => v[0]),
    y: p.map((v) => v[1]),
    slot,
  })
  const handles: Handle[] = [
    {
      kind: 'point',
      at: stray,
      label: 'stray point',
      onDrag: ([x, y]) => setStray([Math.min(11.5, Math.max(0.5, x)), Math.min(11.5, Math.max(0.5, y))]),
    },
  ]

  return (
    <Interactive
      title="One stray point sets the Hausdorff distance"
      caption="Two boundaries: the ground truth and a close prediction, plus one extra predicted point that can be dragged. The Hausdorff distance is the largest distance from any point of either boundary to the other, so it jumps as soon as the stray point leaves the ring. The 95th-percentile Hausdorff distance and the average symmetric surface distance use the whole distribution of distances and barely change."
      readout={
        <>
          <Readout label="h(truth → prediction)" value={formatNumber(r.hAB)} />
          <Readout label="h(prediction → truth)" value={formatNumber(r.hBA)} />
          <Readout label="Hausdorff distance" value={formatNumber(Math.max(r.hAB, r.hBA))} />
          <Readout label="HD95" value={formatNumber(r.hd95)} />
          <Readout label="average surface distance" value={formatNumber(r.assd)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart
          series={[pts('ground-truth boundary', TRUTH, 0), pts('predicted boundary', r.pred, 1)]}
          xRange={VIEW}
          yRange={VIEW}
          equalAspect
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
