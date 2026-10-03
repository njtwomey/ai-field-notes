import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { jsDiv, klDiv, monotonePlan, normalise, tvDist } from '../_shared/ot'

const N = 8
const BINS = Array.from({ length: N }, (_, i) => i + 1)
const MAX_H = 0.45
const Y_RANGE: [number, number] = [-0.5, 0.5]
const X_RANGE: [number, number] = [0, N + 1]
const CDF_X: [number, number] = [0, N + 1]

const PRESETS = {
  overlapping: {
    a: [0.05, 0.2, 0.3, 0.25, 0.1, 0.05, 0.03, 0.02],
    b: [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1],
  },
  near: { a: [0.3, 0.3, 0, 0, 0, 0, 0, 0], b: [0, 0, 0.3, 0.3, 0, 0, 0, 0] },
  far: { a: [0.3, 0.3, 0, 0, 0, 0, 0, 0], b: [0, 0, 0, 0, 0, 0, 0.3, 0.3] },
}

const clamp = (v: number) => Math.min(Math.max(v, 0), MAX_H)
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '∞')

/** CDF as a step function that jumps at each bin centre k by that bin's mass. */
function steps(values: number[]) {
  const x: number[] = [CDF_X[0], 1]
  const y: number[] = [0, 0]
  values.forEach((v, k) => {
    x.push(k + 1, k + 2)
    y.push(v, v)
  })
  x[x.length - 1] = CDF_X[1]
  return { x, y }
}

/**
 * Two histograms on eight bins. Bars are draggable; the optimal plan for the cost |i - j| is the monotone coupling,
 * drawn as arrows and as a matrix. The EMD equals the area between the two CDFs.
 */
export function HistogramTransport() {
  const [a, setA] = useState(PRESETS.overlapping.a)
  const [b, setB] = useState(PRESETS.overlapping.b)

  const r = useMemo(() => {
    const safeA = a.some((v) => v > 0) ? a : a.map(() => 1)
    const safeB = b.some((v) => v > 0) ? b : b.map(() => 1)
    const p = normalise(safeA)
    const q = normalise(safeB)
    const plan = monotonePlan(p, q)
    const emd = plan.reduce((s, e) => s + e.mass * Math.abs(e.i - e.j), 0)
    const z = BINS.map((_, i) => BINS.map((_, j) => plan.find((e) => e.i === i && e.j === j)?.mass ?? 0))
    const moved = plan.reduce((s, e) => s + (e.i === e.j ? 0 : e.mass), 0)
    const vectors: Segment[] = plan
      .filter((e) => e.i !== e.j && e.mass > 0.005)
      .map((e) => ({ from: [e.i + 1, a[e.i] / 2], to: [e.j + 1, -b[e.j] / 2] }))
    let fa = 0
    let fb = 0
    const cdfA: number[] = []
    const cdfB: number[] = []
    const gap: number[] = []
    p.forEach((v, k) => {
      fa += v
      fb += q[k]
      cdfA.push(fa)
      cdfB.push(fb)
      gap.push(Math.abs(fa - fb))
    })
    return { p, q, z, emd, moved, vectors, cdfA, cdfB, gap }
  }, [a, b])

  const bars = useMemo(
    (): XYSeries[] => [
      { name: 'a (source)', type: 'bar', x: BINS, y: a, slot: 0 },
      { name: 'b (target, drawn downwards)', type: 'bar', x: BINS, y: b.map((v) => -v), slot: 1 },
    ],
    [a, b],
  )
  const handles: Handle[] = [
    ...BINS.map((x, i): Handle => ({
      kind: 'point',
      at: [x, a[i]],
      label: `a${x}`,
      onDrag: ([, y]) => setA((prev) => prev.map((v, k) => (k === i ? clamp(y) : v))),
    })),
    ...BINS.map((x, i): Handle => ({
      kind: 'point',
      at: [x, -b[i]],
      label: `b${x}`,
      onDrag: ([, y]) => setB((prev) => prev.map((v, k) => (k === i ? clamp(-y) : v))),
    })),
  ]

  const cdfSeries = useMemo((): XYSeries[] => {
    const sa = steps(r.cdfA)
    const sb = steps(r.cdfB)
    const sg = steps(r.gap)
    return [
      { name: '|A − B| (area = EMD)', type: 'line', x: sg.x, y: sg.y, muted: true, area: true },
      { name: 'CDF of a', type: 'line', x: sa.x, y: sa.y, slot: 0 },
      { name: 'CDF of b', type: 'line', x: sb.x, y: sb.y, slot: 1 },
    ]
  }, [r])

  const preset = (k: keyof typeof PRESETS) => () => {
    setA(PRESETS[k].a)
    setB(PRESETS[k].b)
  }

  return (
    <Interactive
      title="Moving one histogram onto another"
      caption="Drag the top of any bar of a, or the bottom of any bar of b, to change its mass. Each histogram is rescaled to total mass 1. Arrows show mass that moves between bins, from the middle of a source bar to the middle of a target bar; the matrix shows how much. The EMD is the shaded area between the two CDFs. With the disjoint presets, total variation and Jensen–Shannon stay at their maxima while the EMD grows with the gap."
      controls={
        <>
          <ParamButton onClick={preset('overlapping')}>Overlapping</ParamButton>
          <ParamButton onClick={preset('near')}>Disjoint, near</ParamButton>
          <ParamButton onClick={preset('far')}>Disjoint, far</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="EMD (bins)" value={formatNumber(r.emd)} />
          <Readout label="mass moved" value={formatNumber(r.moved)} />
          <Readout label="TV" value={formatNumber(tvDist(r.p, r.q))} />
          <Readout label="JS (nats, max 0.693)" value={formatNumber(jsDiv(r.p, r.q))} />
          <Readout label="KL(a ‖ b)" value={fmt(klDiv(r.p, r.q))} />
        </>
      }
    >
      <XYChart
        height={300}
        series={bars}
        vectors={r.vectors}
        xRange={X_RANGE}
        yRange={Y_RANGE}
        xLabel="bin"
        yLabel="mass"
        handles={handles}
      />
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          height={280}
          x={BINS}
          y={BINS}
          z={r.z}
          range={[0, 0.3]}
          xLabel="target bin j"
          yLabel="source bin i"
          valueLabel="mass P_ij"
        />
        <XYChart
          height={280}
          series={cdfSeries}
          xRange={CDF_X}
          yRange={[0, 1.05]}
          xLabel="bin"
          yLabel="cumulative mass"
        />
      </div>
    </Interactive>
  )
}
