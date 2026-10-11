import { useMemo, useState } from 'react'
import {
  Area,
  Bars,
  Button,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Raster,
  Readout,
  type Segment,
  useAxis,
  Vectors,
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
    () =>
      [
        { name: 'a (source)', x: BINS, y: a, slot: 0 },
        { name: 'b (target, drawn downwards)', x: BINS, y: b.map((v) => -v), slot: 1 },
      ] as const,
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

  const cdfSeries = useMemo(() => {
    const sa = steps(r.cdfA)
    const sb = steps(r.cdfB)
    const sg = steps(r.gap)
    return [
      { name: '|A − B| (area = EMD)', x: sg.x, y: sg.y, muted: true },
      { name: 'CDF of a', x: sa.x, y: sa.y, slot: 0 },
      { name: 'CDF of b', x: sb.x, y: sb.y, slot: 1 },
    ] as const
  }, [r])

  const preset = (k: keyof typeof PRESETS) => () => {
    setA(PRESETS[k].a)
    setB(PRESETS[k].b)
  }

  const xAxis = useAxis({ label: 'bin', range: X_RANGE })
  const yAxis = useAxis({ label: 'mass', range: Y_RANGE })
  const xAxis2 = useAxis({ label: 'target bin j' })
  const yAxis2 = useAxis({ label: 'source bin i' })
  const xAxis3 = useAxis({ label: 'bin', range: CDF_X })
  const yAxis3 = useAxis({ label: 'cumulative mass', range: [0, 1.05] })
  return (
    <Figure
      title="Moving one histogram onto another"
      caption="Drag the top of any bar of a, or the bottom of any bar of b, to change its mass. Each histogram is rescaled to total mass 1. Arrows show mass that moves between bins, from the middle of a source bar to the middle of a target bar; the matrix shows how much. The EMD is the shaded area between the two CDFs. With the disjoint presets, total variation and Jensen–Shannon stay at their maxima while the EMD grows with the gap."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={preset('overlapping')}>
            Overlapping
          </Button>
          <Button variant="outline" size="sm" onClick={preset('near')}>
            Disjoint, near
          </Button>
          <Button variant="outline" size="sm" onClick={preset('far')}>
            Disjoint, far
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="EMD (bins)" value={formatNumber(r.emd)} />
          <Readout label="mass moved" value={formatNumber(r.moved)} />
          <Readout label="TV" value={formatNumber(tvDist(r.p, r.q))} />
          <Readout label="JS (nats, max 0.693)" value={formatNumber(jsDiv(r.p, r.q))} />
          <Readout label="KL(a ‖ b)" value={fmt(klDiv(r.p, r.q))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...bars[0]} />
        <Bars {...bars[1]} />
        <Vectors vectors={r.vectors} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Raster x={BINS} y={BINS} z={r.z} range={[0, 0.3]} valueLabel={'mass P_ij'} />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={280}>
          <Area {...cdfSeries[0]} />
          <Curve {...cdfSeries[1]} />
          <Curve {...cdfSeries[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
