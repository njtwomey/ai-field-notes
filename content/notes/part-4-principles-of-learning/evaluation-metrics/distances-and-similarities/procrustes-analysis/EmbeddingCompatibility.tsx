import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { apply, svd2, type Mat2, type Vec2 } from '@/lib/math/mat2'

const N = 10
const RANGE = 3

/** N points from a seeded standard normal, centred so translation plays no part in the alignment. */
function seedPoints(seed: number, scale: number): Vec2[] {
  const draw = rng(seed)
  const pts: Vec2[] = Array.from({ length: N }, () => [draw.normal() * scale, draw.normal() * scale])
  const mx = pts.reduce((s, p) => s + p[0], 0) / N
  const my = pts.reduce((s, p) => s + p[1], 0) / N
  return pts.map(([x, y]) => [x - mx, y - my])
}

// Fixed once at module load: the "old" embedding and a set of unit-scale noise directions for the "new" one.
const OLD_POINTS = seedPoints(7, 1.15)
const BASE_NOISE = seedPoints(13, 1)

const rotate = ([x, y]: Vec2, deg: number): Vec2 => {
  const t = (deg * Math.PI) / 180
  const [c, s] = [Math.cos(t), Math.sin(t)]
  return [c * x - s * y, s * x + c * y]
}
const add = ([ax, ay]: Vec2, [bx, by]: Vec2, k = 1): Vec2 => [ax + k * bx, ay + k * by]
const dist = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1])

/** Orthogonal Procrustes rotation R = V Uᵀ mapping `newPts` onto `oldPts`, from the 2×2 SVD of Σᵢ newᵢ oldᵢᵀ. */
function fitRotation(newPts: Vec2[], oldPts: Vec2[]): Mat2 {
  let a = 0
  let b = 0
  let c = 0
  let d = 0
  newPts.forEach(([nx, ny], i) => {
    const [ox, oy] = oldPts[i]
    a += nx * ox
    b += nx * oy
    c += ny * ox
    d += ny * oy
  })
  const { u, v } = svd2(a, b, c, d)
  return [
    [v[0][0] * u[0][0] + v[1][0] * u[1][0], v[0][0] * u[0][1] + v[1][0] * u[1][1]],
    [v[0][1] * u[0][0] + v[1][1] * u[1][0], v[0][1] * u[0][1] + v[1][1] * u[1][1]],
  ]
}

/** Nearest old-space index to each shown point, and the fraction that lands back on the same item. */
function agreement(shown: Vec2[], oldPts: Vec2[]): number {
  let hits = 0
  shown.forEach((p, i) => {
    let best = 0
    let bestD = Infinity
    oldPts.forEach((q, j) => {
      const dd = dist(p, q)
      if (dd < bestD) {
        bestD = dd
        best = j
      }
    })
    if (best === i) hits++
  })
  return hits / shown.length
}

/**
 * A shared set of items in an old embedding and a newly retrained one, the second an unknown rotation of the first
 * plus per-item noise. Aligning fits R from all the items and applies it to all of them; dragging the highlighted
 * item (with alignment off) gives it real drift instead, which alignment cannot remove.
 */
export function EmbeddingCompatibility() {
  const theta = useParam(130, { min: 0, max: 360, step: 1 })
  const noise = useParam(0.12, { min: 0, max: 0.6, step: 0.01 })
  const [aligned, setAligned] = useState(false)
  const [drift, setDrift] = useState<Vec2 | null>(null)

  const newRaw = useMemo<Vec2[]>(
    () =>
      OLD_POINTS.map((p, i) => {
        if (i === 0 && drift) return drift
        return add(rotate(p, theta.value), BASE_NOISE[i], noise.value)
      }),
    [theta.value, noise.value, drift],
  )

  const rotation = useMemo(() => fitRotation(newRaw, OLD_POINTS), [newRaw])
  const shown = useMemo(() => (aligned ? newRaw.map((p) => apply(rotation, p)) : newRaw), [aligned, newRaw, rotation])
  const nnAgreement = useMemo(() => agreement(shown, OLD_POINTS), [shown])
  const meanGap = useMemo(() => shown.reduce((s, p, i) => s + dist(p, OLD_POINTS[i]), 0) / N, [shown])

  const series: XYSeries[] = [
    { name: 'old model', type: 'scatter', x: OLD_POINTS.map((p) => p[0]), y: OLD_POINTS.map((p) => p[1]), slot: 0 },
    {
      name: aligned ? 'new model, aligned' : 'new model, raw',
      type: 'scatter',
      x: shown.map((p) => p[0]),
      y: shown.map((p) => p[1]),
      slot: 1,
    },
  ]
  const segments: Segment[] = OLD_POINTS.map((p, i) => ({ from: p, to: shown[i] }))
  const handles: Handle[] = aligned
    ? []
    : [{ kind: 'point', at: newRaw[0], label: 'drifted item', onDrag: (p) => setDrift(p) }]

  return (
    <Interactive
      title="Realigning a retrained embedding"
      caption="Ten shared items in an old embedding (one colour) and a newly retrained one (the other), related by an unknown rotation plus small per-item noise. Toggle alignment to fit R from all ten and watch it snap into place. With alignment off, drag the highlighted item to give it real drift instead of noise: alignment still realigns everything else, but that item's gap does not close."
      controls={
        <>
          <ParamSlider label="rotation of the new space" param={theta} format={(v) => `${Math.round(v)}°`} />
          <ParamSlider label="ordinary noise" param={noise} />
          <ParamSwitch label="align to the old space" checked={aligned} onChange={setAligned} />
          <ParamButton onClick={() => setDrift(null)} disabled={!drift}>
            Undrift item
          </ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="nearest-neighbour agreement" value={`${Math.round(nnAgreement * 100)}%`} />
          <Readout label="mean gap to old position" value={formatNumber(meanGap)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          series={series}
          segments={segments}
          handles={handles}
          xRange={[-RANGE, RANGE]}
          yRange={[-RANGE, RANGE]}
          equalAspect
          bare
        />
      </div>
    </Interactive>
  )
}
