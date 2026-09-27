import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

/** Minimise f(x, y) = (x − 2)² + 2(y − 1)² on the unit circle g(x, y) = x² + y² − 1 = 0. */
const f = (x: number, y: number) => (x - 2) ** 2 + 2 * (y - 1) ** 2
const gradF = (x: number, y: number): [number, number] => [2 * (x - 2), 4 * (y - 1)]
const gradG = (x: number, y: number): [number, number] => [2 * x, 2 * y]

const DEG = Math.PI / 180
const onCircle = (deg: number): [number, number] => [Math.cos(deg * DEG), Math.sin(deg * DEG)]
const X_RANGE: [number, number] = [-1.6, 3.4]
const Y_RANGE: [number, number] = [-1.6, 2.2]
const ARROW = 0.7

/** The constrained minimiser and maximiser, found once by a fine scan of the circle. */
const SCAN = linspace(0, 360, 36001)
const VALUES = SCAN.map((d) => f(...onCircle(d)))
const MIN_DEG = SCAN[VALUES.indexOf(Math.min(...VALUES))]
const MAX_DEG = SCAN[VALUES.indexOf(Math.max(...VALUES))]

/** Level set f = c: an ellipse centred at (2, 1) with semi-axes √c and √(c/2). */
function ellipse(c: number) {
  const t = linspace(0, 2 * Math.PI, 181)
  return { x: t.map((s) => 2 + Math.sqrt(c) * Math.cos(s)), y: t.map((s) => 1 + Math.sqrt(c / 2) * Math.sin(s)) }
}

const CIRCLE = (() => {
  const t = linspace(0, 360, 181)
  return { x: t.map((d) => Math.cos(d * DEG)), y: t.map((d) => Math.sin(d * DEG)) }
})()

/** Background contours of f, joined into one series with NaN breaks so that they share one legend entry. */
const CONTOURS = (() => {
  const x: number[] = []
  const y: number[] = []
  for (const c of [0.25, 1, 3, 6, 10, 15]) {
    const e = ellipse(c)
    x.push(...e.x, NaN)
    y.push(...e.y, NaN)
  }
  return { x, y }
})()

const PROFILE_X = linspace(0, 360, 361)
const PROFILE_Y = PROFILE_X.map((d) => f(...onCircle(d)))

export function LagrangeExplorer() {
  const angle = useParam(120, { min: 0, max: 360, step: 0.5 })
  const [px, py] = onCircle(angle.value)
  const gf = gradF(px, py)
  const gg = gradG(px, py)
  const norm = (v: [number, number]) => Math.hypot(v[0], v[1])
  // Least-squares multiplier: the λ that makes λ∇g closest to ∇f. It is exact at a stationary point.
  const lambda = (gf[0] * gg[0] + gf[1] * gg[1]) / (gg[0] ** 2 + gg[1] ** 2)
  const cosAngle = (gf[0] * gg[0] + gf[1] * gg[1]) / (norm(gf) * norm(gg))
  const angleBetween = Math.acos(Math.max(-1, Math.min(1, cosAngle))) / DEG
  // Derivative of f along the circle, per radian: ∇f · (−sin θ, cos θ).
  const tangential = -gf[0] * py + gf[1] * px

  const series = useMemo((): XYSeries[] => {
    const level = ellipse(f(px, py))
    return [
      { name: 'contours of f', type: 'line', x: CONTOURS.x, y: CONTOURS.y, muted: true },
      { name: 'constraint g = 0', type: 'line', x: CIRCLE.x, y: CIRCLE.y, slot: 0 },
      { name: 'level set of f through the point', type: 'line', x: level.x, y: level.y, slot: 1, dashed: true },
      {
        name: 'constrained minimum',
        type: 'scatter',
        x: [onCircle(MIN_DEG)[0]],
        y: [onCircle(MIN_DEG)[1]],
        emphasis: true,
      },
    ]
  }, [px, py])

  const vectors: Segment[] = [
    { from: [px, py], to: [px + (ARROW * gf[0]) / norm(gf), py + (ARROW * gf[1]) / norm(gf)] },
  ]
  // The normal line of the circle through the point, along ∇g, drawn as a thin guide.
  const normal: Segment[] = [{ from: [px * (1 - ARROW), py * (1 - ARROW)], to: [px * (1 + ARROW), py * (1 + ARROW)] }]

  const profile = useMemo(
    (): XYSeries[] => [
      { name: 'f on the circle', type: 'line', x: PROFILE_X, y: PROFILE_Y, slot: 0 },
      { name: 'point', type: 'scatter', x: [angle.value], y: [f(px, py)], slot: 1 },
    ],
    [angle.value, px, py],
  )

  const handles: Handle[] = [
    {
      kind: 'point',
      at: [px, py],
      label: 'point on the constraint',
      onDrag: ([x, y]) => angle.set((Math.atan2(y, x) / DEG + 360) % 360),
    },
  ]
  const profileHandles: Handle[] = [{ kind: 'x', at: angle.value, label: 'θ', onDrag: angle.set }]

  return (
    <Interactive
      title="Gradients align at a constrained optimum"
      caption="Left: contours of f(x, y) = (x − 2)² + 2(y − 1)² (grey), the constraint x² + y² = 1, and the level set of f through the current point (dashed). The arrow is the direction of ∇f; the thin line through the point is the circle's normal, the direction of ∇g. Drag the point around the circle. Right: f along the circle against the angle θ; drag the vertical line to move the point. Where the arrow lies on the normal line, the level set touches the circle and f along the circle is flat."
      controls={<ParamSlider label="angle θ (degrees)" param={angle} format={(v) => `${formatNumber(v)}°`} />}
      readout={
        <>
          <Readout label="(x, y)" value={`(${formatNumber(px)}, ${formatNumber(py)})`} />
          <Readout label="f" value={formatNumber(f(px, py))} />
          <Readout label="angle between ∇f and ∇g" value={`${formatNumber(angleBetween)}°`} />
          <Readout label="df/dθ along the circle" value={formatNumber(tangential)} />
          <Readout label="λ = ∇f·∇g / ‖∇g‖²" value={formatNumber(lambda)} />
          <Readout label="minimum at θ" value={`${formatNumber(MIN_DEG)}°`} />
          <Readout label="maximum at θ" value={`${formatNumber(MAX_DEG)}°`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={series}
          vectors={vectors}
          segments={normal}
          handles={handles}
          xRange={X_RANGE}
          yRange={Y_RANGE}
          equalAspect
          xLabel="x"
          yLabel="y"
        />
        <XYChart
          height={340}
          series={profile}
          handles={profileHandles}
          xRange={[0, 360]}
          xLabel="angle θ (degrees)"
          yLabel="f(cos θ, sin θ)"
        />
      </div>
    </Interactive>
  )
}
