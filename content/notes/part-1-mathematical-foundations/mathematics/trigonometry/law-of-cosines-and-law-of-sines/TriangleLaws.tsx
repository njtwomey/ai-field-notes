import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Shapes,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { arc, circle, polygon, toDegrees, type Vec } from '../_shared/geometry'

const R = 4
const coord = (initial: number) => slider(-R, R, initial, { step: 0.05, onChart: true })
const dist = (p: Vec, q: Vec) => Math.hypot(p[0] - q[0], p[1] - q[1])

/** The interior angle at p between the rays to q and r, by atan2 of the cross and dot products. */
function angleAt(p: Vec, q: Vec, r: Vec): number {
  const u = [q[0] - p[0], q[1] - p[1]]
  const v = [r[0] - p[0], r[1] - p[1]]
  return Math.abs(Math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))
}

/** A small arc at p marking the interior angle between the rays to q and r. */
function angleArc(p: Vec, q: Vec, r: Vec, radius: number) {
  const a = Math.atan2(q[1] - p[1], q[0] - p[0])
  let b = Math.atan2(r[1] - p[1], r[0] - p[0])
  // Turn the short way round.
  if (b - a > Math.PI) b -= 2 * Math.PI
  if (a - b > Math.PI) b += 2 * Math.PI
  return arc(radius, a, b, 32, p)
}

/** The centre of the circle through three points, or null when they are collinear. */
function circumcentre(A: Vec, B: Vec, C: Vec): Vec | null {
  const d = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
  if (Math.abs(d) < 1e-9) return null
  const a2 = A[0] ** 2 + A[1] ** 2
  const b2 = B[0] ** 2 + B[1] ** 2
  const c2 = C[0] ** 2 + C[1] ** 2
  return [
    (a2 * (B[1] - C[1]) + b2 * (C[1] - A[1]) + c2 * (A[1] - B[1])) / d,
    (a2 * (C[0] - B[0]) + b2 * (A[0] - C[0]) + c2 * (B[0] - A[0])) / d,
  ]
}

/**
 * A triangle with draggable vertices A, B and C, and the sides a, b, c opposite them. The readouts check the law of
 * cosines at C and the law of sines, whose common ratio is the diameter of the circumscribed circle (dashed).
 */
export function TriangleLaws() {
  const state = useFigureState({
    ax: coord(-2.5),
    ay: coord(-1.5),
    bx: coord(2.5),
    by: coord(-1.5),
    cx: coord(-0.5),
    cy: coord(2),
  })
  const A: Vec = useMemo(() => [state.ax, state.ay], [state.ax, state.ay])
  const B: Vec = useMemo(() => [state.bx, state.by], [state.bx, state.by])
  const C: Vec = useMemo(() => [state.cx, state.cy], [state.cx, state.cy])

  const t = useMemo(() => {
    const a = dist(B, C)
    const b = dist(C, A)
    const c = dist(A, B)
    const angA = angleAt(A, B, C)
    const angB = angleAt(B, C, A)
    const angC = angleAt(C, A, B)
    const centre = circumcentre(A, B, C)
    const radius = centre ? dist(centre, A) : NaN
    const ring = centre && radius < 50 ? circle(radius, centre) : { x: [], y: [] }
    const outline = { x: [A[0], B[0], C[0], A[0]], y: [A[1], B[1], C[1], A[1]] }
    const fill = [polygon([A, B, C], 0, 0.12)]
    const arcs = [angleArc(A, B, C, 0.45), angleArc(B, C, A, 0.45), angleArc(C, A, B, 0.45)]
    const mid = (p: Vec, q: Vec): Vec => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]
    const labels = {
      x: [mid(B, C)[0], mid(C, A)[0], mid(A, B)[0]],
      y: [mid(B, C)[1], mid(C, A)[1], mid(A, B)[1]],
      text: [`a = ${formatNumber(a)}`, `b = ${formatNumber(b)}`, `c = ${formatNumber(c)}`],
    }
    return { a, b, c, angA, angB, angC, radius, ring, outline, fill, arcs, labels }
  }, [A, B, C])

  const degenerate = !Number.isFinite(t.radius)
  const ratio = (side: number, ang: number) => (Math.sin(ang) > 1e-9 ? formatNumber(side / Math.sin(ang)) : '–')

  const x = useAxis({ label: 'x', range: [-R, R] })
  const y = useAxis({ label: 'y', range: [-R, R], equal: x })
  return (
    <Figure
      title="The laws of cosines and sines"
      state={state}
      caption="Drag the vertices A, B and C. Side a is opposite A, b opposite B and c opposite C. The first group of readouts checks the law of cosines at C: c² equals a² + b² − 2ab cos C for every triangle, and the correction −2ab cos C is zero when C is a right angle, negative when C is acute and positive when C is obtuse. The second group checks the law of sines: the three ratios side / sin(opposite angle) agree, and their common value is 2R, the diameter of the dashed circumscribed circle. Push C towards the line AB and the circle grows without bound as the triangle flattens."
      readouts={{
        triangle: (
          <>
            <Readout label="A" value={`${formatNumber(toDegrees(t.angA))}°`} />
            <Readout label="B" value={`${formatNumber(toDegrees(t.angB))}°`} />
            <Readout label="C" value={`${formatNumber(toDegrees(t.angC))}°`} />
            <Readout label="A + B + C" value={`${formatNumber(toDegrees(t.angA + t.angB + t.angC))}°`} />
          </>
        ),
        'law of cosines at C': (
          <>
            <Readout label="c²" value={formatNumber(t.c * t.c)} />
            <Readout label="a² + b²" value={formatNumber(t.a * t.a + t.b * t.b)} />
            <Readout label="− 2ab cos C" value={formatNumber(-2 * t.a * t.b * Math.cos(t.angC))} />
          </>
        ),
        'law of sines': (
          <>
            <Readout label="a / sin A" value={ratio(t.a, t.angA)} />
            <Readout label="b / sin B" value={ratio(t.b, t.angB)} />
            <Readout label="c / sin C" value={ratio(t.c, t.angC)} />
            <Readout label="2R" value={degenerate ? 'unbounded' : formatNumber(2 * t.radius)} />
          </>
        ),
      }}
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight legend={false}>
          <Shapes name="triangle" shapes={t.fill} />
          <Curve name="circumscribed circle" x={t.ring.x} y={t.ring.y} muted dashed silent />
          <Curve name="triangle" x={t.outline.x} y={t.outline.y} emphasis silent />
          <Curve name="angle A" x={t.arcs[0].x} y={t.arcs[0].y} slot={0} width={2} silent />
          <Curve name="angle B" x={t.arcs[1].x} y={t.arcs[1].y} slot={1} width={2} silent />
          <Curve name="angle C" x={t.arcs[2].x} y={t.arcs[2].y} slot={2} width={2} silent />
          <Points name="sides" x={t.labels.x} y={t.labels.y} labels={t.labels.text} muted size={1} />
          <Handle {...state.handle(['ax', 'ay'], { label: 'A' })} />
          <Handle {...state.handle(['bx', 'by'], { label: 'B' })} />
          <Handle {...state.handle(['cx', 'cy'], { label: 'C' })} />
        </Plot>
      </div>
    </Figure>
  )
}
