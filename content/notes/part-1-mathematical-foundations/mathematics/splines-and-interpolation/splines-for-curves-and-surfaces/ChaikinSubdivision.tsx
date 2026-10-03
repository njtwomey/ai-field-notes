import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import type { Vec2 } from '../_shared/splines'

const START: Vec2[] = [
  [1.5, 1],
  [2, 4.8],
  [5, 5.4],
  [5.4, 2.6],
  [8.6, 4.6],
  [8.4, 0.8],
]
const X: [number, number] = [0, 10]
const Y: [number, number] = [0, 6]

/** One round of Chaikin's corner cutting on a closed polygon: each edge gives its 1/4 and 3/4 points. */
function chaikin(P: Vec2[]): Vec2[] {
  return P.flatMap((p, i) => {
    const q = P[(i + 1) % P.length]
    return [
      [0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]],
      [0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]],
    ] as Vec2[]
  })
}

/** The closed uniform quadratic B-spline with control points P, sampled per span. */
function quadraticLimit(P: Vec2[], perSpan = 40): Vec2[] {
  const n = P.length
  const out: Vec2[] = []
  for (let i = 0; i < n; i++) {
    const [a, b, c] = [P[i], P[(i + 1) % n], P[(i + 2) % n]]
    for (let k = 0; k < perSpan; k++) {
      const s = k / perSpan
      const w = [0.5 * (1 - s) ** 2, 0.5 + s - s * s, 0.5 * s * s]
      out.push([w[0] * a[0] + w[1] * b[0] + w[2] * c[0], w[0] * a[1] + w[1] * b[1] + w[2] * c[1]])
    }
  }
  return out
}

function segmentDistance(q: Vec2, a: Vec2, b: Vec2): number {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const s = Math.min(1, Math.max(0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)))
  return Math.hypot(q[0] - a[0] - s * dx, q[1] - a[1] - s * dy)
}

const closeLoop = (P: Vec2[]): Vec2[] => [...P, P[0]]

/** Chaikin subdivision of a draggable closed polygon, converging to a uniform quadratic B-spline. */
export function ChaikinSubdivision() {
  const [points, setPoints] = useState<Vec2[]>(START)
  const rounds = useParam(2, { min: 0, max: 6, step: 1 })
  const [limit, setLimit] = useState(true)

  const r = useMemo(() => {
    let refined = points
    for (let k = 0; k < rounds.value; k++) refined = chaikin(refined)
    const curve = quadraticLimit(points)
    // Distance from each refined vertex to the limit curve, measured to a fine polyline through it.
    const fine = closeLoop(quadraticLimit(points, 120))
    const gap = Math.max(
      ...refined.map((q) => Math.min(...fine.slice(1).map((b, i) => segmentDistance(q, fine[i], b)))),
    )
    const series: XYSeries[] = [
      {
        name: 'control polygon',
        type: 'line',
        x: closeLoop(points).map((q) => q[0]),
        y: closeLoop(points).map((q) => q[1]),
        muted: true,
        dashed: true,
      },
    ]
    if (rounds.value > 0)
      series.push({
        name: `after ${rounds.value} round${rounds.value > 1 ? 's' : ''}`,
        type: 'line',
        x: closeLoop(refined).map((q) => q[0]),
        y: closeLoop(refined).map((q) => q[1]),
        slot: 0,
      })
    if (limit)
      series.push({
        name: 'quadratic B-spline (limit)',
        type: 'line',
        x: closeLoop(curve).map((q) => q[0]),
        y: closeLoop(curve).map((q) => q[1]),
        slot: 1,
        dashed: true,
      })
    return { series, count: refined.length, gap }
  }, [points, rounds.value, limit])

  const handles: Handle[] = points.map((q, i) => ({
    kind: 'point',
    at: q,
    onDrag: ([x, y]) =>
      setPoints((prev) =>
        prev.map((old, j) => (j === i ? [Math.min(Math.max(x, X[0]), X[1]), Math.min(Math.max(y, Y[0]), Y[1])] : old)),
      ),
  }))

  return (
    <Interactive
      title="Chaikin's corner cutting"
      caption="Drag the control points. Each round replaces every edge by its points at 1/4 and 3/4, cutting every corner. Step through the rounds: the polygon converges to the uniform quadratic B-spline of the original points (dashed), and the largest gap between polygon vertices and the limit curve falls by about 4 each round."
      controls={
        <>
          <ParamSlider label="Rounds of subdivision" param={rounds} withArrows format={(v) => String(v)} />
          <ParamSwitch label="limit curve" checked={limit} onChange={setLimit} />
        </>
      }
      readout={
        <>
          <Readout label="vertices" value={r.count} />
          <Readout label="max vertex distance to limit" value={formatNumber(r.gap)} />
        </>
      }
    >
      <XYChart series={r.series} xRange={X} yRange={Y} equalAspect handles={handles} ariaLabel="Chaikin subdivision" />
    </Interactive>
  )
}
