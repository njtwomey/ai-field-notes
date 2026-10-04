import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

const N_PER_CLASS = 40
const RANGE: [number, number] = [-4, 4]
const CIRCLE = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
// The classes differ along the direction at 30°; the orthogonal direction carries large, uninformative spread.
const SIGNAL = 30
const toRad = (deg: number) => (deg * Math.PI) / 180

const DATA = (() => {
  const r = stream(7)
  const u: [number, number] = [Math.cos(toRad(SIGNAL)), Math.sin(toRad(SIGNAL))]
  const w: [number, number] = [-u[1], u[0]]
  const pts: { x: number; y: number; c: number }[] = []
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < N_PER_CLASS; i++) {
      const s = (c === 0 ? -0.6 : 0.6) + 0.3 * normal(r)
      const t = 1.4 * normal(r)
      pts.push({ x: s * u[0] + t * w[0], y: s * u[1] + t * w[1], c })
    }
  }
  return pts
})()

/**
 * A Mahalanobis metric M = LᵀL is Euclidean distance after the linear map L. Here L shrinks one direction (at angle θ)
 * by a factor s. The right panel shows the mapped points; the readout is leave-one-out 1-nearest-neighbour accuracy.
 */
export function MetricStretch() {
  const state = useFigureState({
    angle: slider(0, 180, 120, { step: 5, label: 'direction θ (degrees)', format: (v) => v.toFixed(0) }),
    shrink: float(1, { min: 0.05, max: 1, step: 0.05, label: 'shrink factor s' }),
  })

  const r = useMemo(() => {
    const th = toRad(state.angle)
    const v: [number, number] = [Math.cos(th), Math.sin(th)]
    // L = I − (1 − s) v vᵀ: scales the component along v by s and leaves the orthogonal component unchanged.
    const k = 1 - state.shrink
    const map = (x: number, y: number): [number, number] => {
      const p = v[0] * x + v[1] * y
      return [x - k * p * v[0], y - k * p * v[1]]
    }
    const mapped = DATA.map((d) => map(d.x, d.y))
    let correct = 0
    for (let i = 0; i < mapped.length; i++) {
      let best = Infinity
      let label = -1
      for (let j = 0; j < mapped.length; j++) {
        if (j === i) continue
        const dd = (mapped[i][0] - mapped[j][0]) ** 2 + (mapped[i][1] - mapped[j][1]) ** 2
        if (dd < best) {
          best = dd
          label = DATA[j].c
        }
      }
      if (label === DATA[i].c) correct++
    }
    // The unit ball of d_M in the original space: points x with ‖Lx‖ = 1.
    const ball = CIRCLE.map((t): [number, number] => {
      const e: [number, number] = [Math.cos(t), Math.sin(t)]
      const [a, b] = map(e[0], e[1])
      const n = Math.hypot(a, b)
      return [e[0] / n, e[1] / n]
    })
    const group = DATA.map((d) => d.c)
    const original = [
      {
        name: 'points',
        x: DATA.map((d) => d.x),
        y: DATA.map((d) => d.y),
        group,
        groupNames: ['class A', 'class B'],
      },
      { name: 'unit ball of d_M', x: ball.map((p) => p[0]), y: ball.map((p) => p[1]), emphasis: true },
    ] as const
    const transformed = [
      {
        name: 'points after L',
        x: mapped.map((p) => p[0]),
        y: mapped.map((p) => p[1]),
        group,
        groupNames: ['class A', 'class B'],
      },
    ] as const
    return { accuracy: correct / mapped.length, original, transformed }
  }, [state.angle, state.shrink])

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  const xAxis2 = useAxis({ label: '(Lx)₁', range: RANGE })
  const yAxis2 = useAxis({ label: '(Lx)₂', range: RANGE, equal: xAxis2 })
  return (
    <Figure
      title="A Mahalanobis metric is a linear map followed by Euclidean distance"
      state={state}
      caption="Two classes differ along the direction at 30°, and have large, irrelevant spread along the direction at 120°. The metric shrinks the direction at angle θ by the factor s. Left: the original points and the set of points at distance 1 from the origin under the learned metric, an ellipse. Right: the same points after the map L, where the metric is ordinary Euclidean distance. Shrinking the 120° direction makes nearest neighbours share a class; shrinking the 30° direction destroys the signal."

      readouts={
        <>
          <Readout label="leave-one-out 1-NN accuracy" value={formatNumber(r.accuracy)} />
          <Readout label="eigenvalues of M" value={`1, ${formatNumber(state.shrink ** 2)}`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points {...r.original[0]} />
          <Curve {...r.original[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Points {...r.transformed[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
