import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal as drawNormal, stream } from 'aifn-compute/foundation/random'
import { directionField, rk4Step } from '../_shared/ode'

type Field = 'rotation' | 'saddle' | 'sink' | 'pendulum' | 'damped'
const FIELDS: Record<Field, { label: string; f: (x: number, y: number) => [number, number]; div: number }> = {
  rotation: { label: 'rotation', f: (x, y) => [-y, x], div: 0 },
  saddle: { label: 'saddle', f: (x, y) => [0.5 * x, -0.5 * y], div: 0 },
  sink: { label: 'sink', f: (x, y) => [-0.4 * x, -0.4 * y], div: -0.8 },
  pendulum: { label: 'pendulum', f: (x, y) => [y, -Math.sin(x)], div: 0 },
  damped: { label: 'damped pendulum', f: (x, y) => [y, -Math.sin(x) - 0.5 * y], div: -0.5 },
}
const R = 3.5
const RANGE: [number, number] = [-R, R]
const SIDE = 0.8
const H = 0.02

/** The boundary of a square patch (walked anticlockwise) and a grid of points inside it. */
function patch([cx, cy]: [number, number]) {
  const boundary: [number, number][] = []
  const n = 20
  const s = SIDE / 2
  const corners: [number, number][] = [
    [-s, -s],
    [s, -s],
    [s, s],
    [-s, s],
  ]
  for (let k = 0; k < 4; k++) {
    const [ax, ay] = corners[k]
    const [bx, by] = corners[(k + 1) % 4]
    for (let i = 0; i < n; i++) boundary.push([cx + ax + ((bx - ax) * i) / n, cy + ay + ((by - ay) * i) / n])
  }
  const inside: [number, number][] = []
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) inside.push([cx - s + (SIDE * (i + 0.5)) / 6, cy - s + (SIDE * (j + 0.5)) / 6])
  return { boundary, inside }
}

const CLOUD: [number, number][] = (() => {
  const rs = stream(7)
  const normal = () => drawNormal(rs)
  return Array.from({ length: 250 }, () => [1.1 * normal(), 1.1 * normal()] as [number, number])
})()

function flow(f: (x: number, y: number) => [number, number], pts: [number, number][], t: number) {
  const steps = Math.round(t / H)
  const g = (_t: number, p: number[]) => f(p[0], p[1])
  return pts.map((p) => {
    let q: number[] = p
    for (let k = 0; k < steps; k++) q = rk4Step(g, 0, q, H)
    return q as [number, number]
  })
}

/** Shoelace formula for the area of a closed polygon. */
function area(poly: [number, number][]) {
  let a = 0
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i]
    const [x1, y1] = poly[(i + 1) % poly.length]
    a += x0 * y1 - x1 * y0
  }
  return Math.abs(a) / 2
}

export function FlowCloud() {
  const state = useFigureState({
    field: choice<Field>(
      (Object.keys(FIELDS) as Field[]).map((k) => ({ value: k, label: FIELDS[k].label })),
      'pendulum',
      { label: 'vector field' },
    ),
    time: slider(0, 6, 2, { step: 0.1, label: 'time t' }),
  })
  const [centre, setCentre] = useState<[number, number]>([1, 0.5])
  const { f, div } = FIELDS[state.field]

  const arrows = useMemo(() => directionField(f, RANGE, RANGE, 13, 13), [f])
  const moved = useMemo(() => {
    const p = patch(centre)
    const boundary = flow(f, p.boundary, state.time)
    const path: [number, number][] = [centre]
    const g = (_t: number, q: number[]) => f(q[0], q[1])
    let q: number[] = centre
    for (let k = 0; k < Math.round(state.time / H); k++) {
      q = rk4Step(g, 0, q, H)
      if (k % 5 === 4) path.push(q as [number, number])
    }
    return {
      boundary,
      inside: flow(f, p.inside, state.time),
      cloud: flow(f, CLOUD, state.time),
      path,
      area: area(boundary),
    }
  }, [f, centre, state.time])

  const series = useMemo(
    () =>
      [
        { name: 'cloud', x: moved.cloud.map((p) => p[0]), y: moved.cloud.map((p) => p[1]), muted: true },
        {
          name: 'patch boundary',
          x: [...moved.boundary, moved.boundary[0]].map((p) => p[0]),
          y: [...moved.boundary, moved.boundary[0]].map((p) => p[1]),
          slot: 0,
        },
        {
          name: 'patch points',
          x: moved.inside.map((p) => p[0]),
          y: moved.inside.map((p) => p[1]),
          slot: 0,
        },
        {
          name: 'path of the centre',
          x: moved.path.map((p) => p[0]),
          y: moved.path.map((p) => p[1]),
          slot: 1,
        },
        { name: 'start', x: [centre[0]], y: [centre[1]], emphasis: true },
      ] as const,
    [moved, centre],
  )
  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: centre,
        label: 'patch',
        onDrag: ([x, y]) => setCentre([Math.max(-3, Math.min(3, x)), Math.max(-3, Math.min(3, y))]),
      },
    ],
    [centre],
  )
  const predicted = SIDE * SIDE * Math.exp(div * state.time)

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="A flow moves every point at once"
      state={state}
      caption="A square patch of points and a Gaussian cloud (grey) are carried by the flow φₜ of the chosen vector field. Step the time to watch the patch stretch, shear and turn. Drag the black point to move where the patch starts. The patch area always equals its starting area times exp(t · div f): it is preserved where the divergence is 0 and shrinks where it is negative."

      readouts={
        <>
          <Readout label="div f" value={formatNumber(div)} />
          <Readout label="patch area" value={formatNumber(moved.area)} />
          <Readout label="0.64 · exp(t · div f)" value={formatNumber(predicted)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          <Points {...series[0]} />
          <Curve {...series[1]} />
          <Points {...series[2]} />
          <Curve {...series[3]} />
          <Points {...series[4]} />
          <Segments segments={arrows} />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
