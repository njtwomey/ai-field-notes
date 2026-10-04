import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Vec = [number, number]
const R = 3.2
const H = 0.4

/** The warp f(x, y) = (x + a sin y, y + a sin x) and its Jacobian. */
const warp = (a: number, [x, y]: Vec): Vec => [x + a * Math.sin(y), y + a * Math.sin(x)]
/**
 * Inverse of the warp by fixed-point iteration: x = u − a sin y, y = v − a sin x. For a < 1 the map is a contraction,
 * so this converges to the unique preimage.
 */
function unwarp(a: number, [u, v]: Vec): Vec {
  let x = u
  let y = v
  for (let i = 0; i < 100; i++) {
    const nx = u - a * Math.sin(y)
    const ny = v - a * Math.sin(x)
    if (Math.abs(nx - x) + Math.abs(ny - y) < 1e-10) return [nx, ny]
    x = nx
    y = ny
  }
  return [x, y]
}

const jacobian = (a: number, [x, y]: Vec): [[number, number], [number, number]] => [
  [1, a * Math.cos(y)],
  [a * Math.cos(x), 1],
]

/**
 * A grid under a smooth warp. Near the chosen point, the image of a small square (orange) is close to the
 * parallelogram spanned by the Jacobian's columns (dashed): the Jacobian is the local linear approximation.
 */
export function Warp() {
  const state = useFigureState({
    a: float(0.6, { min: 0, max: 0.95, step: 0.01, label: 'warp strength a' }),
  })
  const [point, setPoint] = useState<Vec>([0.8, 0.4])

  const r = useMemo(() => {
    const lines: SeriesSpec[] = []
    const ts = toFlat(linspace(-R, R, 60))
    for (const c of toFlat(linspace(-R, R, 13))) {
      const h = ts.map((t) => warp(state.a, [t, c]))
      const v = ts.map((t) => warp(state.a, [c, t]))
      lines.push({ name: 'warped grid', type: 'line', x: h.map((p) => p[0]), y: h.map((p) => p[1]), slot: 0 })
      lines.push({ name: 'warped grid', type: 'line', x: v.map((p) => p[0]), y: v.map((p) => p[1]), slot: 0 })
    }
    // The square [x, x+h] × [y, y+h], traced around its edge, and its exact image.
    const edge = toFlat(linspace(0, 1, 20))
    const [x0, y0] = point
    const square: Vec[] = [
      ...edge.map((t): Vec => [x0 + H * t, y0]),
      ...edge.map((t): Vec => [x0 + H, y0 + H * t]),
      ...edge.map((t): Vec => [x0 + H * (1 - t), y0 + H]),
      ...edge.map((t): Vec => [x0, y0 + H * (1 - t)]),
    ]
    const image = square.map((p) => warp(state.a, p))
    const J = jacobian(state.a, point)
    const f0 = warp(state.a, point)
    const corners: Vec[] = [
      [0, 0],
      [H, 0],
      [H, H],
      [0, H],
      [0, 0],
    ]
    const linear = corners.map(([dx, dy]): Vec => [
      f0[0] + J[0][0] * dx + J[0][1] * dy,
      f0[1] + J[1][0] * dx + J[1][1] * dy,
    ])
    const series: SeriesSpec[] = [
      ...lines,
      { name: 'image of the square', type: 'line', x: image.map((p) => p[0]), y: image.map((p) => p[1]), slot: 1 },
      {
        name: 'Jacobian approximation',
        type: 'line',
        x: linear.map((p) => p[0]),
        y: linear.map((p) => p[1]),
        slot: 3,
        dashed: true,
      },
    ]
    return { series, J, det: J[0][0] * J[1][1] - J[0][1] * J[1][0] }
  }, [state.a, point])

  // The handle sits at f(point), in output coordinates; dragging it places the square's corner at the input that maps
  // under the pointer.

  const xAxis = useAxis({ label: 'f₁', range: [-R - 1, R + 1] })
  const yAxis = useAxis({ label: 'f₂', range: [-R - 1, R + 1], equal: xAxis })
  return (
    <Figure
      title="The Jacobian is the local linear map"
      state={state}
      caption="The blue grid is the plane after the warp f(x, y) = (x + a sin y, y + a sin x). Drag the round handle to move a small square: its exact image is the orange curve, and the Jacobian's parallelogram (dashed) nearly matches it. det J is the factor by which area is scaled there."

      readouts={
        <>
          <Readout
            label="J"
            value={`[[${formatNumber(r.J[0][0])}, ${formatNumber(r.J[0][1])}], [${formatNumber(r.J[1][0])}, ${formatNumber(r.J[1][1])}]]`}
          />
          <Readout label="det J (area scale)" value={formatNumber(r.det)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(r.series)}
          <Handle
            kind="point"
            at={warp(state.a, point)}
            label="square"
            onDrag={(p) => {
              const [x, y] = unwarp(state.a, p)
              const clamp = (v: number) => Math.min(Math.max(v, -R), R)
              setPoint([clamp(x), clamp(y)])
            }}
          />
        </Plot>
      </div>
    </Figure>
  )
}
