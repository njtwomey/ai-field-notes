import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'

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
  const [a, setA] = useState(0.6)
  const [point, setPoint] = useState<Vec>([0.8, 0.4])

  const r = useMemo(() => {
    const lines: XYSeries[] = []
    const ts = linspace(-R, R, 60)
    for (const c of linspace(-R, R, 13)) {
      const h = ts.map((t) => warp(a, [t, c]))
      const v = ts.map((t) => warp(a, [c, t]))
      lines.push({ name: 'warped grid', type: 'line', x: h.map((p) => p[0]), y: h.map((p) => p[1]), slot: 0 })
      lines.push({ name: 'warped grid', type: 'line', x: v.map((p) => p[0]), y: v.map((p) => p[1]), slot: 0 })
    }
    // The square [x, x+h] × [y, y+h], traced around its edge, and its exact image.
    const edge = linspace(0, 1, 20)
    const [x0, y0] = point
    const square: Vec[] = [
      ...edge.map((t): Vec => [x0 + H * t, y0]),
      ...edge.map((t): Vec => [x0 + H, y0 + H * t]),
      ...edge.map((t): Vec => [x0 + H * (1 - t), y0 + H]),
      ...edge.map((t): Vec => [x0, y0 + H * (1 - t)]),
    ]
    const image = square.map((p) => warp(a, p))
    const J = jacobian(a, point)
    const f0 = warp(a, point)
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
    const series: XYSeries[] = [
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
  }, [a, point])

  // The handle sits at f(point), in output coordinates; dragging it places the square's corner at the input that maps
  // under the pointer.
  const handles: Handle[] = [
    {
      kind: 'point',
      at: warp(a, point),
      label: 'square',
      onDrag: (p) => {
        const [x, y] = unwarp(a, p)
        const clamp = (v: number) => Math.min(Math.max(v, -R), R)
        setPoint([clamp(x), clamp(y)])
      },
    },
  ]

  return (
    <Interactive
      title="The Jacobian is the local linear map"
      caption="The blue grid is the plane after the warp f(x, y) = (x + a sin y, y + a sin x). Drag the round handle to move a small square: its exact image is the orange curve, and the Jacobian's parallelogram (dashed) nearly matches it. det J is the factor by which area is scaled there."
      controls={<ParamSlider label="warp strength a" value={a} onChange={setA} min={0} max={0.95} step={0.01} />}
      readout={
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
        <XYChart
          equalAspect
          xRange={[-R - 1, R + 1]}
          yRange={[-R - 1, R + 1]}
          series={r.series}
          xLabel="f₁"
          yLabel="f₂"
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
