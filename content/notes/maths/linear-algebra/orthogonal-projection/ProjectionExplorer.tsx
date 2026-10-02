import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'

type Vec = [number, number]
const R = 4

/** Projection of a point onto a line through the origin. Drag the point x, or the tip of u to turn the line. */
export function ProjectionExplorer() {
  const direction = useParam(25, { min: 0, max: 180, step: 1 })
  const angle = direction.value
  const [x, setX] = useState<Vec>([1, 3])
  const clamp = (v: number) => Math.round(Math.min(Math.max(v, -R), R) * 10) / 10

  const r = useMemo(() => {
    const t = (angle * Math.PI) / 180
    const u: Vec = [Math.cos(t), Math.sin(t)]
    const c = u[0] * x[0] + u[1] * x[1]
    const p: Vec = [c * u[0], c * u[1]]
    const res: Vec = [x[0] - p[0], x[1] - p[1]]
    const series: XYSeries[] = [
      { name: 'line spanned by u', type: 'line', x: [-R * u[0], R * u[0]], y: [-R * u[1], R * u[1]], slot: 0 },
      { name: 'x', type: 'scatter', x: [x[0]], y: [x[1]], slot: 1 },
      { name: 'projection p', type: 'scatter', x: [p[0]], y: [p[1]], emphasis: true },
      { name: 'residual x − p', type: 'line', x: [p[0], x[0]], y: [p[1], x[1]], slot: 1, dashed: true },
    ]
    return { u, c, p, res, series }
  }, [angle, x])

  const handles: Handle[] = [
    { kind: 'point', at: x, label: 'x', onDrag: ([a, b]) => setX([clamp(a), clamp(b)]) },
    {
      kind: 'point',
      at: r.u,
      label: 'u',
      // u and −u span the same line, so the angle is taken modulo 180°.
      onDrag: ([a, b]) => direction.set(((((Math.atan2(b, a) * 180) / Math.PI) % 180) + 180) % 180),
    },
  ]

  return (
    <Interactive
      title="The closest point on a line"
      caption="Drag the point x to move it. Drag the tip of the arrow u, or use the slider, to turn the line. The projection p is the point on the line closest to x. The residual x − p always meets the line at a right angle, so its dot product with u is zero."
      controls={<ParamSlider label="direction of u (degrees)" param={direction} />}
      readout={
        <>
          <Readout label="coefficient uᵀx" value={formatNumber(r.c)} />
          <Readout label="p" value={`(${formatNumber(r.p[0])}, ${formatNumber(r.p[1])})`} />
          <Readout label="‖x − p‖" value={formatNumber(Math.hypot(...r.res))} />
          <Readout
            label="uᵀ(x − p)"
            value={formatNumber(
              Math.abs(r.u[0] * r.res[0] + r.u[1] * r.res[1]) < 1e-12 ? 0 : r.u[0] * r.res[0] + r.u[1] * r.res[1],
            )}
          />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          equalAspect
          xRange={[-R, R]}
          yRange={[-R, R]}
          xLabel="x₁"
          yLabel="x₂"
          series={r.series}
          vectors={[{ from: [0, 0], to: r.u }]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
