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
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

type Vec = [number, number]
const R = 2.2
const clamp = (v: number) => Math.round(Math.min(Math.max(v, -R), R) * 20) / 20

const pNorm = (v: Vec, p: number) =>
  p === Infinity ? Math.max(Math.abs(v[0]), Math.abs(v[1])) : (Math.abs(v[0]) ** p + Math.abs(v[1]) ** p) ** (1 / p)

/** The sphere of radius r in the ℓp norm, traced by scaling each direction until its norm is r. */
function sphere(p: number, r: number): { x: number[]; y: number[] } {
  const ts = linspace(0, 2 * Math.PI, 721)
  const pts = ts.map((t): Vec => {
    const d: Vec = [Math.cos(t), Math.sin(t)]
    const s = r / pNorm(d, p)
    return [s * d[0], s * d[1]]
  })
  return { x: pts.map((q) => q[0]), y: pts.map((q) => q[1]) }
}

const toSegments = ({ x, y }: { x: number[]; y: number[] }): Segment[] =>
  x.slice(1).map((_, i) => ({ from: [x[i], y[i]], to: [x[i + 1], y[i + 1]] }))

/** Unit balls of the ℓp norms, and the ℓp sphere through a draggable point. */
export function UnitBalls() {
  const p = useParam(1.5, { min: 0.5, max: 8, step: 0.1 })
  const [infinite, setInfinite] = useState(false)
  const [x, setX] = useState<Vec>([1.2, 0.6])
  const pv = infinite ? Infinity : p.value

  // The ℓ1, ℓ2 and ℓ∞ unit spheres as a faint reference, drawn once.
  const references = useMemo(() => [1, 2, Infinity].flatMap((q) => toSegments(sphere(q, 1))), [])

  const r = useMemo(() => {
    const norm = pNorm(x, pv)
    const series: XYSeries[] = [
      { name: `unit ball, p = ${infinite ? '∞' : formatNumber(pv)}`, type: 'line', ...sphere(pv, 1), slot: 0 },
      { name: 'ℓp sphere through x', type: 'line', ...sphere(pv, norm), slot: 1, dashed: true },
      { name: 'x', type: 'scatter', x: [x[0]], y: [x[1]], slot: 1 },
    ]
    return { norm, series }
  }, [x, pv, infinite])

  const handles: Handle[] = [{ kind: 'point', at: x, label: 'x', onDrag: ([a, b]) => setX([clamp(a), clamp(b)]) }]

  return (
    <Interactive
      title="The shape of the unit ball"
      caption="The solid curve is the unit sphere of the ℓp norm, the points with ‖x‖ₚ = 1; the faint curves are the ℓ1 diamond, the ℓ2 circle and the ℓ∞ square. Raise p and the ball swells from the diamond towards the square. Below p = 1 the ball caves in and stops being convex: the formula is no longer a norm. Drag x; the dashed curve is the ℓp sphere through it, and its size is ‖x‖ₚ."
      controls={
        <>
          <ParamSlider label="p" param={p} />
          <ParamSwitch label="p = ∞" checked={infinite} onChange={setInfinite} />
        </>
      }
      readout={
        <>
          <Readout label="‖x‖ₚ" value={formatNumber(r.norm)} />
          <Readout label="‖x‖₁" value={formatNumber(pNorm(x, 1))} />
          <Readout label="‖x‖₂" value={formatNumber(pNorm(x, 2))} />
          <Readout label="‖x‖∞" value={formatNumber(pNorm(x, Infinity))} />
          <Readout label="convex ball" value={pv >= 1 ? 'yes' : 'no'} />
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
          segments={references}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
