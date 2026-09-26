import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

type Vec = [number, number]
type Mat = [[number, number], [number, number]]
const R = 3.5
const f = formatNumber

const mul = (p: Mat, q: Mat): Mat => [
  [p[0][0] * q[0][0] + p[0][1] * q[1][0], p[0][0] * q[0][1] + p[0][1] * q[1][1]],
  [p[1][0] * q[0][0] + p[1][1] * q[1][0], p[1][0] * q[0][1] + p[1][1] * q[1][1]],
]
const apply = (m: Mat, v: Vec): Vec => [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
const show = (m: Mat) => `[[${f(m[0][0])}, ${f(m[0][1])}], [${f(m[1][0])}, ${f(m[1][1])}]]`

/** An asymmetric letter F, so that reflections and rotations are visible. */
const F: Vec[] = [
  [0, 0],
  [0, 2],
  [1.2, 2],
  [1.2, 1.6],
  [0.4, 1.6],
  [0.4, 1.2],
  [1, 1.2],
  [1, 0.8],
  [0.4, 0.8],
  [0.4, 0],
  [0, 0],
]

const trace = (m: Mat) => {
  const pts = F.map((p) => apply(m, p))
  return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) }
}

/** AB and BA applied to the same shape, for a rotation A and a stretch-and-shear B. */
export function Composition() {
  const angle = useParam(60, { min: -180, max: 180, step: 5 })
  const stretch = useParam(1.5, { min: 0.25, max: 2, step: 0.05 })
  const shear = useParam(0.5, { min: -1, max: 1, step: 0.05 })

  const r = useMemo(() => {
    const t = (angle.value * Math.PI) / 180
    const A: Mat = [
      [Math.cos(t), -Math.sin(t)],
      [Math.sin(t), Math.cos(t)],
    ]
    const B: Mat = [
      [stretch.value, shear.value],
      [0, 1],
    ]
    const AB = mul(A, B)
    const BA = mul(B, A)
    const series: XYSeries[] = [
      {
        name: 'shape',
        type: 'line',
        ...trace([
          [1, 0],
          [0, 1],
        ]),
        slot: 0,
        dashed: true,
      },
      { name: 'A B (B first, then A)', type: 'line', ...trace(AB), slot: 1 },
      { name: 'B A (A first, then B)', type: 'line', ...trace(BA), slot: 2 },
    ]
    const gap = Math.max(...AB.flat().map((v, i) => Math.abs(v - BA.flat()[i])))
    return { A, B, AB, BA, gap, series }
  }, [angle.value, stretch.value, shear.value])

  return (
    <Interactive
      title="Order matters"
      caption="A rotates by the chosen angle. B stretches the first coordinate and shears. The dashed letter F is the input. A B applies B first and then A; B A applies them the other way round. The two images differ unless the angle is a multiple of 180° or B is the identity (stretch 1, shear 0), the only cases here in which the two maps commute."
      controls={
        <>
          <ParamSlider label="rotation angle of A (degrees)" param={angle} />
          <ParamSlider label="stretch of B" param={stretch} />
          <ParamSlider label="shear of B" param={shear} />
        </>
      }
      readout={
        <>
          <Readout label="A B" value={show(r.AB)} />
          <Readout label="B A" value={show(r.BA)} />
          <Readout label="largest entry difference" value={f(r.gap)} />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <XYChart equalAspect xRange={[-R, R]} yRange={[-R, R]} xLabel="x₁" yLabel="x₂" series={r.series} />
      </div>
    </Interactive>
  )
}
