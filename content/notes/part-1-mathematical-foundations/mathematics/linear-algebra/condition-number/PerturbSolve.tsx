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
import { linspace } from '@/lib/math'
import { svd2 } from '@/lib/math/mat2'

type Vec = [number, number]
const RB = 3
const RX = 4
const clamp = (v: number, r: number) => Math.round(Math.min(Math.max(v, -r), r) * 20) / 20

/**
 * Solve A x = b for A with rows (1, 0) and (cos φ, sin φ): two lines meeting at angle φ. A circle of perturbations
 * around b maps through A⁻¹ to an ellipse around x, long when the lines are nearly parallel.
 */
export function PerturbSolve() {
  const angle = useParam(20, { min: 3, max: 90, step: 1 })
  const radius = useParam(0.2, { min: 0.05, max: 0.5, step: 0.05 })
  const [b, setB] = useState<Vec>([1, 1.2])

  const r = useMemo(() => {
    const phi = (angle.value * Math.PI) / 180
    const [c, s] = [Math.cos(phi), Math.sin(phi)]
    // A = [[1, 0], [c, s]], so A⁻¹ = [[1, 0], [−c/s, 1/s]].
    const solve = (v: Vec): Vec => [v[0], (v[1] - c * v[0]) / s]
    const x = solve(b)
    const ts = linspace(0, 2 * Math.PI, 121)
    const circle = ts.map((t): Vec => [b[0] + radius.value * Math.cos(t), b[1] + radius.value * Math.sin(t)])
    const ellipse = circle.map(solve)
    const svd = svd2(1, 0, c, s)
    const kappa = svd.s[0] / svd.s[1]
    const far = 4 * RX
    const bSeries: XYSeries[] = [
      { name: 'b + δb, ‖δb‖ = r', type: 'line', x: circle.map((p) => p[0]), y: circle.map((p) => p[1]), slot: 1 },
      { name: 'b', type: 'scatter', x: [b[0]], y: [b[1]], slot: 1 },
    ]
    const xSeries: XYSeries[] = [
      { name: 'equation 1: x₁ = b₁', type: 'line', x: [b[0], b[0]], y: [-far, far], slot: 0, dashed: true },
      {
        name: 'equation 2: cos φ x₁ + sin φ x₂ = b₂',
        type: 'line',
        x: [-far, far],
        y: [(b[1] + c * far) / s, (b[1] - c * far) / s],
        slot: 2,
        dashed: true,
      },
      { name: 'solutions for b + δb', type: 'line', x: ellipse.map((p) => p[0]), y: ellipse.map((p) => p[1]), slot: 1 },
      { name: 'x', type: 'scatter', x: [x[0]], y: [x[1]], emphasis: true },
    ]
    return { x, kappa, sMin: svd.s[1], bSeries, xSeries, A: [c, s] }
  }, [angle.value, radius.value, b])

  const bHandles: Handle[] = [
    { kind: 'point', at: b, label: 'b', onDrag: ([p, q]) => setB([clamp(p, RB), clamp(q, RB)]) },
  ]
  const [c, s] = r.A
  // Dragging the solution sets b = A x, so the two charts stay consistent.
  const xHandles: Handle[] = [
    {
      kind: 'point',
      at: r.x,
      label: 'x',
      onDrag: ([p, q]) => setB([clamp(p, RB), clamp(c * p + s * q, RB)]),
    },
  ]

  return (
    <Interactive
      title="Nearly parallel equations amplify errors"
      caption="Each equation of the 2 × 2 system A x = b is a dashed line on the right, and the solution x is where they cross. The circle on the left holds every right-hand side within distance r of b. On the right, the same circle solved through A⁻¹ becomes an ellipse. Shrink the angle φ between the lines: the ellipse stretches along the lines and the condition number grows. Drag b on the left or x on the right."
      controls={
        <>
          <ParamSlider label="angle φ between the equations (degrees)" param={angle} />
          <ParamSlider label="perturbation size r" param={radius} />
        </>
      }
      readout={
        <>
          <Readout label="κ₂(A) = σ₁/σ₂" value={formatNumber(r.kappa)} />
          <Readout label="largest ‖δx‖ = r/σ₂" value={formatNumber(radius.value / r.sMin)} />
          <Readout label="x" value={`(${formatNumber(r.x[0])}, ${formatNumber(r.x[1])})`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          equalAspect
          xRange={[-RB, RB]}
          yRange={[-RB, RB]}
          xLabel="b₁"
          yLabel="b₂"
          series={r.bSeries}
          handles={bHandles}
        />
        <XYChart
          equalAspect
          xRange={[-RX, RX]}
          yRange={[-RX, RX]}
          xLabel="x₁"
          yLabel="x₂"
          series={r.xSeries}
          handles={xHandles}
        />
      </div>
    </Interactive>
  )
}
