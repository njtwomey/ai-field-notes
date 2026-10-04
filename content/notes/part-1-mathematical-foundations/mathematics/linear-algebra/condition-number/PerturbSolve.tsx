import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { svd2 } from 'aifn/numerics/linalg'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Vec = [number, number]
const RB = 3
const RX = 4
const clamp = (v: number, r: number) => Math.round(Math.min(Math.max(v, -r), r) * 20) / 20

/**
 * Solve A x = b for A with rows (1, 0) and (cos φ, sin φ): two lines meeting at angle φ. A circle of perturbations
 * around b maps through A⁻¹ to an ellipse around x, long when the lines are nearly parallel.
 */
export function PerturbSolve() {
  const state = useFigureState({
    angle: slider(3, 90, 20, { step: 1, label: 'angle φ between the equations (degrees)' }),
    radius: float(0.2, { min: 0.05, max: 0.5, step: 0.05, label: 'perturbation size r' }),
  })
  const [b, setB] = useState<Vec>([1, 1.2])

  const r = useMemo(() => {
    const phi = (state.angle * Math.PI) / 180
    const [c, s] = [Math.cos(phi), Math.sin(phi)]
    // A = [[1, 0], [c, s]], so A⁻¹ = [[1, 0], [−c/s, 1/s]].
    const solve = (v: Vec): Vec => [v[0], (v[1] - c * v[0]) / s]
    const x = solve(b)
    const ts = toFlat(linspace(0, 2 * Math.PI, 121))
    const circle = ts.map((t): Vec => [b[0] + state.radius * Math.cos(t), b[1] + state.radius * Math.sin(t)])
    const ellipse = circle.map(solve)
    const svd = svd2([
      [1, 0],
      [c, s],
    ])
    const kappa = svd.s[0] / svd.s[1]
    const far = 4 * RX
    const bSeries = [
      { name: 'b + δb, ‖δb‖ = r', x: circle.map((p) => p[0]), y: circle.map((p) => p[1]), slot: 1 },
      { name: 'b', x: [b[0]], y: [b[1]], slot: 1 },
    ] as const
    const xSeries = [
      { name: 'equation 1: x₁ = b₁', x: [b[0], b[0]], y: [-far, far], slot: 0, dashed: true },
      {
        name: 'equation 2: cos φ x₁ + sin φ x₂ = b₂',
        x: [-far, far],
        y: [(b[1] + c * far) / s, (b[1] - c * far) / s],
        slot: 2,
        dashed: true,
      },
      { name: 'solutions for b + δb', x: ellipse.map((p) => p[0]), y: ellipse.map((p) => p[1]), slot: 1 },
      { name: 'x', x: [x[0]], y: [x[1]], emphasis: true },
    ] as const
    return { x, kappa, sMin: svd.s[1], bSeries, xSeries, A: [c, s] }
  }, [state.angle, state.radius, b])

  const [c, s] = r.A
  // Dragging the solution sets b = A x, so the two charts stay consistent.

  const xAxis = useAxis({ label: 'b₁', range: [-RB, RB] })
  const yAxis = useAxis({ label: 'b₂', range: [-RB, RB], equal: xAxis })
  const xAxis2 = useAxis({ label: 'x₁', range: [-RX, RX] })
  const yAxis2 = useAxis({ label: 'x₂', range: [-RX, RX], equal: xAxis2 })
  return (
    <Figure
      title="Nearly parallel equations amplify errors"
      state={state}
      caption="Each equation of the 2 × 2 system A x = b is a dashed line on the right, and the solution x is where they cross. The circle on the left holds every right-hand side within distance r of b. On the right, the same circle solved through A⁻¹ becomes an ellipse. Shrink the angle φ between the lines: the ellipse stretches along the lines and the condition number grows. Drag b on the left or x on the right."

      readouts={
        <>
          <Readout label="κ₂(A) = σ₁/σ₂" value={formatNumber(r.kappa)} />
          <Readout label="largest ‖δx‖ = r/σ₂" value={formatNumber(state.radius / r.sMin)} />
          <Readout label="x" value={`(${formatNumber(r.x[0])}, ${formatNumber(r.x[1])})`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...r.bSeries[0]} />
          <Points {...r.bSeries[1]} />
          <Handle kind="point" at={b} label="b" onDrag={([p, q]) => setB([clamp(p, RB), clamp(q, RB)])} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...r.xSeries[0]} />
          <Curve {...r.xSeries[1]} />
          <Curve {...r.xSeries[2]} />
          <Points {...r.xSeries[3]} />
          <Handle kind="point" at={r.x} label="x" onDrag={([p, q]) => setB([clamp(p, RB), clamp(c * p + s * q, RB)])} />
        </Plot>
      </div>
    </Figure>
  )
}
