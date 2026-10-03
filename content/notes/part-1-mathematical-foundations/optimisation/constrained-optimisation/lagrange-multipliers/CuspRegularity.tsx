import { Readout, formatNumber } from 'aifn-render'
import { ConstrainedExplorer, type ConstrainedProblem } from './ConstrainedExplorer'

/** Minimise x on the cusp x³ = y², parametrised as (t², t³). The minimum is the origin, where ∇g = 0. */
const PROBLEM: ConstrainedProblem = {
  f: (x) => x,
  gradF: () => [1, 0],
  gradG: (x, y) => [3 * x * x, -2 * y],
  curve: (t) => [t * t, t * t * t],
  tRange: [-1.2, 1.2],
  xRange: [-0.5, 1.5],
  yRange: [-2, 2],
  levels: [-0.4, 0.001, 0.4, 0.8, 1.2],
  goal: 'min',
}

export function CuspRegularity() {
  return (
    <ConstrainedExplorer
      problem={PROBLEM}
      title="Regularity fails at a cusp"
      caption="Left: contours of f(x, y) = x (grey, vertical lines), the cusp x³ = y², and the arrow of ∇f = (1, 0) against the normal ∇g = (3x², −2y). Drag the point along the curve, or set t, where the point is (t², t³). The minimum is the cusp at the origin. There ∇g = 0, the normal is undefined, and the least-squares λ grows without bound as the point approaches it."
      initialT={0.6}
      tStep={0.01}
      xLabel="x"
      yLabel="y"
      tLabel="parameter t"
      tSymbol="t"
      fLabel="f"
      readout={({ gradG }) => <Readout label="‖∇g‖" value={formatNumber(Math.hypot(gradG[0], gradG[1]))} />}
    />
  )
}
