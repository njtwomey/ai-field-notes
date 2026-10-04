import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, tField, type ConstrainedProblem } from './ConstrainedExplorer'

const DEG = Math.PI / 180

/** Minimise x² + a·y² on the unit circle. The stationary points are (±1, 0) with λ = 1 and (0, ±1) with λ = a. */
function problem(a: number): ConstrainedProblem {
  return {
    f: (x, y) => x * x + a * y * y,
    gradF: (x, y) => [2 * x, 2 * a * y],
    gradG: (x, y) => [2 * x, 2 * y],
    curve: (deg) => [Math.cos(deg * DEG), Math.sin(deg * DEG)],
    tRange: [0, 360],
    closed: true,
    xRange: [-2, 2],
    yRange: [-2, 2],
    goal: 'min',
  }
}

const verdict = (q: number) => (q > 1e-9 ? 'minimum' : q < -1e-9 ? 'maximum' : 'test inconclusive')

export function SecondOrderTest() {
  const state = useFigureState({
    t: tField([0, 360], 40, 0.5, 'angle θ (degrees)', (v) => `${formatNumber(v)}°`),
    a: slider(-1, 3, 0.4, { step: 0.05, label: 'coefficient a' }),
  })
  const p = useMemo(() => problem(state.a), [state.a])
  // Curvature of the Lagrangian along the tangent, and the bordered-Hessian determinant, at each pair of points.
  const onX = 2 * (state.a - 1)
  const onY = 2 * (1 - state.a)
  return (
    <ConstrainedExplorer
      state={state}
      problem={p}
      title="The second-order test: curvature of the Lagrangian along the constraint"
      caption="Left: contours of f(x, y) = x² + a·y² (grey), the unit circle, and the arrow of ∇f against the circle's normal. Drag the point around the circle and set a with the slider. For 0 < a < 1, f is convex, yet (±1, 0) is a constrained maximum: the curvature of the circle outweighs that of f. At a = 1, f is constant on the circle and every point is stationary."
      xLabel="x"
      yLabel="y"
      tLabel="angle θ (degrees)"
      tSymbol="θ"
      fLabel="f"
      formatT={(v) => `${formatNumber(v)}°`}
      readout={() => (
        <>
          <Readout
            label="at (±1, 0): λ = 1, vᵀ∇²ℒv = 2(a − 1)"
            value={`${formatNumber(onX)}, det H̄ = ${formatNumber(-8 * (state.a - 1))}: ${verdict(onX)}`}
          />
          <Readout
            label="at (0, ±1): λ = a, vᵀ∇²ℒv = 2(1 − a)"
            value={`${formatNumber(onY)}, det H̄ = ${formatNumber(8 * (state.a - 1))}: ${verdict(onY)}`}
          />
        </>
      )}
    />
  )
}
