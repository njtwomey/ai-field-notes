import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { eigh2 } from 'aifn-compute/numerics/linalg'

const START: [number, number] = [-1, 1.5]
const MAX_STEPS = 5000
const TOL = 1e-6
const ANGLES = toFlat(linspace(0, 2 * Math.PI, 97))

/**
 * Gradient descent on the least-squares loss of two features with correlation ρ, the second measured in units s times
 * larger than the first. Standardising turns the Hessian [[1, ρs], [ρs, s²]] into [[1, ρ], [ρ, 1]].
 */
function descend(scale: number, rho: number, standardise: boolean) {
  const s = standardise ? 1 : scale
  const [p, q, r] = [1, rho * s, s * s]
  const eig = eigh2([
    [p, q],
    [q, r],
  ])
  const eta = 1 / eig.values[0]
  const opt: [number, number] = [1, 1 / s]
  const loss = (w: [number, number]) => {
    const d0 = w[0] - opt[0]
    const d1 = w[1] - opt[1]
    return 0.5 * (p * d0 * d0 + 2 * q * d0 * d1 + r * d1 * d1)
  }
  const start = loss(START)
  const path: [number, number][] = [START]
  let w = START
  while (path.length <= MAX_STEPS && loss(w) > TOL * start) {
    const d0 = w[0] - opt[0]
    const d1 = w[1] - opt[1]
    w = [w[0] - eta * (p * d0 + q * d1), w[1] - eta * (q * d0 + r * d1)]
    path.push(w)
  }
  // Level sets ½dᵀHd = c are ellipses with semi-axes √(2c/λₖ) along the eigenvectors.
  const cx: number[] = []
  const cy: number[] = []
  for (const c of [1, 0.3, 0.1, 0.03, 0.01].map((f) => f * start)) {
    ANGLES.forEach((t) => {
      const a = Math.sqrt((2 * c) / eig.values[0]) * Math.cos(t)
      const b = Math.sqrt((2 * c) / eig.values[1]) * Math.sin(t)
      cx.push(opt[0] + a * eig.vectors[0][0] + b * eig.vectors[1][0])
      cy.push(opt[1] + a * eig.vectors[0][1] + b * eig.vectors[1][1])
    })
    cx.push(NaN)
    cy.push(NaN)
  }
  return { path, cx, cy, opt, kappa: eig.values[0] / eig.values[1], steps: path.length - 1 }
}

/** How the units of one feature change the shape of the loss and the speed of gradient descent. */
export function ScalingDescent() {
  const state = useFigureState({
    scale: slider(1, 10, 5, { step: 0.5, label: 'scale ratio s' }),
    rho: slider(0, 0.9, 0, { step: 0.05, label: 'correlation ρ' }),
    standardise: setting(false, 'standardise features'),
  })
  const run = useMemo(
    () => descend(state.scale, state.rho, state.standardise),
    [state.scale, state.rho, state.standardise],
  )

  const series = useMemo(
    () =>
      [
        { name: 'loss contours', x: run.cx, y: run.cy, muted: true },
        { name: 'gradient descent', x: run.path.map((w) => w[0]), y: run.path.map((w) => w[1]), slot: 0 },
        { name: 'minimum', x: [run.opt[0]], y: [run.opt[1]], emphasis: true },
      ] as const,
    [run],
  )
  const converged = run.steps < MAX_STEPS

  const xAxis = useAxis({ label: 'w₁', range: [-1.5, 2] })
  const yAxis = useAxis({ label: 'w₂', range: [-1, 2], equal: xAxis })
  return (
    <Figure
      title="Gradient descent before and after standardising"
      purpose="Change the scale ratio and correlation of two features and compare gradient descent paths with and without standardisation."
      state={state}
      caption="Least squares with two features. The second feature is measured in units s times larger than the first, and the two have correlation ρ. Gradient descent uses the learning rate 1/λ_max, where λ_max is the largest eigenvalue of the Hessian. Unscaled, the contours are long thin ellipses and descent crawls along the valley. Standardised, with ρ = 0 they are circles and descent reaches the minimum in one step; correlation alone still stretches them."

      readouts={
        <>
          <Readout label="condition number κ" value={formatNumber(run.kappa)} />
          <Readout label="steps to reduce the loss by 10⁶" value={converged ? String(run.steps) : `> ${MAX_STEPS}`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <Plot
          x={xAxis}
          y={yAxis}
          ariaLabel={'Loss contours and the gradient descent path in the plane of the two coefficients'}
        >
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Points {...series[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
