import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { eigSym } from '@/lib/math/mat2'

const START: [number, number] = [-1, 1.5]
const MAX_STEPS = 5000
const TOL = 1e-6
const ANGLES = linspace(0, 2 * Math.PI, 97)

/**
 * Gradient descent on the least-squares loss of two features with correlation ρ, the second measured in units s times
 * larger than the first. Standardising turns the Hessian [[1, ρs], [ρs, s²]] into [[1, ρ], [ρ, 1]].
 */
function descend(scale: number, rho: number, standardise: boolean) {
  const s = standardise ? 1 : scale
  const [p, q, r] = [1, rho * s, s * s]
  const eig = eigSym(p, q, r)
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
  const scale = useParam(5, { min: 1, max: 10, step: 0.5 })
  const rho = useParam(0, { min: 0, max: 0.9, step: 0.05 })
  const [standardise, setStandardise] = useState(false)
  const run = useMemo(() => descend(scale.value, rho.value, standardise), [scale.value, rho.value, standardise])

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'loss contours', type: 'line', x: run.cx, y: run.cy, muted: true },
      { name: 'gradient descent', type: 'line', x: run.path.map((w) => w[0]), y: run.path.map((w) => w[1]), slot: 0 },
      { name: 'minimum', type: 'scatter', x: [run.opt[0]], y: [run.opt[1]], emphasis: true },
    ],
    [run],
  )
  const converged = run.steps < MAX_STEPS

  return (
    <Interactive
      title="Gradient descent before and after standardising"
      caption="Least squares with two features. The second feature is measured in units s times larger than the first, and the two have correlation ρ. Gradient descent uses the learning rate 1/λ_max, where λ_max is the largest eigenvalue of the Hessian. Unscaled, the contours are long thin ellipses and descent crawls along the valley. Standardised, with ρ = 0 they are circles and descent reaches the minimum in one step; correlation alone still stretches them."
      controls={
        <>
          <ParamSlider label="scale ratio s" param={scale} />
          <ParamSlider label="correlation ρ" param={rho} />
          <ParamSwitch label="standardise features" checked={standardise} onChange={setStandardise} />
        </>
      }
      readout={
        <>
          <Readout label="condition number κ" value={formatNumber(run.kappa)} />
          <Readout label="steps to reduce the loss by 10⁶" value={converged ? String(run.steps) : `> ${MAX_STEPS}`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart
          series={series}
          xLabel="w₁"
          yLabel="w₂"
          xRange={[-1.5, 2]}
          yRange={[-1, 2]}
          equalAspect
          ariaLabel="Loss contours and the gradient descent path in the plane of the two coefficients"
        />
      </div>
    </Interactive>
  )
}
