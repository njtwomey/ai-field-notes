import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, type ConstrainedProblem } from './ConstrainedExplorer'
import { tField } from './t-field'
import { eigh2 } from 'aifn-compute/numerics/linalg'

const DEG = Math.PI / 180

/** Maximise xᵀAx for A = [[p, q], [q, r]] on the unit circle xᵀx = 1. */
function problem(p: number, q: number, r: number): ConstrainedProblem {
  return {
    f: (x, y) => p * x * x + 2 * q * x * y + r * y * y,
    gradF: (x, y) => [2 * (p * x + q * y), 2 * (q * x + r * y)],
    gradG: (x, y) => [2 * x, 2 * y],
    curve: (deg) => [Math.cos(deg * DEG), Math.sin(deg * DEG)],
    tRange: [0, 360],
    closed: true,
    xRange: [-2, 2],
    yRange: [-2, 2],
    goal: 'max',
  }
}

const angleOf = ([x, y]: [number, number]) => ((Math.atan2(y, x) / DEG + 360) % 180).toFixed(1)

export function RayleighQuotient() {
  const state = useFigureState({
    t: tField([0, 360], 100, 0.5, 'angle θ (degrees)', (v) => `${formatNumber(v)}°`),
    p: slider(-2, 2, 2, { step: 0.05, label: 'A₁₁' }),
    q: slider(-2, 2, 0.7, { step: 0.05, label: 'A₁₂ = A₂₁' }),
    r: slider(-2, 2, 1, { step: 0.05, label: 'A₂₂' }),
  })
  const prob = useMemo(() => problem(state.p, state.q, state.r), [state.p, state.q, state.r])
  const eig = eigh2([
    [state.p, state.q],
    [state.q, state.r],
  ])
  return (
    <ConstrainedExplorer
      state={state}
      problem={prob}
      title="The Rayleigh quotient: stationary points are eigenvectors"
      caption="Left: contours of xᵀAx (grey; ellipses when A is definite, hyperbolas when it is indefinite), the unit circle, and the arrow of ∇f = 2Ax against the circle's normal x. Drag the point around the circle, and set the entries of the symmetric matrix A with the sliders. Right: xᵀAx along the circle. The stationary points are ± the two eigenvectors; the values there are the eigenvalues."
      xLabel="x₁"
      yLabel="x₂"
      tLabel="angle θ (degrees)"
      tSymbol="θ"
      fLabel="xᵀAx"
      formatT={(v) => `${formatNumber(v)}°`}
      readout={() => (
        <>
          <Readout label="eigenvalues of A" value={`${formatNumber(eig.values[0])}, ${formatNumber(eig.values[1])}`} />
          <Readout
            label="eigenvector angles"
            value={`${angleOf(eig.vectors[0])}°, ${angleOf(eig.vectors[1])}° (mod 180°)`}
          />
        </>
      )}
    />
  )
}
