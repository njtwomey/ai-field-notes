import { useMemo } from 'react'
import { ParamSlider, Readout, formatNumber, useParam } from 'aifn-render'
import { eigSym } from '@/lib/math/mat2'
import { ConstrainedExplorer, type ConstrainedProblem } from './ConstrainedExplorer'

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
  const p = useParam(2, { min: -2, max: 2, step: 0.05 })
  const q = useParam(0.7, { min: -2, max: 2, step: 0.05 })
  const r = useParam(1, { min: -2, max: 2, step: 0.05 })
  const prob = useMemo(() => problem(p.value, q.value, r.value), [p.value, q.value, r.value])
  const eig = eigSym(p.value, q.value, r.value)
  return (
    <ConstrainedExplorer
      problem={prob}
      title="The Rayleigh quotient: stationary points are eigenvectors"
      caption="Left: contours of xᵀAx (grey; ellipses when A is definite, hyperbolas when it is indefinite), the unit circle, and the arrow of ∇f = 2Ax against the circle's normal x. Drag the point around the circle, and set the entries of the symmetric matrix A with the sliders. Right: xᵀAx along the circle. The stationary points are ± the two eigenvectors; the values there are the eigenvalues."
      initialT={100}
      tStep={0.5}
      xLabel="x₁"
      yLabel="x₂"
      tLabel="angle θ (degrees)"
      tSymbol="θ"
      fLabel="xᵀAx"
      formatT={(v) => `${formatNumber(v)}°`}
      controls={
        <>
          <ParamSlider label="A₁₁" param={p} />
          <ParamSlider label="A₁₂ = A₂₁" param={q} />
          <ParamSlider label="A₂₂" param={r} />
        </>
      }
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
