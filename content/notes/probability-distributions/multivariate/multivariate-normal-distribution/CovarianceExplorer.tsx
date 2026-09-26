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
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'

const N = 400
// Fixed standard-normal pairs z; every sample is μ + L z, so moving a slider deforms the same cloud smoothly.
const Z = (() => {
  const r = rng(3)
  return Array.from({ length: N }, () => [r.normal(), r.normal()] as const)
})()
const ANGLES = linspace(0, 2 * Math.PI, 97)
const RANGE: [number, number] = [-6, 6]

/** Samples, 1σ and 2σ contours of a bivariate Gaussian, with the covariance built from σ₁, σ₂ and ρ. */
export function CovarianceExplorer() {
  const [s1, setS1] = useState(1.5)
  const [s2, setS2] = useState(1)
  const [rho, setRho] = useState(0.6)
  const mx = useParam(0, { min: -3, max: 3, step: 0.1 })
  const my = useParam(0, { min: -3, max: 3, step: 0.1 })

  const { series, inside } = useMemo(() => {
    // Cholesky factor of Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]: L = [[σ₁, 0], [ρσ₂, σ₂√(1 − ρ²)]].
    const l21 = rho * s2
    const l22 = s2 * Math.sqrt(1 - rho * rho)
    const map = (a: number, b: number): [number, number] => [mx.value + s1 * a, my.value + l21 * a + l22 * b]
    const points = Z.map(([a, b]) => map(a, b))
    // A sample is inside the radius-r contour when its z has length below r (Mahalanobis distance = ‖z‖).
    const inside = [1, 2].map((r) => Z.filter(([a, b]) => a * a + b * b < r * r).length / N)
    const contour = (r: number): XYSeries => {
      const pts = ANGLES.map((t) => map(r * Math.cos(t), r * Math.sin(t)))
      return {
        name: `${r}σ contour`,
        type: 'line',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        slot: 1,
        dashed: r === 2,
      }
    }
    const series: XYSeries[] = [
      { name: 'samples', type: 'scatter', x: points.map((p) => p[0]), y: points.map((p) => p[1]), slot: 0 },
      contour(1),
      contour(2),
    ]
    return { series, inside }
  }, [s1, s2, rho, mx.value, my.value])

  const cov = rho * s1 * s2
  const det = s1 * s1 * s2 * s2 * (1 - rho * rho)
  // Eigenvalues of the 2 × 2 covariance: half the trace ± the discriminant.
  const half = (s1 * s1 + s2 * s2) / 2
  const disc = Math.sqrt(((s1 * s1 - s2 * s2) / 2) ** 2 + cov * cov)
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [mx.value, my.value],
      label: 'mean',
      onDrag: ([x, y]) => {
        mx.set(x)
        my.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="Covariance as shape"
      caption="Each sample is μ + Lz for a fixed set of standard-normal pairs z, where LLᵀ = Σ. The standard deviations stretch the cloud along the axes and the correlation ρ shears it. The contours are the points at Mahalanobis distance 1 (solid) and 2 (dashed); they hold about 39% and 86% of the mass in two dimensions. Drag the mean to move the whole distribution."
      controls={
        <>
          <ParamSlider label="σ₁" value={s1} onChange={setS1} min={0.3} max={3} step={0.1} />
          <ParamSlider label="σ₂" value={s2} onChange={setS2} min={0.3} max={3} step={0.1} />
          <ParamSlider label="correlation ρ" value={rho} onChange={setRho} min={-0.95} max={0.95} step={0.05} />
        </>
      }
      readout={
        <>
          <Readout label="cov(x₁, x₂)" value={formatNumber(cov)} />
          <Readout label="det Σ" value={formatNumber(det)} />
          <Readout label="eigenvalues" value={`${formatNumber(half + disc)}, ${formatNumber(half - disc)}`} />
          <Readout label="inside 1σ" value={`${formatNumber(100 * inside[0])}%`} />
          <Readout label="inside 2σ" value={`${formatNumber(100 * inside[1])}%`} />
        </>
      }
    >
      <XYChart series={series} xLabel="x₁" yLabel="x₂" xRange={RANGE} yRange={RANGE} equalAspect handles={handles} />
    </Interactive>
  )
}
