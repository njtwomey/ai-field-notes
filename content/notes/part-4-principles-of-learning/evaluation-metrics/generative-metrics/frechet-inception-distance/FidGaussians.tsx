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
import { frechet2d, type Cov, type Vec } from '../_shared/fid'

const RANGE: [number, number] = [-5, 5]
const ANGLES = linspace(0, 2 * Math.PI, 97)
const REAL_MEAN: Vec = [0, 0]
const REAL_COV: Cov = [
  [1, 0.5],
  [0.5, 1],
]

const cov = (s1: number, s2: number, rho: number): Cov => [
  [s1 * s1, rho * s1 * s2],
  [rho * s1 * s2, s2 * s2],
]

/** The 1σ and 2σ contours of a 2-D Gaussian, via its Cholesky factor. */
function contour(name: string, mean: Vec, c: Cov, radius: number, style: Partial<XYSeries>): XYSeries {
  const l11 = Math.sqrt(c[0][0])
  const l21 = c[1][0] / l11
  const l22 = Math.sqrt(Math.max(c[1][1] - l21 * l21, 0))
  const pts = ANGLES.map((t) => {
    const [a, b] = [radius * Math.cos(t), radius * Math.sin(t)]
    return [mean[0] + l11 * a, mean[1] + l21 * a + l22 * b]
  })
  return { name, type: 'line', x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), ...style }
}

/**
 * A "real" feature distribution (fixed) and a "generated" one whose mean is dragged and whose covariance is set by
 * sliders. FID splits into a mean term and a covariance term; collapsing the generated spread (mode collapse) raises
 * the covariance term even when the means agree.
 */
export function FidGaussians() {
  const [mean, setMean] = useState<Vec>([1, 0.5])
  const s1 = useParam(1.2, { min: 0.1, max: 2.5, step: 0.05 })
  const s2 = useParam(0.8, { min: 0.1, max: 2.5, step: 0.05 })
  const rho = useParam(0, { min: -0.95, max: 0.95, step: 0.05 })

  const gen = cov(s1.value, s2.value, rho.value)
  const r = useMemo(() => frechet2d(REAL_MEAN, REAL_COV, mean, gen), [mean, gen[0][0], gen[0][1], gen[1][1]]) // eslint-disable-line react-hooks/exhaustive-deps

  const series: XYSeries[] = [
    contour('real', REAL_MEAN, REAL_COV, 1, { slot: 0 }),
    contour('real', REAL_MEAN, REAL_COV, 2, { slot: 0, dashed: true }),
    contour('generated', mean, gen, 1, { slot: 1 }),
    contour('generated', mean, gen, 2, { slot: 1, dashed: true }),
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: mean,
      label: 'generated mean',
      onDrag: ([x, y]) => setMean([Math.max(-4, Math.min(4, x)), Math.max(-4, Math.min(4, y))]),
    },
  ]

  return (
    <Interactive
      title="The Fréchet distance between two Gaussians"
      caption="FID fits a Gaussian to the features of real images and another to the features of generated ones, and measures the Fréchet distance between them. Here the features are two-dimensional so the Gaussians can be drawn: solid lines are 1σ contours, dashed 2σ. Drag the generated mean or change its spread. The mean term is the squared distance between centres; the covariance term is zero only when the covariances match, so shrinking the generated spread (as in mode collapse) is penalised even with identical means."
      controls={
        <>
          <ParamSlider label="generated σ₁" param={s1} />
          <ParamSlider label="generated σ₂" param={s2} />
          <ParamSlider label="generated correlation ρ" param={rho} />
        </>
      }
      readout={
        <>
          <Readout label="mean term ‖μ₁ − μ₂‖²" value={formatNumber(r.meanTerm)} />
          <Readout label="covariance term" value={formatNumber(r.covTerm)} />
          <Readout label="FID" value={formatNumber(r.fid)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart
          series={series}
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
          handles={handles}
          xLabel="feature 1"
          yLabel="feature 2"
        />
      </div>
    </Interactive>
  )
}
