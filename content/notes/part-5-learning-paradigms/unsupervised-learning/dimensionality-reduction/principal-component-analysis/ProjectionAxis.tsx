import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { eigSym } from '@/lib/math/mat2'

const N = 100
const ARROW = 3

/** Centred points from a correlated Gaussian: x₂ = 0.6 x₁ + noise. */
function sample(): [number, number][] {
  const r = rng(5)
  const raw = Array.from({ length: N }, () => {
    const a = 1.6 * r.normal()
    return [a, 0.6 * a + 0.6 * r.normal()] as [number, number]
  })
  const mx = raw.reduce((s, p) => s + p[0], 0) / N
  const my = raw.reduce((s, p) => s + p[1], 0) / N
  return raw.map(([x, y]) => [x - mx, y - my])
}

/** Fold an angle in degrees into [−90, 90): u and −u span the same line. */
const fold = (deg: number) => ((((deg + 90) % 180) + 180) % 180) - 90
const rad = (deg: number) => (deg * Math.PI) / 180

export function ProjectionAxis() {
  const angle = useParam(-30, { min: -90, max: 90, step: 1 })
  const data = useMemo(() => {
    const pts = sample()
    const sxx = pts.reduce((s, p) => s + p[0] * p[0], 0) / N
    const sxy = pts.reduce((s, p) => s + p[0] * p[1], 0) / N
    const syy = pts.reduce((s, p) => s + p[1] * p[1], 0) / N
    const eig = eigSym(sxx, sxy, syy)
    const v = eig.vectors[0]
    // Projected variance u'Su along every direction, for the curve in the right panel.
    const thetas = linspace(-90, 90, 181)
    const variance = thetas.map((t) => {
      const [c, s] = [Math.cos(rad(t)), Math.sin(rad(t))]
      return c * c * sxx + 2 * c * s * sxy + s * s * syy
    })
    const total = sxx + syy
    return {
      pts,
      total,
      thetas,
      variance,
      error: variance.map((v) => total - v),
      eig,
      pc1: fold((Math.atan2(v[1], v[0]) * 180) / Math.PI),
    }
  }, [])

  const fit = useMemo(() => {
    const u: [number, number] = [Math.cos(rad(angle.value)), Math.sin(rad(angle.value))]
    const proj = data.pts.map(([x, y]) => {
      const z = u[0] * x + u[1] * y
      return [u[0] * z, u[1] * z] as [number, number]
    })
    const variance = data.pts.reduce((s, [x, y]) => s + (u[0] * x + u[1] * y) ** 2, 0) / N
    const residuals: Segment[] = data.pts.map((p, i) => ({ from: p, to: proj[i] }))
    const series: XYSeries[] = [
      { name: 'axis', type: 'line', x: [-8 * u[0], 8 * u[0]], y: [-8 * u[1], 8 * u[1]], muted: true },
      { name: 'data', type: 'scatter', x: data.pts.map((p) => p[0]), y: data.pts.map((p) => p[1]), slot: 0 },
      { name: 'projection', type: 'scatter', x: proj.map((p) => p[0]), y: proj.map((p) => p[1]), slot: 1 },
    ]
    return { u, variance, error: data.total - variance, residuals, series }
  }, [angle.value, data])

  const vectors = useMemo<Segment[]>(() => [{ from: [0, 0], to: [ARROW * fit.u[0], ARROW * fit.u[1]] }], [fit.u])
  const tipHandle = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: [ARROW * fit.u[0], ARROW * fit.u[1]],
        label: 'u',
        onDrag: ([x, y]) => angle.set(fold((Math.atan2(y, x) * 180) / Math.PI)),
      },
    ],
    [fit.u, angle],
  )
  const angleHandle = useMemo<Handle[]>(
    () => [{ kind: 'x', at: angle.value, label: 'direction', onDrag: (x) => angle.set(x) }],
    [angle],
  )

  return (
    <Interactive
      title="Maximum variance is minimum reconstruction error"
      caption="Left: each point is projected onto the line through the origin in direction u; the grey segments are the reconstruction errors. Drag the tip of the arrow u, or the vertical line on the right, or use the slider. Right: the variance of the projections and the mean squared reconstruction error for every direction. They always sum to the total variance, so the direction that maximises one minimises the other: the first principal component."
      controls={<ParamSlider label="direction of u (degrees)" param={angle} format={(v) => `${v}°`} />}
      readout={
        <>
          <Readout label="projected variance uᵀSu" value={formatNumber(fit.variance)} />
          <Readout label="reconstruction MSE" value={formatNumber(fit.error)} />
          <Readout label="sum (total variance)" value={formatNumber(data.total)} />
          <Readout
            label="first principal component"
            value={`${formatNumber(data.pc1)}°, λ₁ = ${formatNumber(data.eig.values[0])}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={fit.series}
          segments={fit.residuals}
          vectors={vectors}
          handles={tipHandle}
          xRange={[-5, 5]}
          yRange={[-4, 4]}
          equalAspect
          xLabel="x₁"
          yLabel="x₂"
        />
        <XYChart
          height={320}
          xLabel="direction of u (degrees)"
          yLabel="per point"
          xRange={[-90, 90]}
          yRange={[0, undefined]}
          handles={angleHandle}
          series={[
            { name: 'projected variance', type: 'line', x: data.thetas, y: data.variance, slot: 1 },
            { name: 'reconstruction MSE', type: 'line', x: data.thetas, y: data.error, slot: 2 },
            {
              name: 'current u',
              type: 'scatter',
              x: [angle.value, angle.value],
              y: [fit.variance, fit.error],
              emphasis: true,
            },
          ]}
        />
      </div>
    </Interactive>
  )
}
