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
} from '@/components/viz'
import { rng } from '@/lib/math'
import { eigSym } from '@/lib/math/mat2'

const N = 80
const ARROW = 3

/** Zero-mean points from N(0, [[3, 1], [1, 2]]), drawn through the Cholesky factor of the covariance. */
function sample(seed: number): [number, number][] {
  const r = rng(seed)
  const l11 = Math.sqrt(3)
  const l21 = 1 / l11
  const l22 = Math.sqrt(2 - l21 * l21)
  const raw = Array.from({ length: N }, () => {
    const a = r.normal()
    const b = r.normal()
    return [l11 * a, l21 * a + l22 * b] as [number, number]
  })
  const mx = raw.reduce((s, p) => s + p[0], 0) / N
  const my = raw.reduce((s, p) => s + p[1], 0) / N
  return raw.map(([x, y]) => [x - mx, y - my])
}

/** Fold an angle in degrees into [-90, 90): the directions u and -u give the same autoencoder. */
const fold = (deg: number) => ((((deg + 90) % 180) + 180) % 180) - 90

export function LinearAutoencoder() {
  const angle = useParam(-20, { min: -90, max: 90, step: 1 })

  const data = useMemo(() => {
    const pts = sample(11)
    const sxx = pts.reduce((s, p) => s + p[0] * p[0], 0) / N
    const sxy = pts.reduce((s, p) => s + p[0] * p[1], 0) / N
    const syy = pts.reduce((s, p) => s + p[1] * p[1], 0) / N
    const eig = eigSym(sxx, sxy, syy)
    const v = eig.vectors[0]
    return { pts, pcaAngle: fold((Math.atan2(v[1], v[0]) * 180) / Math.PI), bestMse: eig.values[1] }
  }, [])

  const fit = useMemo(() => {
    const th = (angle.value * Math.PI) / 180
    const u: [number, number] = [Math.cos(th), Math.sin(th)]
    const recon = data.pts.map(([x, y]) => {
      const z = u[0] * x + u[1] * y
      return [u[0] * z, u[1] * z] as [number, number]
    })
    const mse = data.pts.reduce((s, [x, y], i) => s + (x - recon[i][0]) ** 2 + (y - recon[i][1]) ** 2, 0) / N
    const residuals: Segment[] = data.pts.map((p, i) => ({ from: p, to: recon[i] }))
    const series: XYSeries[] = [
      { name: 'code axis', type: 'line', x: [-8 * u[0], 8 * u[0]], y: [-8 * u[1], 8 * u[1]], muted: true },
      { name: 'data x', type: 'scatter', x: data.pts.map((p) => p[0]), y: data.pts.map((p) => p[1]), slot: 0 },
      { name: 'reconstruction x̂', type: 'scatter', x: recon.map((p) => p[0]), y: recon.map((p) => p[1]), slot: 1 },
    ]
    return { u, mse, residuals, series }
  }, [angle.value, data])

  const vectors = useMemo<Segment[]>(() => [{ from: [0, 0], to: [ARROW * fit.u[0], ARROW * fit.u[1]] }], [fit.u])
  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: [ARROW * fit.u[0], ARROW * fit.u[1]],
        onDrag: ([x, y]) => angle.set(fold((Math.atan2(y, x) * 180) / Math.PI)),
        label: 'u',
      },
    ],
    [fit.u, angle],
  )

  return (
    <Interactive
      title="A linear autoencoder with a one-dimensional code"
      caption="The encoder computes the code z = uᵀx and the decoder returns x̂ = u z. Drag the tip of the arrow u, or use the slider. The grey segments are the reconstruction errors. The mean squared error is smallest when u points along the first principal direction of the data."
      controls={<ParamSlider label="direction of u (degrees)" param={angle} format={(v) => `${v}°`} />}
      readout={
        <>
          <Readout label="reconstruction MSE" value={formatNumber(fit.mse)} />
          <Readout label="smallest possible MSE" value={formatNumber(data.bestMse)} />
          <Readout label="first principal direction" value={`${formatNumber(data.pcaAngle)}°`} />
        </>
      }
    >
      <XYChart
        series={fit.series}
        segments={fit.residuals}
        vectors={vectors}
        handles={handles}
        xRange={[-6, 6]}
        yRange={[-5, 5]}
        equalAspect
        xLabel="x₁"
        yLabel="x₂"
      />
    </Interactive>
  )
}
