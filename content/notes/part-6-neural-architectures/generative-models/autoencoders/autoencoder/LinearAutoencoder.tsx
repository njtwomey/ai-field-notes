import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'
import { eigh2 } from 'aifn/numerics/linalg'

const N = 80
const ARROW = 3

/** Zero-mean points from N(0, [[3, 1], [1, 2]]), drawn through the Cholesky factor of the covariance. */
function sample(seed: number): [number, number][] {
  const r = stream(seed)
  const l11 = Math.sqrt(3)
  const l21 = 1 / l11
  const l22 = Math.sqrt(2 - l21 * l21)
  const raw = Array.from({ length: N }, () => {
    const a = normal(r)
    const b = normal(r)
    return [l11 * a, l21 * a + l22 * b] as [number, number]
  })
  const mx = raw.reduce((s, p) => s + p[0], 0) / N
  const my = raw.reduce((s, p) => s + p[1], 0) / N
  return raw.map(([x, y]) => [x - mx, y - my])
}

/** Fold an angle in degrees into [-90, 90): the directions u and -u give the same autoencoder. */
const fold = (deg: number) => ((((deg + 90) % 180) + 180) % 180) - 90

export function LinearAutoencoder() {
  const state = useFigureState({
    angle: slider(-90, 90, -20, { step: 1, label: 'direction of u (degrees)', format: (v) => `${v}°` }),
  })

  const data = useMemo(() => {
    const pts = sample(11)
    const sxx = pts.reduce((s, p) => s + p[0] * p[0], 0) / N
    const sxy = pts.reduce((s, p) => s + p[0] * p[1], 0) / N
    const syy = pts.reduce((s, p) => s + p[1] * p[1], 0) / N
    const eig = eigh2([
      [sxx, sxy],
      [sxy, syy],
    ])
    const v = eig.vectors[0]
    return { pts, pcaAngle: fold((Math.atan2(v[1], v[0]) * 180) / Math.PI), bestMse: eig.values[1] }
  }, [])

  const fit = useMemo(() => {
    const th = (state.angle * Math.PI) / 180
    const u: [number, number] = [Math.cos(th), Math.sin(th)]
    const recon = data.pts.map(([x, y]) => {
      const z = u[0] * x + u[1] * y
      return [u[0] * z, u[1] * z] as [number, number]
    })
    const mse = data.pts.reduce((s, [x, y], i) => s + (x - recon[i][0]) ** 2 + (y - recon[i][1]) ** 2, 0) / N
    const residuals: Segment[] = data.pts.map((p, i) => ({ from: p, to: recon[i] }))
    const series = [
      { name: 'code axis', x: [-8 * u[0], 8 * u[0]], y: [-8 * u[1], 8 * u[1]], muted: true },
      { name: 'data x', x: data.pts.map((p) => p[0]), y: data.pts.map((p) => p[1]), slot: 0 },
      { name: 'reconstruction x̂', x: recon.map((p) => p[0]), y: recon.map((p) => p[1]), slot: 1 },
    ] as const
    return { u, mse, residuals, series }
  }, [state.angle, data])

  const vectors = useMemo<Segment[]>(() => [{ from: [0, 0], to: [ARROW * fit.u[0], ARROW * fit.u[1]] }], [fit.u])
  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: [ARROW * fit.u[0], ARROW * fit.u[1]],
        onDrag: ([x, y]) => state.set('angle', fold((Math.atan2(y, x) * 180) / Math.PI)),
        label: 'u',
      },
    ],
    [fit.u, state.bind('angle')],
  )

  const xAxis = useAxis({ label: 'x₁', range: [-6, 6] })
  const yAxis = useAxis({ label: 'x₂', range: [-5, 5], equal: xAxis })
  return (
    <Figure
      title="A linear autoencoder with a one-dimensional code"
      state={state}
      caption="The encoder computes the code z = uᵀx and the decoder returns x̂ = u z. Drag the tip of the arrow u, or use the slider. The grey segments are the reconstruction errors. The mean squared error is smallest when u points along the first principal direction of the data."

      readouts={
        <>
          <Readout label="reconstruction MSE" value={formatNumber(fit.mse)} />
          <Readout label="smallest possible MSE" value={formatNumber(data.bestMse)} />
          <Readout label="first principal direction" value={`${formatNumber(data.pcaAngle)}°`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...fit.series[0]} />
        <Points {...fit.series[1]} />
        <Points {...fit.series[2]} />
        <Segments segments={fit.residuals} />
        <Vectors vectors={vectors} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
