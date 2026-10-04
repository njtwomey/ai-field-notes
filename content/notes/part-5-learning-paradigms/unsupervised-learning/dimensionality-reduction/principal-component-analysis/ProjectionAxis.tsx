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
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'
import { eigh2 } from 'aifn/numerics/linalg'

const N = 100
const ARROW = 3

/** Centred points from a correlated Gaussian: x₂ = 0.6 x₁ + noise. */
function sample(): [number, number][] {
  const r = stream(5)
  const raw = Array.from({ length: N }, () => {
    const a = 1.6 * normal(r)
    return [a, 0.6 * a + 0.6 * normal(r)] as [number, number]
  })
  const mx = raw.reduce((s, p) => s + p[0], 0) / N
  const my = raw.reduce((s, p) => s + p[1], 0) / N
  return raw.map(([x, y]) => [x - mx, y - my])
}

/** Fold an angle in degrees into [−90, 90): u and −u span the same line. */
const fold = (deg: number) => ((((deg + 90) % 180) + 180) % 180) - 90
const rad = (deg: number) => (deg * Math.PI) / 180

export function ProjectionAxis() {
  const state = useFigureState({
    angle: slider(-90, 90, -30, { step: 1, label: 'direction of u (degrees)', format: (v) => `${v}°` }),
  })
  const data = useMemo(() => {
    const pts = sample()
    const sxx = pts.reduce((s, p) => s + p[0] * p[0], 0) / N
    const sxy = pts.reduce((s, p) => s + p[0] * p[1], 0) / N
    const syy = pts.reduce((s, p) => s + p[1] * p[1], 0) / N
    const eig = eigh2([
      [sxx, sxy],
      [sxy, syy],
    ])
    const v = eig.vectors[0]
    // Projected variance u'Su along every direction, for the curve in the right panel.
    const thetas = toFlat(linspace(-90, 90, 181))
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
    const u: [number, number] = [Math.cos(rad(state.angle)), Math.sin(rad(state.angle))]
    const proj = data.pts.map(([x, y]) => {
      const z = u[0] * x + u[1] * y
      return [u[0] * z, u[1] * z] as [number, number]
    })
    const variance = data.pts.reduce((s, [x, y]) => s + (u[0] * x + u[1] * y) ** 2, 0) / N
    const residuals: Segment[] = data.pts.map((p, i) => ({ from: p, to: proj[i] }))
    const series = [
      { name: 'axis', x: [-8 * u[0], 8 * u[0]], y: [-8 * u[1], 8 * u[1]], muted: true },
      { name: 'data', x: data.pts.map((p) => p[0]), y: data.pts.map((p) => p[1]), slot: 0 },
      { name: 'projection', x: proj.map((p) => p[0]), y: proj.map((p) => p[1]), slot: 1 },
    ] as const
    return { u, variance, error: data.total - variance, residuals, series }
  }, [state.angle, data])

  const vectors = useMemo<Segment[]>(() => [{ from: [0, 0], to: [ARROW * fit.u[0], ARROW * fit.u[1]] }], [fit.u])
  const tipHandle = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: [ARROW * fit.u[0], ARROW * fit.u[1]],
        label: 'u',
        onDrag: ([x, y]) => state.set('angle', fold((Math.atan2(y, x) * 180) / Math.PI)),
      },
    ],
    [fit.u, state.bind('angle')],
  )
  const angleHandle = useMemo<Handle[]>(
    () => [{ kind: 'x', at: state.angle, label: 'direction', onDrag: (x) => state.set('angle', x) }],
    [state.bind('angle')],
  )

  const xAxis = useAxis({ label: 'x₁', range: [-5, 5] })
  const yAxis = useAxis({ label: 'x₂', range: [-4, 4], equal: xAxis })
  const xAxis2 = useAxis({ label: 'direction of u (degrees)', range: [-90, 90] })
  const yAxis2 = useAxis({ label: 'per point', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Maximum variance is minimum reconstruction error"
      state={state}
      caption="Left: each point is projected onto the line through the origin in direction u; the grey segments are the reconstruction errors. Drag the tip of the arrow u, or the vertical line on the right, or use the slider. Right: the variance of the projections and the mean squared reconstruction error for every direction. They always sum to the total variance, so the direction that maximises one minimises the other: the first principal component."

      readouts={
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
        <Plot x={xAxis} y={yAxis}>
          <Curve {...fit.series[0]} />
          <Points {...fit.series[1]} />
          <Points {...fit.series[2]} />
          <Segments segments={fit.residuals} />
          <Vectors vectors={vectors} />
          {(tipHandle ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve name="projected variance" x={data.thetas} y={data.variance} slot={1} />
          <Curve name="reconstruction MSE" x={data.thetas} y={data.error} slot={2} />
          <Points name="current u" x={[state.angle, state.angle]} y={[fit.variance, fit.error]} emphasis />
          {(angleHandle ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
