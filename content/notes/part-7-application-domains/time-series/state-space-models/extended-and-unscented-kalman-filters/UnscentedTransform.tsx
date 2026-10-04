import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'

type Fn = 'sin' | 'square' | 'exp'

const FUNCTIONS: Record<Fn, { g: (z: number) => number; dg: (z: number) => number; label: string }> = {
  sin: { g: Math.sin, dg: Math.cos, label: 'g(z) = sin z' },
  square: { g: (z) => z * z, dg: (z) => 2 * z, label: 'g(z) = z²' },
  exp: { g: Math.exp, dg: Math.exp, label: 'g(z) = exp z' },
}

const GRID = 4001
const BINS = 60

const gaussian = (y: number, m: number, v: number) => Math.exp(-((y - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)

/**
 * Push z ~ N(μ, σ²) through g three ways: exactly (a fine grid over μ ± 6σ, weighted by the density), by linearising at
 * μ (the extended Kalman filter), and by the unscented transform with κ = 2: sigma points μ and μ ± √3 σ with weights
 * 2/3, 1/6, 1/6.
 */
export function UnscentedTransform() {
  const state = useFigureState({
    fn: choice<Fn>(
      [
        { value: 'sin', label: 'sin z' },
        { value: 'square', label: 'z²' },
        { value: 'exp', label: 'exp z' },
      ],
      'sin',
      { label: 'function' },
    ),
    mu: float(Math.PI / 2, { min: -2, max: 2, step: 0.05, label: 'input mean μ' }),
    sigma: float(1, { min: 0.05, max: 1.5, step: 0.05, label: 'input sd σ' }),
  })

  const r = useMemo(() => {
    const { g, dg } = FUNCTIONS[state.fn]
    const m = state.mu
    const s = state.sigma
    // Exact moments and density of y = g(z), from a grid over z weighted by the normal density.
    const zs = Array.from({ length: GRID }, (_, i) => m + s * (-6 + (12 * i) / (GRID - 1)))
    const weights = zs.map((z) => gaussian(z, m, s * s))
    const total = weights.reduce((a, b) => a + b, 0)
    const ys = zs.map(g)
    const mean = ys.reduce((a, y, i) => a + weights[i] * y, 0) / total
    const variance = ys.reduce((a, y, i) => a + weights[i] * (y - mean) ** 2, 0) / total
    // Histogram over the range of g on μ ± 3.5σ, which holds all but 0.05% of the mass.
    const inner = ys.filter((_, i) => Math.abs(zs[i] - m) <= 3.5 * s)
    const lo = Math.min(...inner)
    const hi = Math.max(...inner)
    const width = (hi - lo) / BINS || 1
    const mass = new Array<number>(BINS).fill(0)
    ys.forEach((y, i) => {
      const b = Math.floor((y - lo) / width)
      if (b >= 0 && b < BINS) mass[b] += weights[i] / total
    })
    const centres = mass.map((_, b) => lo + (b + 0.5) * width)
    const density = mass.map((p) => p / width)

    const ekf = { mean: g(m), variance: dg(m) ** 2 * s * s }
    const points = [m, m + Math.sqrt(3) * s, m - Math.sqrt(3) * s]
    const w = [2 / 3, 1 / 6, 1 / 6]
    const images = points.map(g)
    const utMean = images.reduce((a, y, i) => a + w[i] * y, 0)
    const ut = { mean: utMean, variance: images.reduce((a, y, i) => a + w[i] * (y - utMean) ** 2, 0) }
    return { g, dg, mean, variance, centres, density, lo, hi, ekf, ut, points, images }
  }, [state.fn, state.mu, state.sigma])

  const m = state.mu
  const s = state.sigma
  const zAxis = Array.from({ length: 201 }, (_, i) => m - 3.5 * s + (7 * s * i) / 200)
  const top = [
    { name: FUNCTIONS[state.fn].label, x: zAxis, y: zAxis.map(r.g), emphasis: true },
    {
      name: 'tangent at μ (EKF)',
      x: zAxis,
      y: zAxis.map((z) => r.g(m) + r.dg(m) * (z - m)),
      slot: 0,
      dashed: true,
    },
    { name: 'sigma points (UKF)', x: r.points, y: r.images, slot: 1 },
  ] as const
  const span = r.hi - r.lo || 1
  const ys = Array.from({ length: 301 }, (_, i) => r.lo - 0.15 * span + (1.3 * span * i) / 300)
  const peak = Math.max(...r.density)
  const curve = (name: string, est: { mean: number; variance: number }, slot: number): SeriesSpec =>
    est.variance < 1e-9
      ? // A zero-variance estimate is a point mass: draw it as a vertical line at its mean.
        { name, type: 'line', x: [est.mean, est.mean], y: [0, peak * 1.2], slot }
      : { name, type: 'line', x: ys, y: ys.map((y) => gaussian(y, est.mean, est.variance)), slot }
  const bottom: SeriesSpec[] = [
    { name: 'true density of y', type: 'bar', x: r.centres, y: r.density, muted: true },
    curve('linearised (EKF)', r.ekf, 0),
    curve('unscented (UKF)', r.ut, 1),
    curve('Gaussian with the true moments', { mean: r.mean, variance: r.variance }, 2),
  ]
  // Cap the density axis at the exact density's scale, so a near-degenerate linearised Gaussian does not flatten the rest.
  const yMax = 1.4 * Math.max(peak, 1 / Math.sqrt(2 * Math.PI * Math.max(r.variance, 1e-9)))
  const fmt = (est: { mean: number; variance: number }) =>
    `${formatNumber(est.mean)}, ${formatNumber(est.variance < 1e-12 ? 0 : Math.sqrt(est.variance))}`

  const xAxis = useAxis({ label: 'z', hold: 'union' })
  const yAxis = useAxis({ label: 'g(z)', hold: 'union' })
  const xAxis2 = useAxis({ label: 'y', hold: 'union' })
  const yAxis2 = useAxis({ label: 'density', range: [0, yMax] })
  return (
    <Figure
      title="A Gaussian through a nonlinearity"
      state={state}
      caption="Top: the function g, its tangent at the input mean (what the extended Kalman filter uses) and the three sigma points of the unscented transform. Bottom: the exact distribution of y = g(z) for z ~ N(μ, σ²), with three Gaussian approximations. At μ = π/2 the sine has zero slope, so linearisation reports no uncertainty at all, while the sigma points see the curvature. For g(z) = z² the unscented transform is exact."

      readouts={
        <>
          <Readout label="mean, sd of y: exact" value={fmt({ mean: r.mean, variance: r.variance })} />
          <Readout label="linearised" value={fmt(r.ekf)} />
          <Readout label="unscented" value={fmt(r.ut)} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Curve {...top[0]} />
          <Curve {...top[1]} />
          <Points {...top[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={240}>
          {seriesLayers(bottom)}
        </Plot>
      </div>
    </Figure>
  )
}
