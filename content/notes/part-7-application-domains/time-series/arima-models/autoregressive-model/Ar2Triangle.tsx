import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { minRootModulus, quadraticRoots, simulateArma, theoreticalAcf } from '../_shared/arma'

const H = 20
const LAGS = Array.from({ length: H }, (_, i) => i + 1)
const N = 150
const X_RANGE: [number, number] = [-2.2, 2.2]
const Y_RANGE: [number, number] = [-1.2, 1.2]
const PARABOLA_X = Array.from({ length: 81 }, (_, i) => -2 + (4 * i) / 80)

/** The static parts of the parameter plane: the stationarity triangle and the real/complex boundary inside it. */
const REGION: SeriesSpec[] = [
  { name: 'stationarity triangle', type: 'line', x: [-2, 0, 2, -2], y: [-1, 1, -1, -1], slot: 0 },
  {
    name: 'φ₁² + 4φ₂ = 0 (complex roots below)',
    type: 'line',
    x: PARABOLA_X,
    y: PARABOLA_X.map((a) => -(a * a) / 4),
    slot: 1,
    dashed: true,
  },
]

/** Drag (φ₁, φ₂) around the AR(2) parameter plane and watch the autocorrelation function and a sample path change. */
export function Ar2Triangle() {
  const state = useFigureState({
    phi1: float(1, { min: X_RANGE[0], max: X_RANGE[1], step: 0.01, label: 'φ₁' }),
    phi2: float(-0.5, { min: Y_RANGE[0], max: Y_RANGE[1], step: 0.01, label: 'φ₂' }),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const p1 = state.phi1
  const p2 = state.phi2
  const roots = quadraticRoots(-p1, -p2)
  const moduli = roots.map(([re, im]) => Math.hypot(re, im))
  const stationary = moduli.every((m) => m > 1)
  const complex = roots.some(([, im]) => im !== 0)
  // Complex roots r·e^{±iω} give a damped cosine in the ACF with period 2π/ω.
  const period = complex ? (2 * Math.PI) / Math.abs(Math.atan2(roots[0][1], roots[0][0])) : null

  const seedValue = state.seed
  const r = useMemo(() => {
    const phi = [p1, p2]
    const x = simulateArma(phi, [], N, seedValue)
    const acf = minRootModulus(-p1, -p2) > 1 ? theoreticalAcf(phi, [], H).slice(1) : null
    const plane: SeriesSpec[] = [...REGION, { name: '(φ₁, φ₂)', type: 'scatter', x: [p1], y: [p2], emphasis: true }]
    const path = [{ name: 'x_t', x: x.map((_, t) => t + 1), y: x, slot: 0 }] as const
    const acfSeries: SeriesSpec[] = acf ? [{ name: 'ρ(h)', type: 'bar', x: LAGS, y: acf, slot: 0 }] : []
    return { plane, path, acfSeries }
  }, [p1, p2, seedValue])

  const rootText = roots
    .map(([re, im]) => (im === 0 ? formatNumber(re) : `${formatNumber(re)} ± ${formatNumber(Math.abs(im))}i`))
    .filter((s, i, all) => all.indexOf(s) === i)
    .join(', ')

  const xAxis = useAxis({ label: 'φ₁', range: X_RANGE })
  const yAxis = useAxis({ label: 'φ₂', range: Y_RANGE })
  const xAxis2 = useAxis({ label: 'lag h', hold: 'union' })
  const yAxis2 = useAxis({ label: 'ρ(h)', range: [-1, 1] })
  const xAxis3 = useAxis({ label: 't', hold: 'union' })
  const yAxis3 = useAxis({ label: 'x_t', hold: 'union' })
  return (
    <Figure
      title="The AR(2) parameter plane"
      state={state}
      caption="Drag the point, or use the sliders, to set φ₁ and φ₂ in x_t = φ₁x_{t−1} + φ₂x_{t−2} + ε_t. Inside the triangle both roots of 1 − φ₁z − φ₂z² lie outside the unit circle and the process is stationary. Above the dashed parabola the roots are real and the ACF decays as a mix of two exponentials. Below it the roots are complex and the ACF is a damped cosine, so the path shows pseudo-cycles. Outside the triangle the path explodes and no ACF exists."

      readouts={
        <>
          <Readout label="roots of 1 − φ₁z − φ₂z²" value={rootText || 'none'} />
          <Readout label="smallest root modulus" value={moduli.length ? formatNumber(Math.min(...moduli)) : '∞'} />
          <Readout label="stationary" value={stationary ? 'yes' : 'no'} />
          <Readout label="ACF period 2π/ω" value={period && stationary ? formatNumber(period) : '—'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(r.plane)}
          <Handle
            kind="point"
            at={[p1, p2]}
            label="(φ₁, φ₂)"
            onDrag={([a, b]) => {
              state.set('phi1', a)
              state.set('phi2', b)
            }}
          />
        </Plot>
        <div className="space-y-4">
          <Plot x={xAxis2} y={yAxis2} height={140}>
            {seriesLayers(r.acfSeries)}
          </Plot>
          <Plot x={xAxis3} y={yAxis3} height={140}>
            <Curve {...r.path[0]} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
