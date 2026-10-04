import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { minRootModulus, pacfFromAcf, sampleAcf, simulateArma, theoreticalAcf } from '../_shared/arma'

const H = 20
const LAGS = Array.from({ length: H }, (_, i) => i + 1)

/** Sliders set an ARMA(2, 2) model; the figure shows a simulated path and its sample and theoretical ACF and PACF. */
export function ArmaExplorer() {
  const state = useFigureState({
    phi1: float(0.6, { min: -1.9, max: 1.9, step: 0.05, label: 'φ₁' }),
    phi2: float(0, { min: -0.95, max: 0.95, step: 0.05, label: 'φ₂' }),
    theta1: float(0, { min: -1.5, max: 1.5, step: 0.05, label: 'θ₁' }),
    theta2: float(0, { min: -0.95, max: 0.95, step: 0.05, label: 'θ₂' }),
    n: int(200, { min: 50, max: 1000, step: 50, label: 'length n', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const arModulus = minRootModulus(-state.phi1, -state.phi2)
  const maModulus = minRootModulus(state.theta1, state.theta2)
  const stationary = arModulus > 1
  const invertible = maModulus > 1

  const r = useMemo(() => {
    const phi = [state.phi1, state.phi2]
    const theta = [state.theta1, state.theta2]
    const x = simulateArma(phi, theta, state.n, state.seed)
    const acf = sampleAcf(x, H)
    const pacf = pacfFromAcf(acf)
    const trueAcf = stationary ? theoreticalAcf(phi, theta, H) : null
    return { x, acf, pacf, trueAcf, truePacf: trueAcf ? pacfFromAcf(trueAcf) : null }
  }, [state.phi1, state.phi2, state.theta1, state.theta2, state.n, state.seed, stationary])

  const band = 1.96 / Math.sqrt(state.n)
  const charts = useMemo(() => {
    const bands: SeriesSpec[] = [
      { name: '±1.96/√n', type: 'line', x: [0.5, H + 0.5], y: [band, band], muted: true, dashed: true },
      { name: '−1.96/√n', type: 'line', x: [0.5, H + 0.5], y: [-band, -band], muted: true, dashed: true },
    ]
    const correlogram = (sample: number[], truth: number[] | null): SeriesSpec[] => [
      { name: 'sample', type: 'bar', x: LAGS, y: sample, slot: 0 },
      ...(truth ? [{ name: 'theoretical', type: 'scatter' as const, x: LAGS, y: truth, emphasis: true }] : []),
      ...bands,
    ]
    return {
      path: [{ name: 'x_t', type: 'line', x: r.x.map((_, t) => t + 1), y: r.x, slot: 0 }] as SeriesSpec[],
      acf: correlogram(r.acf.slice(1), r.trueAcf ? r.trueAcf.slice(1) : null),
      pacf: correlogram(r.pacf, r.truePacf),
    }
  }, [r, band])

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ label: 'x_t', hold: 'union' })
  const xAxis2 = useAxis({ label: 'lag h', hold: 'union' })
  const yAxis2 = useAxis({ label: 'ACF', range: [-1, 1] })
  const xAxis3 = useAxis({ label: 'lag h', hold: 'union' })
  const yAxis3 = useAxis({ label: 'PACF', range: [-1, 1] })
  return (
    <Figure
      title="ARMA(2, 2) explorer"
      state={state}
      caption="The sliders set x_t = φ₁x_{t−1} + φ₂x_{t−2} + ε_t + θ₁ε_{t−1} + θ₂ε_{t−2} with unit-variance Gaussian noise. Bars are the sample ACF and PACF of the simulated path; dots are the model's theoretical values; dashed lines are the ±1.96/√n band for white noise. With θ₁ = θ₂ = 0 (pure AR) the PACF cuts off after the last non-zero φ. With φ₁ = φ₂ = 0 (pure MA) the ACF cuts off after the last non-zero θ. A mixed model tails off in both. Outside the stationarity region the path explodes and the theoretical values are undefined."

      readouts={
        <>
          <Readout
            label="smallest AR root modulus"
            value={`${Number.isFinite(arModulus) ? formatNumber(arModulus) : '∞'} (${stationary ? 'stationary' : 'not stationary'})`}
          />
          <Readout
            label="smallest MA root modulus"
            value={`${Number.isFinite(maModulus) ? formatNumber(maModulus) : '∞'} (${invertible ? 'invertible' : 'not invertible'})`}
          />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={200}>
          {seriesLayers(charts.path)}
        </Plot>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={xAxis2} y={yAxis2} height={220}>
            {seriesLayers(charts.acf)}
          </Plot>
          <Plot x={xAxis3} y={yAxis3} height={220}>
            {seriesLayers(charts.pacf)}
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
