import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { minRootModulus, pacfFromAcf, sampleAcf, simulateArma, theoreticalAcf } from '../_shared/arma'

const H = 20
const LAGS = Array.from({ length: H }, (_, i) => i + 1)

/** Sliders set an ARMA(2, 2) model; the figure shows a simulated path and its sample and theoretical ACF and PACF. */
export function ArmaExplorer() {
  const phi1 = useParam(0.6, { min: -1.9, max: 1.9, step: 0.05 })
  const phi2 = useParam(0, { min: -0.95, max: 0.95, step: 0.05 })
  const theta1 = useParam(0, { min: -1.5, max: 1.5, step: 0.05 })
  const theta2 = useParam(0, { min: -0.95, max: 0.95, step: 0.05 })
  const n = useParam(200, { min: 50, max: 1000, step: 50 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const arModulus = minRootModulus(-phi1.value, -phi2.value)
  const maModulus = minRootModulus(theta1.value, theta2.value)
  const stationary = arModulus > 1
  const invertible = maModulus > 1

  const r = useMemo(() => {
    const phi = [phi1.value, phi2.value]
    const theta = [theta1.value, theta2.value]
    const x = simulateArma(phi, theta, n.value, seed.value)
    const acf = sampleAcf(x, H)
    const pacf = pacfFromAcf(acf)
    const trueAcf = stationary ? theoreticalAcf(phi, theta, H) : null
    return { x, acf, pacf, trueAcf, truePacf: trueAcf ? pacfFromAcf(trueAcf) : null }
  }, [phi1.value, phi2.value, theta1.value, theta2.value, n.value, seed.value, stationary])

  const band = 1.96 / Math.sqrt(n.value)
  const charts = useMemo(() => {
    const bands: XYSeries[] = [
      { name: '±1.96/√n', type: 'line', x: [0.5, H + 0.5], y: [band, band], muted: true, dashed: true },
      { name: '−1.96/√n', type: 'line', x: [0.5, H + 0.5], y: [-band, -band], muted: true, dashed: true },
    ]
    const correlogram = (sample: number[], truth: number[] | null): XYSeries[] => [
      { name: 'sample', type: 'bar', x: LAGS, y: sample, slot: 0 },
      ...(truth ? [{ name: 'theoretical', type: 'scatter' as const, x: LAGS, y: truth, emphasis: true }] : []),
      ...bands,
    ]
    return {
      path: [{ name: 'x_t', type: 'line', x: r.x.map((_, t) => t + 1), y: r.x, slot: 0 }] as XYSeries[],
      acf: correlogram(r.acf.slice(1), r.trueAcf ? r.trueAcf.slice(1) : null),
      pacf: correlogram(r.pacf, r.truePacf),
    }
  }, [r, band])

  return (
    <Interactive
      title="ARMA(2, 2) explorer"
      caption="The sliders set x_t = φ₁x_{t−1} + φ₂x_{t−2} + ε_t + θ₁ε_{t−1} + θ₂ε_{t−2} with unit-variance Gaussian noise. Bars are the sample ACF and PACF of the simulated path; dots are the model's theoretical values; dashed lines are the ±1.96/√n band for white noise. With θ₁ = θ₂ = 0 (pure AR) the PACF cuts off after the last non-zero φ. With φ₁ = φ₂ = 0 (pure MA) the ACF cuts off after the last non-zero θ. A mixed model tails off in both. Outside the stationarity region the path explodes and the theoretical values are undefined."
      controls={
        <>
          <ParamSlider label="φ₁" param={phi1} />
          <ParamSlider label="φ₂" param={phi2} />
          <ParamSlider label="θ₁" param={theta1} />
          <ParamSlider label="θ₂" param={theta2} />
          <ParamSlider label="length n" param={n} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
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
        <XYChart series={charts.path} xLabel="t" yLabel="x_t" height={200} />
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <XYChart series={charts.acf} xLabel="lag h" yLabel="ACF" yRange={[-1, 1]} height={220} />
          <XYChart series={charts.pacf} xLabel="lag h" yLabel="PACF" yRange={[-1, 1]} height={220} />
        </div>
      </div>
    </Interactive>
  )
}
