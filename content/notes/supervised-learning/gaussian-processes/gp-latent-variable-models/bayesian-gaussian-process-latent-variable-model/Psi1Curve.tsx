import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'

const GRID = linspace(-4, 4, 161)
const MC_AT = linspace(-3.5, 3.5, 15)
const DRAWS = 400
/** Fixed standard normals, so the Monte Carlo dots move smoothly as S changes. */
const NORMALS = (() => {
  const g = rng(5)
  return MC_AT.map(() => Array.from({ length: DRAWS }, () => g.normal()))
})()
const X_RANGE: [number, number] = [-4, 4]
const Y_RANGE: [number | undefined, number | undefined] = [0, 1.05]

/** ψ₁ for one latent dimension: the squared exponential kernel averaged over x ~ N(μ, S), against μ − z. */
export function Psi1Curve() {
  const logAlpha = useParam(0, { min: -2, max: 1, step: 0.01 })
  const variance = useParam(0.5, { min: 0, max: 3, step: 0.01 })
  const alpha = 10 ** logAlpha.value
  const S = variance.value

  const series = useMemo((): XYSeries[] => {
    const kernel = GRID.map((d) => Math.exp(-0.5 * alpha * d * d))
    const psi = GRID.map((d) => Math.exp((-0.5 * alpha * d * d) / (alpha * S + 1)) / Math.sqrt(alpha * S + 1))
    const mc = MC_AT.map(
      (mu, i) => NORMALS[i].reduce((s, e) => s + Math.exp(-0.5 * alpha * (mu + Math.sqrt(S) * e) ** 2), 0) / DRAWS,
    )
    return [
      { name: 'kernel k(μ, z), S = 0', type: 'line', x: GRID, y: kernel, slot: 0, dashed: true },
      { name: 'ψ₁ = E[k(x, z)], x ~ N(μ, S)', type: 'line', x: GRID, y: psi, slot: 1 },
      { name: `Monte Carlo, ${DRAWS} draws`, type: 'scatter', x: MC_AT, y: mc, slot: 2 },
    ]
  }, [alpha, S])

  return (
    <Interactive
      title="The kernel averaged over an uncertain input"
      caption="One latent dimension, σ_f = 1, inducing input z at 0. The dashed curve is the kernel between a fixed input μ and z. The solid curve is ψ₁, the kernel averaged over x ~ N(μ, S), in closed form; the dots are Monte Carlo averages that check it. Uncertainty in x lowers the peak by √(αS + 1) and widens the curve by the same factor. As α → 0 both curves flatten to 1 whatever S is: the dimension no longer affects the kernel, and the bound is best served by setting q(x) to the prior."
      controls={
        <>
          <ParamSlider label="inverse squared length-scale α" param={logAlpha} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="variational variance S" param={variance} />
        </>
      }
      readout={
        <>
          <Readout label="peak ψ₁ = 1/√(αS + 1)" value={formatNumber(1 / Math.sqrt(alpha * S + 1))} />
          <Readout
            label="effective length-scale √((αS + 1)/α)"
            value={formatNumber(Math.sqrt((alpha * S + 1) / alpha))}
          />
        </>
      }
    >
      <XYChart series={series} xLabel="μ − z" yLabel="covariance" xRange={X_RANGE} yRange={Y_RANGE} height={340} />
    </Interactive>
  )
}
