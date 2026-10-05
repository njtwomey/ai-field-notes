import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const GRID = toFlat(linspace(-4, 4, 161))
const MC_AT = toFlat(linspace(-3.5, 3.5, 15))
const DRAWS = 400
/** Fixed standard normals, so the Monte Carlo dots move smoothly as S changes. */
const NORMALS = (() => {
  const g = stream(5)
  return MC_AT.map(() => Array.from({ length: DRAWS }, () => normal(g)))
})()
const X_RANGE: [number, number] = [-4, 4]
const Y_RANGE: [number | undefined, number | undefined] = [0, 1.05]

/** ψ₁ for one latent dimension: the squared exponential kernel averaged over x ~ N(μ, S), against μ − z. */
export function Psi1Curve() {
  const state = useFigureState({
    logAlpha: float(0, {
      min: -2,
      max: 1,
      step: 0.01,
      label: 'inverse squared length-scale α',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    variance: float(0.5, { min: 0, max: 3, step: 0.01, label: 'variational variance S' }),
  })
  const alpha = 10 ** state.logAlpha
  const S = state.variance

  const series = useMemo(() => {
    const kernel = GRID.map((d) => Math.exp(-0.5 * alpha * d * d))
    const psi = GRID.map((d) => Math.exp((-0.5 * alpha * d * d) / (alpha * S + 1)) / Math.sqrt(alpha * S + 1))
    const mc = MC_AT.map(
      (mu, i) => NORMALS[i].reduce((s, e) => s + Math.exp(-0.5 * alpha * (mu + Math.sqrt(S) * e) ** 2), 0) / DRAWS,
    )
    return [
      { name: 'kernel k(μ, z), S = 0', x: GRID, y: kernel, slot: 0, dashed: true },
      { name: 'ψ₁ = E[k(x, z)], x ~ N(μ, S)', x: GRID, y: psi, slot: 1 },
      { name: `Monte Carlo, ${DRAWS} draws`, x: MC_AT, y: mc, slot: 2 },
    ] as const
  }, [alpha, S])

  const xAxis = useAxis({ label: 'μ − z', range: X_RANGE })
  const yAxis = useAxis({ label: 'covariance', range: Y_RANGE })
  return (
    <Figure
      title="The kernel averaged over an uncertain input"
      state={state}
      caption="One latent dimension, σ_f = 1, inducing input z at 0. The dashed curve is the kernel between a fixed input μ and z. The solid curve is ψ₁, the kernel averaged over x ~ N(μ, S), in closed form; the dots are Monte Carlo averages that check it. Uncertainty in x lowers the peak by √(αS + 1) and widens the curve by the same factor. As α → 0 both curves flatten to 1 whatever S is: the dimension no longer affects the kernel, and the bound is best served by setting q(x) to the prior."

      readouts={
        <>
          <Readout label="peak ψ₁ = 1/√(αS + 1)" value={formatNumber(1 / Math.sqrt(alpha * S + 1))} />
          <Readout
            label="effective length-scale √((αS + 1)/α)"
            value={formatNumber(Math.sqrt((alpha * S + 1) / alpha))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Points {...series[2]} />
      </Plot>
    </Figure>
  )
}
