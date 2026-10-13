import { useMemo } from 'react'
import {
  Area,
  Contours,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { expectedImprovement, probabilityOfImprovement } from 'aifn-methods/optimisation/bayesian'
import { Normal } from 'aifn-compute/probability/distributions'
import { linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'

const F_AXIS = toFlat(linspace(-2.5, 2.5, 401))
const MU_AXIS = toFlat(linspace(-0.5, 2, 101))
const SD_AXIS = toFlat(linspace(0.05, 1.5, 59))
const LEVELS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]

/**
 * Expected improvement for one candidate. Left: the posterior density of f(λ), the region below the incumbent (its mass
 * is PI) and the integrand (f* − ξ − f) p(f) (its area is EI). Right: EI over the candidate's mean and standard deviation.
 */
export function ImprovementGeometry() {
  const state = useFigureState({
    mu: slider(-0.5, 2, 0.4, { step: 0.05, label: 'posterior mean μ' }),
    sigma: slider(0.05, 1.5, 0.5, { step: 0.05, label: 'posterior sd σ' }),
    best: slider(-1, 1.5, 0, { step: 0.05, label: 'incumbent f*', onChart: true }),
    xi: float(0, { label: 'margin ξ', ge: 0, le: 1, suggestions: [0, 0.05, 0.1, 0.25] }),
  })
  const { mu, sigma, best, xi } = state

  const density = useMemo(() => {
    const p = toFlat(Normal(mu, sigma).prob(tensor(F_AXIS)))
    const target = best - xi
    const below = F_AXIS.map((f, i) => (f <= target ? i : -1)).filter((i) => i >= 0)
    return {
      p,
      shadeX: below.map((i) => F_AXIS[i]),
      shadeY: below.map((i) => p[i]),
      integrand: below.map((i) => (target - F_AXIS[i]) * p[i]),
    }
  }, [mu, sigma, best, xi])

  const surface = useMemo(() => {
    const m = SD_AXIS.flatMap(() => MU_AXIS)
    const s = SD_AXIS.flatMap((v) => MU_AXIS.map(() => v))
    const ei = toFlat(expectedImprovement(tensor(m), tensor(s), best, { xi }))
    return SD_AXIS.map((_, i) => ei.slice(i * MU_AXIS.length, (i + 1) * MU_AXIS.length))
  }, [best, xi])

  const ei = expectedImprovement(mu, sigma, best, { xi })
  const pi = probabilityOfImprovement(mu, sigma, best, { xi })
  const exploit = (best - xi - mu) * pi
  const explore = ei - exploit

  const fAxis = useAxis({ label: 'value of f(λ)', range: [-2.5, 2.5] })
  const pAxis = useAxis({ label: 'density / integrand', hold: 'union' })
  const muAxis = useAxis({ label: 'posterior mean μ', range: [-0.5, 2] })
  const sdAxis = useAxis({ label: 'posterior sd σ', range: [0.05, 1.5] })
  return (
    <Figure
      title="What expected improvement integrates"
      state={state}
      caption="Left: the posterior density of the candidate's value f(λ) ~ N(μ, σ²). The shaded mass below the incumbent (less the margin ξ) is the probability of improvement; the dashed curve is the improvement times the density, and its area is the expected improvement. Drag the vertical line f* to move the incumbent. Right: expected improvement over the candidate's mean and standard deviation, with contours of equal EI; the dot is the candidate. The contours bend: along each one a worse mean is paid for by a wider spread, which is the exploration–exploitation trade-off in a single picture."
      readouts={
        <>
          <Readout label="z = (f* − ξ − μ)/σ" value={formatNumber((best - xi - mu) / sigma)} />
          <Readout label="PI = Φ(z)" value={formatNumber(pi)} />
          <Readout label="exploitation (f* − ξ − μ)Φ(z)" value={formatNumber(exploit)} />
          <Readout label="exploration σφ(z)" value={formatNumber(explore)} />
          <Readout label="EI" value={formatNumber(ei)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={fAxis} y={pAxis} title="one candidate">
          <Area name="posterior density p(f)" x={F_AXIS} y={density.p} muted opacity={0.12} />
          {density.shadeX.length > 1 && (
            <Area name="mass below f* − ξ (PI)" x={density.shadeX} y={density.shadeY} slot={0} opacity={0.35} />
          )}
          {density.shadeX.length > 1 && (
            <Curve name="(f* − ξ − f) p(f), area EI" x={density.shadeX} y={density.integrand} slot={1} dashed />
          )}
          <Handle {...state.handle('best', { label: 'f*' })} />
        </Plot>
        <Plot x={muAxis} y={sdAxis} title="EI over (μ, σ)">
          <Raster x={MU_AXIS} y={SD_AXIS} z={surface} valueLabel="expected improvement" />
          <Contours x={MU_AXIS} y={SD_AXIS} z={surface} levels={LEVELS} />
          <Curve name="μ = f*" x={[best, best]} y={[0.05, 1.5]} muted dashed />
          <Points name="the candidate" x={[mu]} y={[sigma]} emphasis size={9} />
        </Plot>
      </Plots>
    </Figure>
  )
}
