import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { maxValueEntropySearch } from 'aifn-methods/optimisation/bayesian'
import { Normal, TruncatedNormal } from 'aifn-compute/probability/distributions'
import { linspace, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const F_AXIS = toFlat(linspace(-3, 3, 481))
const GAMMA_AXIS = toFlat(linspace(-3, 4, 141))
/** MES for one sample of the minimum, as a function of γ = (μ − f*)/σ: μ = γ, σ = 1, f* = 0. */
const MES_CURVE = toFlat(maxValueEntropySearch(tensor(GAMMA_AXIS), 1, { samples: [0] }) as Tensor)

/**
 * What one sample f* of the minimum value tells about a candidate's value: conditioning on f(λ) ≥ f* truncates its
 * Gaussian posterior, and the drop in entropy is the max-value entropy search acquisition for that sample.
 */
export function MaxValueInformation() {
  const state = useFigureState({
    mu: slider(-2, 2, 0, { step: 0.05, label: 'posterior mean μ' }),
    sigma: slider(0.1, 1.5, 0.5, { step: 0.05, label: 'posterior sd σ' }),
    ystar: slider(-3, 2, -0.5, { step: 0.05, label: 'sample of the minimum f*', onChart: true }),
  })
  const { mu, sigma, ystar } = state

  const result = useMemo(() => {
    const before = Normal(mu, sigma)
    const after = TruncatedNormal(mu, sigma, ystar, Infinity)
    const p = toFlat(before.prob(tensor(F_AXIS)) as Tensor)
    const q = toFlat(after.prob(tensor(F_AXIS)) as Tensor)
    const hBefore = Number(before.entropy())
    const hAfter = Number(after.entropy())
    const mes = maxValueEntropySearch(mu, sigma, { samples: [ystar] })
    return { p, q, hBefore, hAfter, mes }
  }, [mu, sigma, ystar])

  const gamma = (mu - ystar) / sigma
  const fAxis = useAxis({ label: 'value of f(λ)', range: [-3, 3] })
  const dAxis = useAxis({ label: 'density', hold: 'union' })
  const gAxis = useAxis({ label: 'γ = (μ − f*)/σ', range: [-3, 4] })
  const iAxis = useAxis({ label: 'information about min f (nats)', range: [0, 2] })
  return (
    <Figure
      title="What one sample of the minimum teaches about a candidate"
      state={state}
      caption="Left: the posterior density of a candidate's value f(λ) ~ N(μ, σ²), and the same density conditioned on f(λ) ≥ f*, where f* is one sample of the minimum value of f. Drag the vertical line f*. The drop in entropy between the two is the max-value entropy search score for that sample. Right: that score as a function of γ = (μ − f*)/σ alone. It falls steadily with γ, so with a single sample, maximising it is the same as maximising the probability Φ(−γ) that the candidate's value falls below f*."
      readouts={
        <>
          <Readout label="γ" value={formatNumber(gamma)} />
          <Readout label="entropy before (nats)" value={formatNumber(result.hBefore)} />
          <Readout label="entropy after" value={formatNumber(result.hAfter)} />
          <Readout label="drop" value={formatNumber(result.hBefore - result.hAfter)} />
          <Readout label="MES score" value={formatNumber(result.mes)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={fAxis} y={dAxis} title="one candidate">
          <Area name="posterior p(f)" x={F_AXIS} y={result.p} slot={0} opacity={0.15} />
          <Curve name="given f ≥ f*" x={F_AXIS} y={result.q} slot={1} />
          <Handle {...state.handle('ystar', { label: 'f*' })} />
        </Plot>
        <Plot x={gAxis} y={iAxis} title="score against γ">
          <Curve name="γφ(γ)/2Φ(γ) − log Φ(γ)" x={GAMMA_AXIS} y={MES_CURVE} slot={1} />
          <Points name="this candidate" x={[gamma]} y={[result.mes]} emphasis size={9} />
        </Plot>
      </Plots>
    </Figure>
  )
}
