import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { expectedImprovement, multiPointExpectedImprovement } from 'aifn-methods/optimisation/bayesian'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'

/** The worked example: incumbent 0.20, candidate A ~ N(0.19, 0.02²), and a partner for it. */
const BEST = 0.2
const A = { mean: 0.19, sd: 0.02 }
const PARTNERS = {
  near: { mean: 0.19, sd: 0.02 },
  far: { mean: 0.2, sd: 0.03 },
} as const
type Partner = keyof typeof PARTNERS
/** Correlations along the curve, and Monte Carlo draws per point of it and for the readout. */
const RHOS = toFlat(linspace(-0.9, 1, 39))
const CURVE_DRAWS = 20000
const READOUT_DRAWS = 400000

const covariance = (rho: number, s1: number, s2: number) =>
  tensor([
    [s1 * s1, rho * s1 * s2],
    [rho * s1 * s2, s2 * s2],
  ])

/**
 * The multi-point expected improvement of a pair, by Monte Carlo, against the correlation of their outcomes, between
 * its bounds: the larger single-point EI below and the sum of the two above.
 */
export function MultiPointImprovement() {
  const state = useFigureState({
    partner: choice<Partner>(
      [
        { value: 'near', label: 'A′ ~ N(0.19, 0.02²), a near-copy of A' },
        { value: 'far', label: 'B ~ N(0.20, 0.03²)' },
      ],
      'near',
      { label: 'paired with A' },
    ),
    rho: slider(-0.9, 1, 0.9, { step: 0.05, label: 'correlation ρ of the two outcomes' }),
  })
  const p = PARTNERS[state.partner]
  const means = tensor([A.mean, p.mean])
  const eiA = expectedImprovement(A.mean, A.sd, BEST)
  const eiP = expectedImprovement(p.mean, p.sd, BEST)

  // Common random numbers along the curve, so it is smooth in ρ.
  const curve = useComputed(
    () =>
      Array.from(
        RHOS,
        (rho) =>
          multiPointExpectedImprovement(means, covariance(rho, A.sd, p.sd), BEST, {
            samples: CURVE_DRAWS,
            stream: stream('q-ei/curve'),
          }).value,
      ),
    [state.partner],
  )
  const here = useComputed(
    () =>
      multiPointExpectedImprovement(means, covariance(state.rho, A.sd, p.sd), BEST, {
        samples: READOUT_DRAWS,
        stream: stream('q-ei/readout'),
      }),
    [state.partner, state.rho],
  )

  const rhoAxis = useAxis({ label: 'correlation ρ', range: [-0.9, 1] })
  const eAxis = useAxis({ label: 'expected improvement', range: [0, 0.03] })
  return (
    <Figure
      title="A second point is worth more the less it is predictable from the first"
      state={state}
      caption="The multi-point expected improvement of A with a partner, estimated from joint draws of their two outcomes, against the correlation ρ of those outcomes. The incumbent is 0.20 and A ~ N(0.19, 0.02²). The dashed lines are the bounds: the larger of the two single-point EIs below, their sum above. For the near-copy A′ the pair is worth A's EI alone at ρ = 1, 0.0164 at ρ = 0.9 and 0.0220 at ρ = 0. For B, uncorrelated with A, it is 0.0214. Negative correlation, which a Gaussian process never gives two nearby points, would push the pair towards the upper bound."
      readouts={
        <>
          <Readout label="EI of A alone" value={formatNumber(eiA)} />
          <Readout label="EI of the partner alone" value={formatNumber(eiP)} />
          <Readout label="q-EI of the pair" value={formatNumber(here.value.value)} />
          <Readout label="its standard error" value={formatNumber(here.value.se)} />
          <Readout label="bounds" value={`${formatNumber(Math.max(eiA, eiP))} to ${formatNumber(eiA + eiP)}`} />
        </>
      }
    >
      <Plot x={rhoAxis} y={eAxis} height={340}>
        <Curve name="larger single EI" x={[-0.9, 1]} y={[Math.max(eiA, eiP), Math.max(eiA, eiP)]} muted dashed />
        <Curve name="sum of the two EIs" x={[-0.9, 1]} y={[eiA + eiP, eiA + eiP]} muted dashed />
        <Curve name="q-EI of the pair" x={RHOS} y={curve.value} slot={0} stale={curve.stale} />
        <Points name="at ρ" x={[state.rho]} y={[here.value.value]} emphasis size={10} />
      </Plot>
    </Figure>
  )
}
