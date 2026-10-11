import { useMemo } from 'react'
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
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { log1p, sqrt, variance } from 'aifn-compute/foundation/tensor'
import { NegativeBinomial, Poisson } from 'aifn-compute/probability/distributions'

const DRAWS = 3000
/** Means from 0.25 to 128, evenly spaced on a log scale. */
const MEANS = Array.from({ length: 19 }, (_, i) => 0.25 * 2 ** (i / 2))

const FAMILIES = [
  { value: 'poisson', label: 'Poisson' },
  { value: 'negative-binomial', label: 'negative binomial' },
] as const

/** How the variance of counts, and of their square root and logarithm, changes with the mean. */
export function VarianceStabilising() {
  const state = useFigureState({
    family: choice(FAMILIES, 'poisson', { label: 'counts' }),
    r: slider(0.5, 20, 2, { step: 0.5, label: 'negative binomial r' }),
  })
  const { family, r } = state
  const view = useMemo(() => {
    const raw: number[] = []
    const root: number[] = []
    const log: number[] = []
    MEANS.forEach((mu, i) => {
      // The negative binomial with r successes and mean μ has p = r / (r + μ) and variance μ + μ²/r.
      const dist = family === 'poisson' ? Poisson(mu) : NegativeBinomial(r, r / (r + mu))
      const x = dist.sample(child(stream('variance-stabilising'), family, i), { shape: [DRAWS] })
      raw.push(variance(x) as number)
      root.push(variance(sqrt(x)) as number)
      log.push(variance(log1p(x)) as number)
    })
    return { raw, root, log }
  }, [family, r])
  const mx = useAxis({ label: 'mean μ', log: true })
  const vy = useAxis({ label: 'variance', log: true })
  const last = MEANS.length - 1
  return (
    <Figure
      title="Variance against mean, before and after a transform"
      state={state}
      caption={`For each mean μ from 0.25 to 128, ${DRAWS} counts are drawn and three variances are measured: of the counts, of their square roots and of log(1 + x). Both axes are logarithmic. Poisson counts have variance μ, a line of slope 1; their square roots have variance close to 1/4 once μ exceeds about 3, which is what variance-stabilising means. log(1 + x) over-corrects Poisson counts: its variance keeps falling, roughly as 1/μ. Negative binomial counts have variance μ + μ²/r, and for them the logarithm stabilises the variance at large μ while the square root does not.`}
      readouts={
        <>
          <Readout label={`variance of x at μ = ${MEANS[last]}`} value={formatNumber(view.raw[last])} />
          <Readout label="of √x" value={formatNumber(view.root[last])} />
          <Readout label="of log(1 + x)" value={formatNumber(view.log[last])} />
        </>
      }
    >
      <Plot x={mx} y={vy}>
        <Curve name="variance of x" x={MEANS} y={view.raw} slot={0} />
        <Points name="variance of x" x={MEANS} y={view.raw} slot={0} />
        <Curve name="variance of √x" x={MEANS} y={view.root} slot={1} />
        <Points name="variance of √x" x={MEANS} y={view.root} slot={1} />
        <Curve name="variance of log(1 + x)" x={MEANS} y={view.log} slot={2} />
        <Points name="variance of log(1 + x)" x={MEANS} y={view.log} slot={2} />
        <Curve name="1/4" x={[MEANS[0], MEANS[last]]} y={[0.25, 0.25]} muted dashed silent />
      </Plot>
    </Figure>
  )
}
