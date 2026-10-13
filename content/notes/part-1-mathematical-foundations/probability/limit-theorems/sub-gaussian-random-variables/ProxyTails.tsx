import { useMemo } from 'react'
import { concentrationStudy, tailBounds, type SummandLaw } from 'aifn-methods/theory/concentration'
import { stream } from 'aifn-compute/foundation/random'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const FLOOR = 1e-5
const TRIALS = 20_000
const LAWS = [
  { value: 'bernoulli', label: 'Bernoulli(p)' },
  { value: 'uniform', label: 'uniform on [0, 1]' },
  { value: 'arcsine', label: 'arcsine on [0, 1]' },
] as const

/**
 * The tail of the mean of n variables in [0, 1], simulated, beside the sub-Gaussian (Hoeffding) bound, whose variance
 * proxy 1/4 is the same for every law on [0, 1], and Bernstein's bound, which uses the law's variance.
 */
export function ProxyTails() {
  const state = useFigureState({
    law: choice(LAWS, 'bernoulli', { label: 'law of each variable' }),
    p: slider(0.02, 0.98, 0.1, { step: 0.01, label: 'Bernoulli p', when: (v) => v.law === 'bernoulli' }),
    n: int(100, { min: 1, max: 2000, suggestions: [10, 100, 500], label: 'variables averaged n' }),
    t: slider(0.01, 0.4, 0.1, { step: 0.005, label: 'deviation t' }),
  })
  const law = state.law as SummandLaw
  const study = useComputed(
    () => concentrationStudy(stream('sub-gaussian/proxy'), { law, p: state.p, n: state.n, trials: TRIALS }),
    [law, state.p, state.n],
    { mode: 'release' },
  )
  const c = study.value
  const at = tailBounds([state.t], state.n, c.mean, c.variance)
  // The share of simulated means at least t from μ, read at the handle (the study's curve is on a fixed grid).
  const simulatedAt = useMemo(() => {
    let k = 0
    for (const v of c.means) if (Math.abs(v - c.mean) >= state.t - 1e-12) k++
    return k / c.means.length
  }, [c, state.t])

  const tAxis = useAxis({ label: 'deviation t', range: [0, 0.5] })
  const pAxis = useAxis({ label: 'P(|X̄ − μ| ≥ t)', log: true, range: [FLOOR, 1] })
  const floor = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(FLOOR, v))
  return (
    <Figure
      title="One variance proxy for every law on [0, 1]"
      state={state}
      caption="The probability that the mean of n independent variables in [0, 1] misses its mean μ by at least t, on a log scale. The simulated tail (ink, 20,000 means) sits under the sub-Gaussian bound 2e^(−2nt²), whose variance proxy 1/4 is the same for every law on [0, 1], and under Bernstein's bound, which uses the law's variance σ². A fair coin (p = 0.5) has variance 1/4 and the sub-Gaussian bound tracks its tail. A Bernoulli law with p near 0 or 1, the uniform law (σ² = 1/12) and the arcsine law (σ² = 1/8) have smaller variances; their tails fall faster than the proxy allows, and Bernstein's bound pulls ahead for small t. The simulated tail stops where no simulated mean deviated that far; bounds are floored at 10⁻⁵. Drag the deviation t on the chart."
      readouts={
        <>
          <Readout label="variance σ²" value={fmt(c.variance)} />
          <Readout label="sub-Gaussian proxy" value="0.25" />
          <Readout label={`simulated at t = ${fmt(state.t)}`} value={fmt(simulatedAt)} />
          <Readout label="sub-Gaussian bound" value={fmt(at.hoeffding[0])} />
          <Readout label="Bernstein bound" value={fmt(at.bernstein[0])} />
        </>
      }
    >
      <Plot x={tAxis} y={pAxis} height={340}>
        <Curve
          name="simulated"
          x={c.t}
          y={Array.from(c.empirical, (v) => (v > 0 ? v : NaN))}
          emphasis
          showPoints
          stale={study.stale}
        />
        <Curve name="sub-Gaussian, proxy 1/4 (Hoeffding)" slot={0} x={c.t} y={floor(c.hoeffding)} />
        <Curve name="Bernstein, variance σ²" slot={1} x={c.t} y={floor(c.bernstein)} dashed />
        <Handle {...state.handle('t', { label: 't' })} />
      </Plot>
    </Figure>
  )
}
