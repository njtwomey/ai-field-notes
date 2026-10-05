import { useMemo } from 'react'
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
  useFigureState,
} from 'aifn-render'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'

const M_MAX = 50
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]

/** Probability that a test which ignores within-unit correlation rejects a true null: |N(0, D)| ≥ z_{1−α/2}. */
function falsePositiveRate(m: number, rho: number, alpha: number): number {
  const deff = 1 + (m - 1) * rho
  return 2 * (1 - normalCdf(normalQuantile(1 - alpha / 2) / Math.sqrt(deff)))
}

/**
 * A/A tests randomised by user but analysed as if every page view were independent. With m views per user and
 * intra-user correlation ρ, the true variance of the mean is 1 + (m − 1)ρ times the naive one, so the naive z-statistic
 * is N(0, D) under the null rather than N(0, 1).
 */
export function UnitMismatch() {
  const state = useFigureState({
    m: int(10, { min: 1, max: M_MAX, step: 1, label: 'page views per user, m' }),
    rho: slider(0, 0.5, 0.1, { step: 0.01, label: 'intra-user correlation ρ' }),
    alpha: choice(ALPHAS, '0.05', { label: 'α' }),
  })

  const series = useMemo(() => {
    const a = Number(state.alpha)
    const ms = Array.from({ length: M_MAX }, (_, i) => i + 1)
    return [
      { name: 'false-positive rate', x: ms, y: ms.map((k) => falsePositiveRate(k, state.rho, a)), slot: 0 },
      { name: 'α', x: [1, M_MAX], y: [a, a], slot: 1, dashed: true },
    ] as const
  }, [state.rho, state.alpha])

  const deff = 1 + (state.m - 1) * state.rho

  const xAxis = useAxis({ label: 'page views per user, m', range: [1, M_MAX] })
  const yAxis = useAxis({ label: 'P(significant A/A test)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="A/A false positives when the analysis unit is finer than the randomisation unit"
      state={state}
      caption="Users are randomised, but the test treats each of a user's m page views as an independent observation. Views of one user are correlated with intra-user correlation ρ, so the naive standard error is too small by a factor √(1 + (m − 1)ρ). The curve is the resulting probability that an A/A test is declared significant. Drag the line labelled m, or use its slider, to change the number of views per user."

      readouts={
        <>
          <Readout label="design effect 1 + (m − 1)ρ" value={formatNumber(deff)} />
          <Readout label="naive se ÷ true se" value={formatNumber(1 / Math.sqrt(deff))} />
          <Readout
            label="A/A false-positive rate"
            value={formatNumber(falsePositiveRate(state.m, state.rho, Number(state.alpha)))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle kind="x" at={state.m} label="m" onDrag={(x) => state.set('m', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
