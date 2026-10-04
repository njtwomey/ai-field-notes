import { useMemo } from 'react'
import { Bars, Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { Binomial } from 'aifn/probability/distributions'

/** P(S = k) for S ~ Binomial(n, p). */
const binomialPmf = (k: number, n: number, p: number) => Binomial(n, p).prob(k) as number
/** P(S ≥ k) for S ~ Binomial(n, p). */
const binomialUpper = (k: number, n: number, p: number) => Binomial(n, p).survival(k - 1) as number

const P0 = 0.01

/**
 * The quality-control test of H₀: p = 1% against H₁: p > 1%. S ~ Binom(n, p) counts defects; the critical value c is
 * the smallest count whose upper tail under H₀ is at most α.
 */
export function CriticalRegion() {
  const state = useFigureState({
    n: int(200, { min: 20, max: 1500, step: 10, label: 'items inspected n' }),
    alpha: float(0.05, { min: 0.01, max: 0.2, step: 0.01, label: 'significance level α' }),
    p1: float(0.02, {
      min: 0.011,
      max: 0.06,
      step: 0.001,
      label: 'alternative defect rate',
      format: (v) => `${(100 * v).toFixed(1)}%`,
    }),
  })

  const result = useMemo(() => {
    let c = 0
    while (binomialUpper(c, state.n, P0) > state.alpha) c++
    const top = Math.ceil(state.n * state.p1 + 4 * Math.sqrt(state.n * state.p1 * (1 - state.p1)) + 4)
    const ks = Array.from({ length: top + 1 }, (_, k) => k)
    const kept = ks.filter((k) => k < c)
    const rejected = ks.filter((k) => k >= c)
    const series = [
      { name: 'H₀ pmf, do not reject', x: kept, y: kept.map((k) => binomialPmf(k, state.n, P0)), muted: true },
      { name: 'H₀ pmf, reject', x: rejected, y: rejected.map((k) => binomialPmf(k, state.n, P0)), slot: 0 },
      { name: 'H₁ pmf', x: ks, y: ks.map((k) => binomialPmf(k, state.n, state.p1)), slot: 1, dashed: true },
    ] as const
    return { c, series, size: binomialUpper(c, state.n, P0), power: binomialUpper(c, state.n, state.p1) }
  }, [state.n, state.alpha, state.p1])

  const xAxis = useAxis({ label: 'defects S', hold: 'union' })
  const yAxis = useAxis({ label: 'probability', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Critical region for a defect rate"
      state={state}
      caption="Grey and blue bars are the distribution of the defect count S when the defect rate is 1%. The blue bars, S ≥ c, form the critical region: its total probability is the size of the test, at most α. The dashed line is the distribution of S under an alternative defect rate. Its mass over the critical region is the power."

      readouts={
        <>
          <Readout label="critical value c" value={result.c} />
          <Readout label="size P₀(S ≥ c)" value={formatNumber(result.size)} />
          <Readout label="power P₁(S ≥ c)" value={formatNumber(result.power)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...result.series[0]} />
        <Bars {...result.series[1]} />
        <Curve {...result.series[2]} />
      </Plot>
    </Figure>
  )
}
