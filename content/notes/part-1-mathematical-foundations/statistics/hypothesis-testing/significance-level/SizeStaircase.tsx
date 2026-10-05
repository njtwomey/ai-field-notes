import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { Binomial } from 'aifn-compute/probability/distributions'

/** P(S ≥ k) for S ~ Binomial(n, p). */
const binomialUpper = (k: number, n: number, p: number) => Binomial(n, p).survival(k - 1) as number

const N_MAX = 100

/** Attained size of the exact one-sided binomial test of a fair coin, H₁: p > 1/2, for every n up to N_MAX. */
export function SizeStaircase() {
  const state = useFigureState({
    alpha: float(0.05, { min: 0.01, max: 0.1, step: 0.005, label: 'nominal level α' }),
    n: int(20, { min: 5, max: N_MAX, step: 1, label: 'tosses n' }),
  })

  const result = useMemo(() => {
    const ns = Array.from({ length: N_MAX - 4 }, (_, i) => i + 5)
    const sizes = ns.map((m) => {
      let c = 0
      while (binomialUpper(c, m, 0.5) > state.alpha) c++
      return binomialUpper(c, m, 0.5)
    })
    const series = [
      { name: 'attained size', x: ns, y: sizes, slot: 0 },
      { name: 'nominal α', x: [5, N_MAX], y: [state.alpha, state.alpha], slot: 1, dashed: true },
      { name: 'selected n', x: [state.n], y: [sizes[state.n - 5]], emphasis: true },
    ] as const
    let c = 0
    while (binomialUpper(c, state.n, 0.5) > state.alpha) c++
    return { series, c, size: sizes[state.n - 5] }
  }, [state.alpha, state.n])

  const xAxis = useAxis({ label: 'tosses n', hold: 'union' })
  const yAxis = useAxis({ label: 'P(reject | H₀)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="The size of an exact test is a staircase"
      state={state}
      caption="An exact test of a fair coin rejects when the number of heads reaches a critical count c. Only whole counts are possible, so the probability of rejecting a true null, the attained size, jumps with n and stays at or below the nominal level α (dashed)."

      readouts={
        <>
          <Readout label="critical count c" value={result.c} />
          <Readout label="attained size" value={formatNumber(result.size)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...result.series[0]} />
        <Curve {...result.series[1]} />
        <Points {...result.series[2]} />
      </Plot>
    </Figure>
  )
}
