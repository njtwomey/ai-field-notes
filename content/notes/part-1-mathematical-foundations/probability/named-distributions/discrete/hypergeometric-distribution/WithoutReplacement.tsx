import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { Binomial, Hypergeometric } from 'aifn-compute/probability/distributions'

/** Drawing without replacement (hypergeometric) against with replacement (binomial) as the population grows. */
export function WithoutReplacement() {
  const state = useFigureState({
    n: int(10, { min: 1, max: 40, suggestions: [5, 10, 20, 40], label: 'draws n' }),
    share: slider(0.05, 0.95, 0.4, { step: 0.05, label: 'success fraction K/N' }),
    population: int(20, {
      min: 1,
      max: 2000,
      scale: 'log10',
      suggestions: [10, 20, 50, 200, 2000],
      label: 'population N',
    }),
  })
  const { n, share } = state
  // The population holds at least the n items drawn.
  const N = Math.max(n, state.population)
  const K = Math.round(share * N)
  const p = K / N

  const result = useMemo(() => {
    const ks = Array.from({ length: n + 1 }, (_, k) => k)
    const hyperLaw = Hypergeometric(N, K, n)
    const binomLaw = Binomial(n, p)
    const hyper = ks.map((k) => hyperLaw.prob(k))
    const binom = ks.map((k) => binomLaw.prob(k))
    const tv = 0.5 * ks.reduce((s, k) => s + Math.abs(hyper[k] - binom[k]), 0)
    const series = [
      { name: 'without replacement (hypergeometric)', x: ks, y: hyper },
      { name: 'with replacement (binomial)', x: ks, y: binom, slot: 1, dashed: true },
    ] as const
    return { series, tv, variance: hyperLaw.variance() }
  }, [N, K, n, p])

  const fpc = N === 1 ? 0 : (N - n) / (N - 1)
  const xAxis = useAxis({ label: 'successes drawn k', hold: 'union' })
  const yAxis = useAxis({ label: 'P(X = k)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Sampling without replacement"
      state={state}
      caption="Draw n items from a population of N, a fraction K/N of which are successes. Without replacement (points) the count is hypergeometric. With replacement (dashed) it is binomial with p = K/N. Both have mean np. Without replacement the variance is smaller by the factor (N − n)/(N − 1). Grow N and the two coincide; shrink N towards n and the hypergeometric narrows until, at N = n, the count is always K."
      readouts={
        <>
          <Readout label="N" value={N} />
          <Readout label="K" value={K} />
          <Readout label="(N − n)/(N − 1)" value={formatNumber(fpc)} />
          <Readout label="variance, hypergeometric" value={formatNumber(result.variance)} />
          <Readout label="variance, binomial" value={formatNumber(n * p * (1 - p))} />
          <Readout label="total-variation distance" value={formatNumber(result.tv)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Points {...result.series[0]} />
        <Curve {...result.series[1]} />
      </Plot>
    </Figure>
  )
}
