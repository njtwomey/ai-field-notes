import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { Binomial, Poisson } from 'aifn-compute/probability/distributions'

/** Binomial(n, λ/n) against Poisson(λ) as n grows, with the total-variation distance and Le Cam's bound. */
export function BinomialLimit() {
  const state = useFigureState({
    lambda: slider(0.5, 10, 3, { step: 0.5, label: 'rate λ' }),
    pieces: int(10, { min: 3, max: 1000, scale: 'log10', suggestions: [10, 30, 100, 1000], label: 'pieces n' }),
  })
  const lambda = state.lambda
  // Each piece holds an event with probability λ/n ≤ 1.
  const n = Math.max(Math.ceil(lambda), state.pieces)

  const result = useMemo(() => {
    const kMax = Math.max(12, Math.ceil(lambda + 5 * Math.sqrt(lambda)))
    const ks = Array.from({ length: kMax + 1 }, (_, k) => k)
    const poisson = Poisson(lambda)
    const binomial = Binomial(n, lambda / n)
    const pois = ks.map((k) => poisson.prob(k))
    const binom = ks.map((k) => binomial.prob(k))
    // Mass beyond kMax is negligible for the ranges plotted.
    const tv = 0.5 * ks.reduce((s, k) => s + Math.abs(pois[k] - binom[k]), 0)
    const series = [
      { name: 'Poisson(λ)', x: ks, y: pois, slot: 0 },
      { name: `Binomial(n, λ/n)`, x: ks, y: binom, slot: 1, dashed: true },
    ] as const
    return { series, tv }
  }, [lambda, n])

  const xAxis = useAxis({ label: 'k', hold: 'union' })
  const yAxis = useAxis({ label: 'P(X = k)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="The law of rare events"
      state={state}
      caption="Split an interval into n pieces, each holding an event with probability λ/n. As n grows, the binomial count (dashed) approaches the Poisson pmf (points). The distance between them is bounded by λ²/n."
      readouts={
        <>
          <Readout label="n" value={n} />
          <Readout label="p = λ/n" value={formatNumber(lambda / n)} />
          <Readout label="total-variation distance" value={formatNumber(result.tv)} />
          <Readout label="Le Cam bound λ²/n" value={formatNumber((lambda * lambda) / n)} />
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
