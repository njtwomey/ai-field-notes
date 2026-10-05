import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { logChoose } from 'aifn-compute/numerics/special'

const NS = Array.from({ length: 60 }, (_, i) => 10 * (i + 1))
const FLOOR = 1e-12

/** P(|X̄ − p| ≥ ε) for the mean of n Bernoulli(p) draws, summed exactly from the binomial pmf. */
function exactTail(n: number, p: number, eps: number): number {
  let total = 0
  for (let k = 0; k <= n; k++) {
    // A small tolerance keeps k/n exactly at p ± ε inside the event despite rounding.
    if (Math.abs(k / n - p) >= eps - 1e-12) {
      total += Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))
    }
  }
  return total
}

/** Exact two-sided tail of a Bernoulli mean against the Chebyshev and Hoeffding bounds, on a log scale. */
export function BoundsVsN() {
  const state = useFigureState({
    eps: float(0.1, { min: 0.02, max: 0.3, step: 0.01, label: 'deviation ε' }),
    p: slider(0.05, 0.95, 0.5, { step: 0.05, label: 'success probability p' }),
  })

  const series = useMemo(() => {
    const chebyshev = NS.map((n) => Math.min(1, (state.p * (1 - state.p)) / (n * state.eps ** 2)))
    const hoeffding = NS.map((n) => Math.min(1, 2 * Math.exp(-2 * n * state.eps ** 2)))
    const exact = NS.map((n) => Math.max(FLOOR, exactTail(n, state.p, state.eps)))
    return [
      { name: 'exact', x: NS, y: exact, slot: 0 },
      { name: 'Chebyshev p(1−p)/(nε²)', x: NS, y: chebyshev.map((v) => Math.max(FLOOR, v)), slot: 1 },
      { name: 'Hoeffding 2e^(−2nε²)', x: NS, y: hoeffding.map((v) => Math.max(FLOOR, v)), slot: 2 },
    ] as const
  }, [state.eps, state.p])

  // Smallest n (from the two formulas) that brings each bound down to 0.05.
  const nCheb = Math.ceil((state.p * (1 - state.p)) / (0.05 * state.eps ** 2))
  const nHoeff = Math.ceil(Math.log(2 / 0.05) / (2 * state.eps ** 2))

  const xAxis = useAxis({ label: 'n', hold: 'union' })
  const yAxis = useAxis({ label: 'P(|X̄ − p| ≥ ε)', range: [FLOOR, 1], log: true })
  return (
    <Figure
      title="Chebyshev and Hoeffding against the exact tail"
      state={state}
      caption="The probability that the mean of n Bernoulli(p) draws misses p by at least ε. The Chebyshev bound falls like 1/n, the Hoeffding bound and the exact tail fall exponentially. Hoeffding ignores the variance, so it is loosest when p is near 0 or 1, where Chebyshev can win for small n."

      readouts={
        <>
          <Readout label="n for Chebyshev ≤ 0.05" value={formatNumber(nCheb)} />
          <Readout label="n for Hoeffding ≤ 0.05" value={formatNumber(nHoeff)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
