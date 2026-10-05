import { useMemo } from 'react'
import { Figure, formatNumber, int, Plot, Raster, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { logFactorial } from 'aifn-compute/numerics/special'

/** The joint pmf of a three-category multinomial, over the counts of the first two categories. */
export function MultinomialSimplex() {
  const state = useFigureState({
    n: int(12, { min: 1, max: 40, step: 1, suggestions: [5, 12, 20, 40], label: 'draws n' }),
    w1: slider(0.05, 1, 0.5, { step: 0.05, label: 'weight of category 1' }),
    w2: slider(0.05, 1, 0.3, { step: 0.05, label: 'weight of category 2' }),
    w3: slider(0.05, 1, 0.2, { step: 0.05, label: 'weight of category 3' }),
  })
  const weights = [state.w1, state.w2, state.w3]
  const total = weights.reduce((a, b) => a + b, 0)
  const pi = weights.map((w) => w / total)

  const grid = useMemo(() => {
    const counts = Array.from({ length: state.n + 1 }, (_, i) => i)
    const logPi = pi.map(Math.log)
    let top = 0
    // Row-major: z[x2][x1]. Cells with x1 + x2 > n are impossible and left empty.
    const z = counts.map((x2) =>
      counts.map((x1) => {
        const x3 = state.n - x1 - x2
        if (x3 < 0) return NaN
        const logP =
          logFactorial(state.n) -
          logFactorial(x1) -
          logFactorial(x2) -
          logFactorial(x3) +
          x1 * logPi[0] +
          x2 * logPi[1] +
          x3 * logPi[2]
        const p = Math.exp(logP)
        top = Math.max(top, p)
        return p
      }),
    )
    return { counts, z, range: [0, top] as [number, number] }
  }, [state.n, pi[0], pi[1], pi[2]]) // eslint-disable-line react-hooks/exhaustive-deps

  const corr = -Math.sqrt((pi[0] * pi[1]) / ((1 - pi[0]) * (1 - pi[1])))

  const xAxis = useAxis({ label: 'count x₁', hold: 'union' })
  const yAxis = useAxis({ label: 'count x₂', hold: 'union' })
  return (
    <Figure
      title="Three categories, n draws"
      state={state}
      caption="Each cell is the probability of seeing x₁ draws of category 1 and x₂ of category 2 in n draws; category 3 takes the remaining n − x₁ − x₂. Cells above the diagonal are impossible. The weights are normalised to probabilities π. The mass sits near nπ and tilts along the diagonal, because one more draw in one category means one fewer elsewhere."
      readouts={
        <>
          <Readout label="π" value={pi.map((p) => formatNumber(p)).join(', ')} />
          <Readout label="mean nπ" value={pi.map((p) => formatNumber(state.n * p)).join(', ')} />
          <Readout label="corr(X₁, X₂)" value={formatNumber(corr)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Raster x={grid.counts} y={grid.counts} z={grid.z} range={grid.range} valueLabel={'probability'} />
      </Plot>
    </Figure>
  )
}
