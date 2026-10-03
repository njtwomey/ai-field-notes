import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, formatNumber } from 'aifn-render'
import { logFactorial } from '@/lib/math/special'

/** The joint pmf of a three-category multinomial, over the counts of the first two categories. */
export function MultinomialSimplex() {
  const [n, setN] = useState(12)
  const [weights, setWeights] = useState([0.5, 0.3, 0.2])
  const total = weights.reduce((a, b) => a + b, 0)
  const pi = weights.map((w) => w / total)

  const grid = useMemo(() => {
    const counts = Array.from({ length: n + 1 }, (_, i) => i)
    const logPi = pi.map(Math.log)
    let top = 0
    // Row-major: z[x2][x1]. Cells with x1 + x2 > n are impossible and left empty.
    const z = counts.map((x2) =>
      counts.map((x1) => {
        const x3 = n - x1 - x2
        if (x3 < 0) return NaN
        const logP =
          logFactorial(n) -
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
  }, [n, pi[0], pi[1], pi[2]]) // eslint-disable-line react-hooks/exhaustive-deps

  const corr = -Math.sqrt((pi[0] * pi[1]) / ((1 - pi[0]) * (1 - pi[1])))
  const setWeight = (k: number) => (v: number) => setWeights((prev) => prev.map((old, j) => (j === k ? v : old)))

  return (
    <Interactive
      title="Three categories, n draws"
      caption="Each cell is the probability of seeing x₁ draws of category 1 and x₂ of category 2 in n draws; category 3 takes the remaining n − x₁ − x₂. Cells above the diagonal are impossible. The weights are normalised to probabilities π. The mass sits near nπ and tilts along the diagonal, because one more draw in one category means one fewer elsewhere."
      controls={
        <>
          <ParamSlider label="draws n" value={n} onChange={setN} min={1} max={40} step={1} />
          {weights.map((w, k) => (
            <ParamSlider
              key={k}
              label={`weight of category ${k + 1}`}
              value={w}
              onChange={setWeight(k)}
              min={0.05}
              max={1}
              step={0.05}
            />
          ))}
        </>
      }
      readout={
        <>
          <Readout label="π" value={pi.map((p) => formatNumber(p)).join(', ')} />
          <Readout label="mean nπ" value={pi.map((p) => formatNumber(n * p)).join(', ')} />
          <Readout label="corr(X₁, X₂)" value={formatNumber(corr)} />
        </>
      }
    >
      <Heatmap
        x={grid.counts}
        y={grid.counts}
        z={grid.z}
        range={grid.range}
        xLabel="count x₁"
        yLabel="count x₂"
        valueLabel="probability"
        height={360}
      />
    </Interactive>
  )
}
