import { useMemo } from 'react'
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'
import { ecdf, ecdfAt } from 'aifn-compute/probability/stats'
import { kolmogorovSf, ksStatistic, twoSampleKsSf } from 'aifn-compute/probability/tests'
import { Curve, Figure, formatNumber, int, Plot, Plots, Readout, Segments, useAxis, useFigureState } from 'aifn-render'

const LO = -3.5
const HI = 3.5
const law = Normal(0, 1)
const cdf = (t: number) => law.cdf(t) as number
const GRID = Array.from({ length: 141 }, (_, i) => LO + ((HI - LO) * i) / 140)
const TRUTH = GRID.map(cdf)
const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/** The empirical cdf of a sample as a step curve from LO to HI, from the engine's ecdf. */
function steps(sample: readonly number[]) {
  const { values, probabilities } = ecdf(sample)
  const v = toFlat(values)
  const p = toFlat(probabilities)
  const x: number[] = [LO]
  const y: number[] = [0]
  v.forEach((value, i) => {
    x.push(value, value)
    y.push(i === 0 ? 0 : p[i - 1], p[i])
  })
  x.push(HI)
  y.push(1)
  return { x, y }
}

/**
 * Symmetrisation on the class of half-lines: a sample and an independent ghost sample, the largest gap of the sample
 * from the truth and from the ghost, and the exact tail of the first beside twice the exact tail of the second at
 * half the deviation (the ghost-sample lemma's right-hand side).
 */
export function GhostSample() {
  const state = useFigureState({
    m: int(50, { ge: 5, le: 300, suggestions: [20, 50, 200], label: 'sample size m' }),
    seed: int(1, { ge: 0, le: 999, label: 'seed' }),
  })
  const { m, seed } = state
  const draws = useMemo(() => {
    const s = child(stream('symmetrisation'), seed)
    const x = toFlat(normal(child(s, 'sample'), 0, 1, { shape: [m] }) as Tensor)
    const ghost = toFlat(normal(child(s, 'ghost'), 0, 1, { shape: [m] }) as Tensor)
    return { x, ghost, fx: steps(x), fg: steps(ghost) }
  }, [m, seed])
  const one = useMemo(() => ksStatistic(draws.x, cdf), [draws])
  const two = useMemo(() => ksStatistic(draws.x, draws.ghost), [draws])
  const at = (sample: number[], t: number) => toFlat(ecdfAt(sample, [t]))[0]
  // A one-sample gap with the empirical cdf below the truth is reached just before the draw.
  const oneTop = one.sign > 0 ? at(draws.x, one.location) : cdf(one.location) - one.statistic

  const tails = useMemo(() => {
    const tMax = Math.min(0.9, 6 / Math.sqrt(m))
    const t = Array.from({ length: 60 }, (_, i) => (tMax * (i + 1)) / 60)
    return {
      t,
      direct: t.map((d) => kolmogorovSf(d, m)),
      ghost: t.map((d) => Math.min(1, 2 * twoSampleKsSf(d / 2, m, m))),
    }
  }, [m])
  const valid = Math.sqrt(2 / m)
  const tEnd = tails.t[tails.t.length - 1]

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'cumulative probability', range: [0, 1] })
  const ta = useAxis({ label: 'deviation t', range: [0, tEnd] })
  const pa = useAxis({ label: 'probability', range: [1e-6, 1], log: true })
  return (
    <Figure
      title="The ghost sample replaces the truth"
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="sup gap to the truth" value={fmt(one.statistic)} />
          <Readout label="sup gap to the ghost" value={fmt(two.statistic)} />
        </>
      }
      caption="The class is the half-lines (−∞, t], whose empirical means form the empirical cdf. Left: a sample of m draws from N(0, 1) (slot 0), an independent ghost sample of the same size (slot 1) and the true cdf. The thick segment is the uniform deviation of the sample from the truth; the thin one is its deviation from the ghost, which needs no knowledge of the truth. Right: the exact probability that the first exceeds t (the engine's kolmogorovSf), against the symmetrisation bound, twice the exact probability that the second exceeds t/2 (twoSampleKsSf). The bound holds wherever mt² ≥ 2, right of the thin dashed vertical line, and is loose by roughly a factor of 3 in t."
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya} title="sample, ghost and truth">
          <Curve name="true cdf" x={GRID} y={TRUTH} emphasis thin dashed />
          <Curve name="sample S" x={draws.fx.x} y={draws.fx.y} slot={0} />
          <Curve name="ghost S′" x={draws.fg.x} y={draws.fg.y} slot={1} />
          <Segments
            name="sup gap to the truth"
            segments={[{ from: [one.location, cdf(one.location)], to: [one.location, oneTop] }]}
            emphasis
            width={3}
          />
          <Segments
            name="sup gap to the ghost"
            segments={[
              {
                from: [two.location, at(draws.x, two.location)],
                to: [two.location, at(draws.ghost, two.location)],
              },
            ]}
            slot={2}
            width={2}
          />
        </Plot>
        <Plot x={ta} y={pa} title="tails of the two deviations">
          <Curve name="P(sup gap to truth ≥ t)" x={tails.t} y={tails.direct} slot={0} />
          <Curve name="2 P(sup gap to ghost ≥ t/2)" x={tails.t} y={tails.ghost} emphasis dashed />
          <Curve name="mt² = 2" x={[valid, valid]} y={[1e-6, 1]} thin dashed silent />
        </Plot>
      </Plots>
    </Figure>
  )
}
