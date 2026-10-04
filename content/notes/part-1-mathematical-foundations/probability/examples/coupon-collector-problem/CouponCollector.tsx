import { useMemo } from 'react'
import { Bars, Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'

const harmonic = (n: number) => Array.from({ length: n }, (_, k) => 1 / (k + 1)).reduce((a, b) => a + b, 0)

/** Simulates collecting all n coupons many times; compares total and per-stage waits with the geometric-sum answer. */
export function CouponCollector() {
  const state = useFigureState({
    n: int(10, { min: 2, max: 100, step: 1, label: 'coupons n' }),
    runs: int(2000, { min: 200, max: 10000, step: 200, label: 'runs' }),
    seed: int(1, { min: 0, max: 30, step: 1, label: 'seed' }),
  })

  const result = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    const totals: number[] = []
    const stageSums = new Array<number>(state.n).fill(0)
    const have = new Uint32Array(state.n)
    for (let r = 1; r <= state.runs; r++) {
      let found = 0
      let draws = 0
      let since = 0
      while (found < state.n) {
        draws++
        since++
        const c = Math.floor(uniform() * state.n)
        if (have[c] !== r) {
          have[c] = r
          stageSums[found] += since
          found++
          since = 0
        }
      }
      totals.push(draws)
    }
    const mean = totals.reduce((a, b) => a + b, 0) / state.runs
    const sd = Math.sqrt(totals.reduce((a, t) => a + (t - mean) ** 2, 0) / (state.runs - 1))
    return { totals, mean, sd, stageMeans: stageSums.map((s) => s / state.runs) }
  }, [state.n, state.runs, state.seed])

  const exactMean = state.n * harmonic(state.n)
  const exactSd = Math.sqrt(
    state.n * state.n * Array.from({ length: state.n }, (_, k) => 1 / (k + 1) ** 2).reduce((a, b) => a + b, 0) -
      exactMean,
  )

  const histogram = useMemo(() => {
    const max = Math.max(...result.totals)
    const width = Math.max(1, Math.ceil((max - state.n) / 40))
    const bins = Math.floor((max - state.n) / width) + 1
    const counts = new Array<number>(bins).fill(0)
    for (const t of result.totals) counts[Math.floor((t - state.n) / width)]++
    const x = counts.map((_, i) => state.n + i * width + width / 2)
    const top = Math.max(...counts) / (state.runs * width)
    return [
      { name: 'simulated runs', x, y: counts.map((c) => c / (state.runs * width)), slot: 0 },
      { name: 'exact mean n·Hₙ', x: [exactMean, exactMean], y: [0, top], dashed: true, slot: 1 },
    ] as const
  }, [result, state.n, state.runs, exactMean])

  const stages = useMemo(() => {
    const ks = Array.from({ length: state.n }, (_, k) => k + 1)
    return [
      { name: 'exact n / (n − k + 1)', x: ks, y: ks.map((k) => state.n / (state.n - k + 1)), slot: 1 },
      { name: 'simulated mean wait', x: ks, y: result.stageMeans, slot: 0 },
    ] as const
  }, [result, state.n])

  const xAxis = useAxis({ label: 'draws to complete', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'k-th new coupon', hold: 'union' })
  const yAxis2 = useAxis({ label: 'expected draws', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Collecting every coupon"
      state={state}
      caption="Left: the distribution of the number of draws needed to collect all n coupons, over many simulated runs, with the exact mean. Right: the wait for the k-th new coupon. Each wait is geometric with success probability (n − k + 1)/n, so the last few coupons take most of the time."

      readouts={
        <>
          <Readout label="exact mean n·Hₙ" value={formatNumber(exactMean)} />
          <Readout label="simulated mean" value={formatNumber(result.mean)} />
          <Readout label="exact sd" value={formatNumber(exactSd)} />
          <Readout label="simulated sd" value={formatNumber(result.sd)} />
          <Readout
            label="n ln n + γn + 1/2"
            value={formatNumber(state.n * Math.log(state.n) + 0.5772156649 * state.n + 0.5)}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Bars {...histogram[0]} />
          <Curve {...histogram[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...stages[0]} />
          <Points {...stages[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
