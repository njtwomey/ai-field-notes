import { useMemo } from 'react'
import { Bars, choice, Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform, type Stream } from 'aifn/foundation/random'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalPdf } from 'aifn/numerics/special'

type Source = 'normal' | 'exponential' | 'uniform'

const REPEATS = 3000
const BINS = 40

/** Each population has mean 1 and standard deviation 1, so only the shape differs. */
const draw: Record<Source, (r: Stream) => number> = {
  normal: (r) => 1 + normal(r),
  exponential: (r) => -Math.log(1 - uniform(r)),
  uniform: (r) => 1 + Math.sqrt(3) * (2 * uniform(r) - 1),
}

/** The sample mean of n draws, repeated many times, against the normal curve with standard deviation σ/√n. */
export function SamplingDistribution() {
  const state = useFigureState({
    n: int(10, { min: 1, max: 100, step: 1, label: 'sample size n' }),
    source: choice<Source>(
      [
        { value: 'normal', label: 'normal' },
        { value: 'exponential', label: 'exponential' },
        { value: 'uniform', label: 'uniform' },
      ],
      'exponential',
      { label: 'population' },
    ),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const result = useMemo(() => {
    const r = stream(state.seed)
    const means = Array.from({ length: REPEATS }, () => {
      let total = 0
      for (let i = 0; i < state.n; i++) total += draw[state.source](r)
      return total / state.n
    })
    const se = 1 / Math.sqrt(state.n)
    const lo = 1 - 4.5 * se
    const hi = 1 + 4.5 * se
    const width = (hi - lo) / BINS
    const counts = new Array(BINS).fill(0)
    for (const m of means) {
      const b = Math.floor((m - lo) / width)
      if (b >= 0 && b < BINS) counts[b]++
    }
    const grand = means.reduce((a, b) => a + b, 0) / REPEATS
    const spread = Math.sqrt(means.reduce((a, b) => a + (b - grand) ** 2, 0) / (REPEATS - 1))
    const xs = toFlat(linspace(lo, hi, 200))
    const series = [
      {
        name: 'sample means',
        x: counts.map((_, i) => lo + (i + 0.5) * width),
        y: counts.map((c) => c / (REPEATS * width)),
        slot: 0,
      },
      { name: 'N(μ, σ²/n)', x: xs, y: xs.map((x) => normalPdf((x - 1) / se) / se), slot: 1 },
    ] as const
    return { series, spread, se }
  }, [state.n, state.source, state.seed])

  const xAxis = useAxis({ label: 'sample mean', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="The spread of the sample mean"
      state={state}
      caption="Each bar counts sample means from 3,000 independent samples of size n. The population has mean 1 and standard deviation 1 in every case. The means spread with standard deviation 1/√n, the standard error, whatever the population's shape. For a skewed population the histogram is skewed at small n and becomes normal as n grows."
      readouts={
        <>
          <Readout label="sd of the 3,000 means" value={formatNumber(result.spread)} />
          <Readout label="σ/√n" value={formatNumber(result.se)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Bars {...result.series[0]} />
        <Curve {...result.series[1]} />
      </Plot>
    </Figure>
  )
}
