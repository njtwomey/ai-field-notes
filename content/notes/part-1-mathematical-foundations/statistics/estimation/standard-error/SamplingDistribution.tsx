import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'

type Source = 'normal' | 'exponential' | 'uniform'

const REPEATS = 3000
const BINS = 40

/** Each population has mean 1 and standard deviation 1, so only the shape differs. */
const draw: Record<Source, (r: ReturnType<typeof rng>) => number> = {
  normal: (r) => 1 + r.normal(),
  exponential: (r) => -Math.log(1 - r.uniform()),
  uniform: (r) => 1 + Math.sqrt(3) * (2 * r.uniform() - 1),
}

/** The sample mean of n draws, repeated many times, against the normal curve with standard deviation σ/√n. */
export function SamplingDistribution() {
  const [n, setN] = useState(10)
  const [source, setSource] = useState<Source>('exponential')
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const r = rng(seed)
    const means = Array.from({ length: REPEATS }, () => {
      let total = 0
      for (let i = 0; i < n; i++) total += draw[source](r)
      return total / n
    })
    const se = 1 / Math.sqrt(n)
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
    const xs = linspace(lo, hi, 200)
    const series: XYSeries[] = [
      {
        name: 'sample means',
        type: 'bar',
        x: counts.map((_, i) => lo + (i + 0.5) * width),
        y: counts.map((c) => c / (REPEATS * width)),
        slot: 0,
      },
      { name: 'N(μ, σ²/n)', type: 'line', x: xs, y: xs.map((x) => normalPdf((x - 1) / se) / se), slot: 1 },
    ]
    return { series, spread, se }
  }, [n, source, seed])

  return (
    <Interactive
      title="The spread of the sample mean"
      caption="Each bar counts sample means from 3,000 independent samples of size n. The population has mean 1 and standard deviation 1 in every case. The means spread with standard deviation 1/√n, the standard error, whatever the population's shape. For a skewed population the histogram is skewed at small n and becomes normal as n grows."
      controls={
        <>
          <ParamSlider label="sample size n" value={n} onChange={setN} min={1} max={100} step={1} />
          <ParamChoice
            label="population"
            value={source}
            onChange={setSource}
            options={[
              { value: 'normal', label: 'normal' },
              { value: 'exponential', label: 'exponential' },
              { value: 'uniform', label: 'uniform' },
            ]}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New samples</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="sd of the 3,000 means" value={formatNumber(result.spread)} />
          <Readout label="σ/√n" value={formatNumber(result.se)} />
        </>
      }
    >
      <XYChart height={280} series={result.series} xLabel="sample mean" yLabel="density" yRange={[0, undefined]} />
    </Interactive>
  )
}
