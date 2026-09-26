import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const harmonic = (n: number) => Array.from({ length: n }, (_, k) => 1 / (k + 1)).reduce((a, b) => a + b, 0)

/** Simulates collecting all n coupons many times; compares total and per-stage waits with the geometric-sum answer. */
export function CouponCollector() {
  const [n, setN] = useState(10)
  const [runs, setRuns] = useState(2000)
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const { uniform } = rng(seed)
    const totals: number[] = []
    const stageSums = new Array<number>(n).fill(0)
    const have = new Uint32Array(n)
    for (let r = 1; r <= runs; r++) {
      let found = 0
      let draws = 0
      let since = 0
      while (found < n) {
        draws++
        since++
        const c = Math.floor(uniform() * n)
        if (have[c] !== r) {
          have[c] = r
          stageSums[found] += since
          found++
          since = 0
        }
      }
      totals.push(draws)
    }
    const mean = totals.reduce((a, b) => a + b, 0) / runs
    const sd = Math.sqrt(totals.reduce((a, t) => a + (t - mean) ** 2, 0) / (runs - 1))
    return { totals, mean, sd, stageMeans: stageSums.map((s) => s / runs) }
  }, [n, runs, seed])

  const exactMean = n * harmonic(n)
  const exactSd = Math.sqrt(
    n * n * Array.from({ length: n }, (_, k) => 1 / (k + 1) ** 2).reduce((a, b) => a + b, 0) - exactMean,
  )

  const histogram = useMemo((): XYSeries[] => {
    const max = Math.max(...result.totals)
    const width = Math.max(1, Math.ceil((max - n) / 40))
    const bins = Math.floor((max - n) / width) + 1
    const counts = new Array<number>(bins).fill(0)
    for (const t of result.totals) counts[Math.floor((t - n) / width)]++
    const x = counts.map((_, i) => n + i * width + width / 2)
    const top = Math.max(...counts) / (runs * width)
    return [
      { name: 'simulated runs', type: 'bar', x, y: counts.map((c) => c / (runs * width)), slot: 0 },
      { name: 'exact mean n·Hₙ', type: 'line', x: [exactMean, exactMean], y: [0, top], dashed: true, slot: 1 },
    ]
  }, [result, n, runs, exactMean])

  const stages = useMemo((): XYSeries[] => {
    const ks = Array.from({ length: n }, (_, k) => k + 1)
    return [
      { name: 'exact n / (n − k + 1)', type: 'line', x: ks, y: ks.map((k) => n / (n - k + 1)), slot: 1 },
      { name: 'simulated mean wait', type: 'scatter', x: ks, y: result.stageMeans, slot: 0 },
    ]
  }, [result, n])

  return (
    <Interactive
      title="Collecting every coupon"
      caption="Left: the distribution of the number of draws needed to collect all n coupons, over many simulated runs, with the exact mean. Right: the wait for the k-th new coupon. Each wait is geometric with success probability (n − k + 1)/n, so the last few coupons take most of the time."
      controls={
        <>
          <ParamSlider label="coupons n" value={n} onChange={setN} min={2} max={100} step={1} />
          <ParamSlider label="runs" value={runs} onChange={setRuns} min={200} max={10000} step={200} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={30} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="exact mean n·Hₙ" value={formatNumber(exactMean)} />
          <Readout label="simulated mean" value={formatNumber(result.mean)} />
          <Readout label="exact sd" value={formatNumber(exactSd)} />
          <Readout label="simulated sd" value={formatNumber(result.sd)} />
          <Readout label="n ln n + γn + 1/2" value={formatNumber(n * Math.log(n) + 0.5772156649 * n + 0.5)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart height={300} xLabel="draws to complete" yLabel="density" series={histogram} yRange={[0, undefined]} />
        <XYChart
          height={300}
          xLabel="k-th new coupon"
          yLabel="expected draws"
          series={stages}
          yRange={[0, undefined]}
        />
      </div>
    </Interactive>
  )
}
