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
} from '@/components/viz'
import { rng } from '@/lib/math'
import { normalCdf, normalQuantile } from '@/lib/math/special'

type Stat = 'mean' | 'median'

const BINS = 40
const ALPHA = 0.05

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
function median(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m])
}
/** Empirical quantile with linear interpolation between order statistics. */
function quantile(sorted: number[], p: number) {
  const h = (sorted.length - 1) * Math.min(Math.max(p, 0), 1)
  const i = Math.floor(h)
  return i + 1 < sorted.length ? sorted[i] + (h - i) * (sorted[i + 1] - sorted[i]) : sorted[i]
}

/**
 * Nonparametric bootstrap of the mean or median of a skewed (log-normal) sample. Draws B resamples with replacement,
 * then forms the percentile, basic and BCa 95% intervals from the same replicates.
 */
export function BootstrapIntervals() {
  const [n, setN] = useState(20)
  const [B, setB] = useState(2000)
  const [stat, setStat] = useState<Stat>('median')
  const [seed, setSeed] = useState(1)

  const r = useMemo(() => {
    const f = stat === 'mean' ? mean : median
    const g = rng(seed)
    const x = Array.from({ length: n }, () => Math.exp(g.normal()))
    const theta = f(x)
    const reps = Array.from({ length: B }, () => f(Array.from({ length: n }, () => x[Math.floor(g.uniform() * n)])))
    const sorted = [...reps].sort((a, b) => a - b)
    const m = mean(reps)
    const se = Math.sqrt(reps.reduce((a, t) => a + (t - m) ** 2, 0) / (B - 1))
    const pct: [number, number] = [quantile(sorted, ALPHA / 2), quantile(sorted, 1 - ALPHA / 2)]
    const basic: [number, number] = [2 * theta - pct[1], 2 * theta - pct[0]]
    // BCa: bias correction z0 from the fraction of replicates below θ̂, acceleration a from the jackknife.
    const below = reps.filter((t) => t < theta).length + 0.5 * reps.filter((t) => t === theta).length
    const z0 = normalQuantile(Math.min(Math.max(below / B, 1 / B), 1 - 1 / B))
    const jack = x.map((_, i) => f(x.filter((__, j) => j !== i)))
    const jm = mean(jack)
    const num = jack.reduce((a, t) => a + (jm - t) ** 3, 0)
    const den = jack.reduce((a, t) => a + (jm - t) ** 2, 0)
    const acc = den > 0 ? num / (6 * den ** 1.5) : 0
    const adj = (z: number) => normalCdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    const zq = normalQuantile(1 - ALPHA / 2)
    const bca: [number, number] = [quantile(sorted, adj(-zq)), quantile(sorted, adj(zq))]

    const lo = sorted[0]
    const hi = sorted[B - 1]
    const width = (hi - lo) / BINS || 1
    const counts = new Array<number>(BINS).fill(0)
    for (const t of reps) counts[Math.min(BINS - 1, Math.floor((t - lo) / width))]++
    const density = counts.map((c) => c / (B * width))
    const top = Math.max(...density)
    const level = (k: number) => top * (1.08 + 0.08 * k)
    const bar = (name: string, [a, b]: [number, number], k: number, slot: number): XYSeries => ({
      name,
      type: 'line',
      x: [a, a, NaN, a, b, NaN, b, b],
      y: [
        level(k) - 0.03 * top,
        level(k) + 0.03 * top,
        NaN,
        level(k),
        level(k),
        NaN,
        level(k) - 0.03 * top,
        level(k) + 0.03 * top,
      ],
      slot,
    })
    const series: XYSeries[] = [
      {
        name: 'bootstrap replicates',
        type: 'bar',
        x: counts.map((_, i) => lo + (i + 0.5) * width),
        y: density,
        slot: 0,
      },
      { name: 'estimate θ̂', type: 'line', x: [theta, theta], y: [0, level(3)], emphasis: true, dashed: true },
      bar('percentile', pct, 0, 1),
      bar('basic', basic, 1, 2),
      bar('BCa', bca, 2, 3),
    ]
    return { theta, se, pct, basic, bca, z0, acc, series, top: level(3) * 1.04, bias: m - theta }
  }, [n, B, stat, seed])

  const fmt = ([a, b]: [number, number]) => `[${formatNumber(a)}, ${formatNumber(b)}]`
  return (
    <Interactive
      title="The bootstrap distribution and three intervals"
      caption="A sample of size n from a right-skewed log-normal distribution (true median 1, true mean 1.65) is resampled B times with replacement. The histogram is the bootstrap distribution of the statistic; the three bars above it are 95% intervals built from the same replicates. The percentile and BCa intervals follow the skew of the bootstrap distribution; the basic interval reflects it about θ̂. For the median of a small sample the histogram is spiky, because a resampled median can only take a few of the observed values."
      controls={
        <>
          <ParamChoice
            label="statistic"
            value={stat}
            onChange={setStat}
            options={[
              { value: 'mean', label: 'mean' },
              { value: 'median', label: 'median' },
            ]}
          />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={5} max={100} step={1} />
          <ParamSlider label="resamples B" value={B} onChange={setB} min={100} max={5000} step={100} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="θ̂" value={formatNumber(r.theta)} />
          <Readout label="bootstrap SE" value={formatNumber(r.se)} />
          <Readout label="bias estimate" value={formatNumber(r.bias)} />
          <Readout label="percentile" value={fmt(r.pct)} />
          <Readout label="basic" value={fmt(r.basic)} />
          <Readout label="BCa" value={fmt(r.bca)} />
          <Readout label="z₀, a" value={`${formatNumber(r.z0)}, ${formatNumber(r.acc)}`} />
        </>
      }
    >
      <XYChart height={320} series={r.series} yRange={[0, r.top]} xLabel="value of the statistic" yLabel="density" />
    </Interactive>
  )
}
