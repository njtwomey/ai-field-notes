import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { incompleteGamma } from '@/lib/math/special'

const BINS = 10
const GRID = linspace(-4, 4, 161)

/** Asymptotic Kolmogorov tail probability P(K > t) = 2 Σ (−1)^(k−1) exp(−2k²t²). */
function kolmogorovTail(t: number) {
  if (t < 0.3) return 1
  let s = 0
  for (let k = 1; k <= 100; k++) s += (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * t * t)
  return Math.min(1, Math.max(0, 2 * s))
}

/** Upper tail of the χ² distribution with `df` degrees of freedom, via the regularised lower incomplete gamma. */
const chiSquareTail = (x: number, df: number) => 1 - incompleteGamma(df / 2, x / 2)

/** Empirical cdf of a sorted sample at x. */
function ecdf(sorted: number[], x: number) {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] <= x) lo = mid + 1
    else hi = mid
  }
  return lo / sorted.length
}

export function DriftTests() {
  const [shift, setShift] = useState(0.1)
  const [scale, setScale] = useState(1)
  const [n, setN] = useState(1000)
  const [seed, setSeed] = useState(1)

  const r = useMemo(() => {
    const g = rng(seed)
    const ref = Array.from({ length: n }, () => g.normal()).sort((a, b) => a - b)
    const cur = Array.from({ length: n }, () => shift + scale * g.normal()).sort((a, b) => a - b)
    // Two-sample KS statistic: largest gap between the empirical cdfs, checked at every sample point.
    let d = 0
    let at = 0
    for (const x of [...ref, ...cur]) {
      const gap = Math.abs(ecdf(ref, x) - ecdf(cur, x))
      if (gap > d) {
        d = gap
        at = x
      }
    }
    const ne = (n * n) / (n + n)
    const pKs = kolmogorovTail(Math.sqrt(ne) * d)
    // PSI on the reference sample's deciles, so each expected share is 1/10.
    const edges = Array.from({ length: BINS - 1 }, (_, i) => ref[Math.floor(((i + 1) * n) / BINS)])
    const bin = (x: number) => edges.filter((e) => x >= e).length
    const e = new Array<number>(BINS).fill(0)
    const a = new Array<number>(BINS).fill(0)
    ref.forEach((x) => e[bin(x)]++)
    cur.forEach((x) => a[bin(x)]++)
    let psi = 0
    for (let i = 0; i < BINS; i++) {
      const ei = Math.max(e[i], 0.5) / n
      const ai = Math.max(a[i], 0.5) / n
      psi += (ai - ei) * Math.log(ai / ei)
    }
    const pPsi = chiSquareTail(psi / (2 / n), BINS - 1)
    const series: XYSeries[] = [
      { name: 'reference', type: 'line', x: GRID, y: GRID.map((x) => ecdf(ref, x)), slot: 0 },
      { name: 'current', type: 'line', x: GRID, y: GRID.map((x) => ecdf(cur, x)), slot: 1 },
    ]
    const gap: Segment[] = [{ from: [at, ecdf(ref, at)], to: [at, ecdf(cur, at)] }]
    return { d, pKs, psi, pPsi, series, gap }
  }, [shift, scale, n, seed])

  return (
    <Interactive
      title="Two-sample tests on one feature"
      caption="Empirical cdfs of a reference window drawn from N(0, 1) and a current window drawn from N(μ, σ²). The KS statistic D is the largest vertical gap, marked by the arrow. With large windows, both tests flag shifts too small to matter, while the PSI rule of thumb of 0.1 stays silent; with small windows, a real shift can pass unnoticed."
      controls={
        <>
          <ParamSlider label="mean shift μ" value={shift} onChange={setShift} min={-1} max={1} step={0.01} />
          <ParamSlider label="scale σ" value={scale} onChange={setScale} min={0.5} max={2} step={0.01} />
          <ParamSlider label="window size n (each)" value={n} onChange={setN} min={50} max={5000} step={50} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New samples</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="KS D" value={formatNumber(r.d)} />
          <Readout label="KS p-value" value={formatNumber(r.pKs)} />
          <Readout label="PSI (10 bins)" value={formatNumber(r.psi)} />
          <Readout label="PSI χ² p-value" value={formatNumber(r.pPsi)} />
        </>
      }
    >
      <XYChart
        series={r.series}
        vectors={r.gap}
        xLabel="feature value"
        yLabel="empirical cdf"
        xRange={[-4, 4]}
        yRange={[0, 1]}
        height={300}
      />
    </Interactive>
  )
}
