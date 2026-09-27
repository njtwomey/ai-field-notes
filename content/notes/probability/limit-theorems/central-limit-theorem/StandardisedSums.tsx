import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'

type DistId = 'uniform' | 'exponential' | 'lognormal'

type Law = {
  label: string
  mean: number
  sd: number
  skew: number
  /** E|X − μ|³ / σ³, the ratio in the Berry–Esseen bound. */
  rho: number
  draw: (r: ReturnType<typeof rng>) => number
}

const LAWS: Record<DistId, Law> = {
  uniform: { label: 'uniform', mean: 0.5, sd: Math.sqrt(1 / 12), skew: 0, rho: 1.299, draw: (r) => r.uniform() },
  exponential: { label: 'exponential', mean: 1, sd: 1, skew: 2, rho: 2.415, draw: (r) => -Math.log(1 - r.uniform()) },
  lognormal: {
    label: 'log-normal',
    mean: Math.exp(0.5),
    sd: Math.sqrt((Math.E - 1) * Math.E),
    skew: 6.185,
    rho: 6.35,
    draw: (r) => Math.exp(r.normal()),
  },
}

const REPS = 20000
const MAX_N = 60
/** Shevtsova's constant for identically distributed summands. */
const BERRY_ESSEEN_C = 0.4748
const LO = -4
const HI = 4
const BINS = 64
const WIDTH = (HI - LO) / BINS
const CENTRES = Array.from({ length: BINS }, (_, i) => LO + (i + 0.5) * WIDTH)
const GRID = linspace(LO, HI, 200)

/** Histogram of standardised sums of n draws against the standard normal density. */
export function StandardisedSums() {
  const [id, setId] = useState<DistId>('exponential')
  const [n, setN] = useState(1)
  const law = LAWS[id]

  // Partial sums for every replicate and every n up to MAX_N, simulated once per distribution.
  const sums = useMemo(() => {
    const r = rng(7)
    const out = new Float64Array(REPS * MAX_N)
    for (let k = 0; k < REPS; k++) {
      let s = 0
      for (let j = 0; j < MAX_N; j++) {
        s += law.draw(r)
        out[k * MAX_N + j] = s
      }
    }
    return out
  }, [law])

  const { series, ks } = useMemo(() => {
    const z = new Float64Array(REPS)
    const scale = law.sd * Math.sqrt(n)
    for (let k = 0; k < REPS; k++) z[k] = (sums[k * MAX_N + n - 1] - n * law.mean) / scale
    const counts = new Array<number>(BINS).fill(0)
    for (const v of z) {
      const b = Math.floor((v - LO) / WIDTH)
      if (b >= 0 && b < BINS) counts[b]++
    }
    // Kolmogorov distance between the empirical distribution of the standardised sums and Φ.
    const sorted = Float64Array.from(z).sort()
    let d = 0
    for (let k = 0; k < REPS; k++) {
      const phi = normalCdf(sorted[k])
      d = Math.max(d, Math.abs((k + 1) / REPS - phi), Math.abs(k / REPS - phi))
    }
    const s: XYSeries[] = [
      { name: 'standardised sums', type: 'bar', x: CENTRES, y: counts.map((c) => c / (REPS * WIDTH)), slot: 0 },
      { name: 'standard normal', type: 'line', x: GRID, y: GRID.map(normalPdf), emphasis: true },
    ]
    return { series: s, ks: d }
  }, [sums, law, n])

  return (
    <Interactive
      title="Standardised sums approach the normal"
      caption="Histogram of (Sₙ − nμ)/(σ√n) over 20,000 simulated sums of n draws, with the standard normal density. The uniform sum looks normal by n ≈ 5. The exponential needs tens of draws, and the heavily skewed log-normal needs far more. The distance to the normal shrinks like skewness/√n."
      controls={
        <>
          <ParamChoice
            label="distribution of one draw"
            value={id}
            onChange={setId}
            options={(Object.keys(LAWS) as DistId[]).map((k) => ({ value: k, label: LAWS[k].label }))}
          />
          <ParamSlider label="n (draws per sum)" value={n} onChange={setN} min={1} max={MAX_N} step={1} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="skewness of the sum" value={formatNumber(law.skew / Math.sqrt(n))} />
          <Readout label="sup |Fₙ − Φ| (simulated)" value={formatNumber(ks)} />
          <Readout label="Berry–Esseen bound" value={formatNumber((BERRY_ESSEEN_C * law.rho) / Math.sqrt(n))} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="standardised sum z"
        yLabel="density"
        series={series}
        xRange={[LO, HI]}
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}
