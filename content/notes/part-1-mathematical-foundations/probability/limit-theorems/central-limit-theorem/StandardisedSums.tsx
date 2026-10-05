import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import type { Stream } from 'aifn-compute/foundation/random'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'

type DistId = 'uniform' | 'exponential' | 'lognormal'

type Law = {
  label: string
  mean: number
  sd: number
  skew: number
  /** E|X − μ|³ / σ³, the ratio in the Berry–Esseen bound. */
  rho: number
  draw: (r: Stream) => number
}

const LAWS: Record<DistId, Law> = {
  uniform: { label: 'uniform', mean: 0.5, sd: Math.sqrt(1 / 12), skew: 0, rho: 1.299, draw: (r) => uniform(r) },
  exponential: { label: 'exponential', mean: 1, sd: 1, skew: 2, rho: 2.415, draw: (r) => -Math.log(1 - uniform(r)) },
  lognormal: {
    label: 'log-normal',
    mean: Math.exp(0.5),
    sd: Math.sqrt((Math.E - 1) * Math.E),
    skew: 6.185,
    rho: 6.35,
    draw: (r) => Math.exp(normal(r)),
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
const GRID = toFlat(linspace(LO, HI, 200))

/** Histogram of standardised sums of n draws against the standard normal density. */
export function StandardisedSums() {
  const state = useFigureState({
    id: choice<DistId>(
      (Object.keys(LAWS) as DistId[]).map((k) => ({ value: k, label: LAWS[k].label })),
      'exponential',
      { label: 'distribution of one draw' },
    ),
    n: int(1, { min: 1, max: MAX_N, step: 1, label: 'n (draws per sum)' }),
  })
  const law = LAWS[state.id]

  // Partial sums for every replicate and every n up to MAX_N, simulated once per distribution.
  const sums = useMemo(() => {
    const r = stream(7)
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
    const scale = law.sd * Math.sqrt(state.n)
    for (let k = 0; k < REPS; k++) z[k] = (sums[k * MAX_N + state.n - 1] - state.n * law.mean) / scale
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
    const s: SeriesSpec[] = [
      { name: 'standardised sums', type: 'bar', x: CENTRES, y: counts.map((c) => c / (REPS * WIDTH)), slot: 0 },
      { name: 'standard normal', type: 'line', x: GRID, y: GRID.map((v: number) => normalPdf(v)), emphasis: true },
    ]
    return { series: s, ks: d }
  }, [sums, law, state.n])

  const xAxis = useAxis({ label: 'standardised sum z', range: [LO, HI] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Standardised sums approach the normal"
      state={state}
      caption="Histogram of (Sₙ − nμ)/(σ√n) over 20,000 simulated sums of n draws, with the standard normal density. The uniform sum looks normal by n ≈ 5. The exponential needs tens of draws, and the heavily skewed log-normal needs far more. The distance to the normal shrinks like skewness/√n."

      readouts={
        <>
          <Readout label="skewness of the sum" value={formatNumber(law.skew / Math.sqrt(state.n))} />
          <Readout label="sup |Fₙ − Φ| (simulated)" value={formatNumber(ks)} />
          <Readout label="Berry–Esseen bound" value={formatNumber((BERRY_ESSEEN_C * law.rho) / Math.sqrt(state.n))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
