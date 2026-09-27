import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

const DIMS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
const LOG_DIMS = DIMS.map((d) => Math.log10(d))
const QUERIES = 10
const HIST_RANGE: [number, number] = [0, 3]
const BINS = 30

type Norm = '1' | '2' | 'inf'
const NORMS = [
  { value: '1', label: 'L1' },
  { value: '2', label: 'L2' },
  { value: 'inf', label: 'L∞' },
] as const

function distance(a: Float64Array, b: Float64Array, norm: Norm): number {
  let s = 0
  for (let j = 0; j < a.length; j++) {
    const t = Math.abs(a[j] - b[j])
    if (norm === '1') s += t
    else if (norm === '2') s += t * t
    else s = Math.max(s, t)
  }
  return norm === '2' ? Math.sqrt(s) : s
}

/** Distances from random queries to n uniform points in the unit cube, for dimensions from 1 to 1000. */
export function DistanceConcentration() {
  const n = useParam(500, { min: 50, max: 1000, step: 50 })
  const dimIndex = useParam(6, { min: 0, max: DIMS.length - 1, step: 1 })
  const [norm, setNorm] = useState<Norm>('2')

  const sim = useMemo(() => {
    const g = rng(3)
    return DIMS.map((d) => {
      const pts = Array.from({ length: n.value }, () => Float64Array.from({ length: d }, () => g.uniform()))
      const contrasts: number[] = []
      let first: number[] = []
      for (let q = 0; q < QUERIES; q++) {
        const query = Float64Array.from({ length: d }, () => g.uniform())
        const ds = pts.map((p) => distance(p, query, norm))
        const lo = Math.min(...ds)
        contrasts.push((Math.max(...ds) - lo) / lo)
        if (q === 0) first = ds
      }
      return { d, contrast: contrasts.reduce((a, v) => a + v, 0) / QUERIES, first }
    })
  }, [n.value, norm])

  const chosen = sim[dimIndex.value]
  const mean = chosen.first.reduce((a, v) => a + v, 0) / chosen.first.length
  const scaled = chosen.first.map((v) => v / mean)
  const hist = new Array<number>(BINS).fill(0)
  const width = 3 / BINS
  for (const v of scaled) hist[Math.min(BINS - 1, Math.floor(v / width))] += 1 / scaled.length
  const centres = hist.map((_, i) => (i + 0.5) * width)

  const contrastSeries: XYSeries[] = [
    { name: 'relative contrast', type: 'line', x: LOG_DIMS, y: sim.map((s) => s.contrast), slot: 0 },
    { name: 'chosen d', type: 'scatter', x: [Math.log10(chosen.d)], y: [chosen.contrast], emphasis: true },
  ]
  const histSeries: XYSeries[] = [{ name: 'fraction of points', type: 'bar', x: centres, y: hist, slot: 0 }]

  return (
    <Interactive
      title="Distances concentrate as dimension grows"
      caption="Points uniform in the unit cube [0, 1]^d. Left: the relative contrast (farthest − nearest) / nearest distance from a random query, averaged over 10 queries, against log₁₀ d on a logarithmic scale. It falls roughly like 1/√d. Right: the distances from one query at the chosen d, divided by their mean. In two dimensions they spread from near 0 to twice the mean; in a thousand dimensions nearly every point is at almost the same distance, so the nearest neighbour is barely nearer than the farthest. The L1 norm keeps somewhat more contrast than L2, and L∞ less."
      controls={
        <>
          <ParamSlider label="dimension d" param={dimIndex} format={(v) => String(DIMS[v])} withArrows />
          <ParamSlider label="points n" param={n} format={(v) => String(v)} />
          <ParamChoice label="norm" value={norm} onChange={setNorm} options={NORMS} />
        </>
      }
      readout={
        <>
          <Readout label="d" value={String(chosen.d)} />
          <Readout label="relative contrast" value={formatNumber(chosen.contrast)} />
          <Readout label="nearest / mean" value={formatNumber(Math.min(...scaled))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={contrastSeries} xLabel="log₁₀ d" yLabel="(max − min) / min" yLog height={320} />
        <XYChart
          series={histSeries}
          xLabel="distance / mean distance"
          yLabel="fraction"
          xRange={HIST_RANGE}
          height={320}
        />
      </div>
    </Interactive>
  )
}
