import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

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
  const state = useFigureState({
    dimIndex: float(6, { min: 0, max: DIMS.length - 1, step: 1, label: 'dimension d', format: (v) => String(DIMS[v]) }),
    n: int(500, { min: 50, max: 1000, step: 50, label: 'points n', format: (v) => String(v) }),
    norm: choice<Norm>(NORMS, '2', { label: 'norm' }),
  })

  const sim = useMemo(() => {
    const g = stream(3)
    return DIMS.map((d) => {
      const pts = Array.from({ length: state.n }, () => Float64Array.from({ length: d }, () => uniform(g)))
      const contrasts: number[] = []
      let first: number[] = []
      for (let q = 0; q < QUERIES; q++) {
        const query = Float64Array.from({ length: d }, () => uniform(g))
        const ds = pts.map((p) => distance(p, query, state.norm))
        const lo = Math.min(...ds)
        contrasts.push((Math.max(...ds) - lo) / lo)
        if (q === 0) first = ds
      }
      return { d, contrast: contrasts.reduce((a, v) => a + v, 0) / QUERIES, first }
    })
  }, [state.n, state.norm])

  const chosen = sim[state.dimIndex]
  const mean = chosen.first.reduce((a, v) => a + v, 0) / chosen.first.length
  const scaled = chosen.first.map((v) => v / mean)
  const hist = new Array<number>(BINS).fill(0)
  const width = 3 / BINS
  for (const v of scaled) hist[Math.min(BINS - 1, Math.floor(v / width))] += 1 / scaled.length
  const centres = hist.map((_, i) => (i + 0.5) * width)

  const contrastSeries = [
    { name: 'relative contrast', x: LOG_DIMS, y: sim.map((s) => s.contrast), slot: 0 },
    { name: 'chosen d', x: [Math.log10(chosen.d)], y: [chosen.contrast], emphasis: true },
  ] as const
  const histSeries = [{ name: 'fraction of points', x: centres, y: hist, slot: 0 }] as const

  const xAxis = useAxis({ label: 'log₁₀ d', hold: 'union' })
  const yAxis = useAxis({ label: '(max − min) / min', hold: 'union', log: true })
  const xAxis2 = useAxis({ label: 'distance / mean distance', range: HIST_RANGE })
  const yAxis2 = useAxis({ label: 'fraction', hold: 'union' })
  return (
    <Figure
      title="Distances concentrate as dimension grows"
      state={state}
      caption="Points uniform in the unit cube [0, 1]^d. Left: the relative contrast (farthest − nearest) / nearest distance from a random query, averaged over 10 queries, against log₁₀ d on a logarithmic scale. It falls roughly like 1/√d. Right: the distances from one query at the chosen d, divided by their mean. In two dimensions they spread from near 0 to twice the mean; in a thousand dimensions nearly every point is at almost the same distance, so the nearest neighbour is barely nearer than the farthest. The L1 norm keeps somewhat more contrast than L2, and L∞ less."

      readouts={
        <>
          <Readout label="d" value={String(chosen.d)} />
          <Readout label="relative contrast" value={formatNumber(chosen.contrast)} />
          <Readout label="nearest / mean" value={formatNumber(Math.min(...scaled))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve {...contrastSeries[0]} />
          <Points {...contrastSeries[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Bars {...histSeries[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
