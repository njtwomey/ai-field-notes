import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  Segments,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { Cauchy, Exponential, Gumbel, Logistic } from 'aifn/probability/distributions'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { stream, uniform } from 'aifn/foundation/random'

/** Four standard laws with closed-form quantile functions, each with the x range its panels show. */
const FAMILIES = {
  exponential: { name: 'Exponential', law: Exponential(1), range: [0, 6] },
  gumbel: { name: 'Gumbel', law: Gumbel(0, 1), range: [-3, 7] },
  logistic: { name: 'Logistic', law: Logistic(0, 1), range: [-8, 8] },
  cauchy: { name: 'Cauchy', law: Cauchy(0, 1), range: [-10, 10] },
} as const
type Choice = keyof typeof FAMILIES
const BINS = 40

/**
 * Left: the cdf, with a uniform draw u mapped across to F⁻¹(u). Right: a histogram of many such draws against the pdf.
 */
export function InverseTransform() {
  const state = useFigureState({
    family: choice<Choice>(
      (Object.keys(FAMILIES) as Choice[]).map((c) => ({ value: c, label: FAMILIES[c].name })),
      'exponential',
      { label: 'distribution' },
    ),
    u: slider(0.01, 0.99, 0.7, { step: 0.01, label: 'uniform draw u' }),
    n: int(1000, { ge: 10, le: 30000, scale: 'log10', suggestions: [100, 1000, 10000, 30000], label: 'samples' }),
  })
  const { u, n } = state
  const d = FAMILIES[state.family].law
  const [lo, hi] = FAMILIES[state.family].range
  const quantile = (p: number) => d.quantile(p)
  const x = quantile(u)

  const cdfChart = useMemo((): {
    series: SeriesSpec[]
    segments: { from: [number, number]; to: [number, number] }[]
  } => {
    const xs = toFlat(linspace(lo, hi, 300))
    const clamped = Math.min(Math.max(x, lo), hi)
    return {
      series: [
        { name: 'cdf F(x)', type: 'line', x: xs, y: xs.map((v) => d.cdf(v)), slot: 0 },
        { name: 'u', type: 'scatter', x: [lo], y: [u], slot: 1 },
        { name: 'x = F⁻¹(u)', type: 'scatter', x: [clamped], y: [0], emphasis: true },
      ],
      // The mapping: across from u to the curve, then down to x.
      segments: [
        { from: [lo, u], to: [clamped, u] },
        { from: [clamped, u], to: [clamped, 0] },
      ],
    }
  }, [d, lo, hi, u, x])

  const histogram = useMemo((): { series: SeriesSpec[]; inRange: number } => {
    const r = stream(11)
    const width = (hi - lo) / BINS
    const counts = new Array(BINS).fill(0)
    let inRange = 0
    for (let i = 0; i < n; i++) {
      const sample = d.quantile(uniform(r))
      const bin = Math.floor((sample - lo) / width)
      if (bin >= 0 && bin < BINS) {
        counts[bin]++
        inRange++
      }
    }
    const centres = counts.map((_, i) => lo + (i + 0.5) * width)
    const xs = toFlat(linspace(lo, hi, 300))
    return {
      series: [
        { name: 'samples', type: 'bar', x: centres, y: counts.map((c) => c / (n * width)), slot: 0 },
        { name: 'pdf', type: 'line', x: xs, y: xs.map((v) => d.prob(v)), slot: 1 },
      ],
      inRange: inRange / n,
    }
  }, [d, lo, hi, n])

  const xAxis = useAxis({ label: 'x', range: [lo, hi] })
  const yAxis = useAxis({ label: 'u = F(x)', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'x', range: [lo, hi] })
  const yAxis2 = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="From uniform numbers to any distribution"
      caption="Move u with the slider or drag the dashed line on the cdf: u maps across to the cdf and down to x = F⁻¹(u). Many uniform draws mapped this way form a histogram that matches the pdf. Only the quantile function changes between distributions."
      state={state}
      readouts={
        <>
          <Readout label="u" value={formatNumber(u)} />
          <Readout label="x = F⁻¹(u)" value={formatNumber(x)} />
          <Readout label="samples inside the plotted range" value={`${(100 * histogram.inRange).toFixed(1)}%`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(cdfChart.series)}
          <Segments segments={cdfChart.segments} />
          <Handle kind="y" at={u} label="u" onDrag={(v) => state.set('u', v)} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers(histogram.series)}
        </Plot>
      </div>
    </Figure>
  )
}
