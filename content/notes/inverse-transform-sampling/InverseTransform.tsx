import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { defaults, distribution } from '@/lib/distributions'
import { linspace, rng } from '@/lib/math'

const CHOICES = ['exponential', 'gumbel', 'logistic', 'cauchy'] as const
type Choice = (typeof CHOICES)[number]
const BINS = 40

/**
 * Left: the cdf, with a uniform draw u mapped across to F⁻¹(u). Right: a histogram of many such draws against the pdf.
 */
export function InverseTransform() {
  const [choice, setChoice] = useState<Choice>('exponential')
  const uParam = useParam(0.7, { min: 0.01, max: 0.99, step: 0.01 })
  const u = uParam.value
  const [logN, setLogN] = useState(3)
  const d = distribution(choice)
  const params = useMemo(() => defaults(d), [d])
  const [lo, hi] = d.range(params)
  const quantile = d.quantile!
  const x = quantile(u, params)
  const handles: Handle[] = [{ kind: 'y', at: u, label: 'u', onDrag: uParam.set }]

  const cdfChart = useMemo((): { series: XYSeries[]; segments: { from: [number, number]; to: [number, number] }[] } => {
    const xs = linspace(lo, hi, 300)
    const clamped = Math.min(Math.max(x, lo), hi)
    return {
      series: [
        { name: 'cdf F(x)', type: 'line', x: xs, y: xs.map((v) => d.cdf(v, params)), slot: 0 },
        { name: 'u', type: 'scatter', x: [lo], y: [u], slot: 1 },
        { name: 'x = F⁻¹(u)', type: 'scatter', x: [clamped], y: [0], emphasis: true },
      ],
      // The mapping: across from u to the curve, then down to x.
      segments: [
        { from: [lo, u], to: [clamped, u] },
        { from: [clamped, u], to: [clamped, 0] },
      ],
    }
  }, [d, params, lo, hi, u, x])

  const histogram = useMemo((): { series: XYSeries[]; inRange: number } => {
    const n = Math.round(10 ** logN)
    const r = rng(11)
    const width = (hi - lo) / BINS
    const counts = new Array(BINS).fill(0)
    let inRange = 0
    for (let i = 0; i < n; i++) {
      const sample = quantile(r.uniform(), params)
      const bin = Math.floor((sample - lo) / width)
      if (bin >= 0 && bin < BINS) {
        counts[bin]++
        inRange++
      }
    }
    const centres = counts.map((_, i) => lo + (i + 0.5) * width)
    const xs = linspace(lo, hi, 300)
    return {
      series: [
        { name: 'samples', type: 'bar', x: centres, y: counts.map((c) => c / (n * width)), slot: 0 },
        { name: 'pdf', type: 'line', x: xs, y: xs.map((v) => d.density(v, params)), slot: 1 },
      ],
      inRange: inRange / n,
    }
  }, [d, params, lo, hi, logN, quantile])

  return (
    <Interactive
      title="From uniform numbers to any distribution"
      caption="Move u with the slider or drag the dashed line on the cdf: u maps across to the cdf and down to x = F⁻¹(u). Many uniform draws mapped this way form a histogram that matches the pdf. Only the quantile function changes between distributions."
      controls={
        <>
          <ParamChoice
            label="distribution"
            value={choice}
            onChange={setChoice}
            options={CHOICES.map((c) => ({ value: c, label: distribution(c).name }))}
          />
          <ParamSlider label="uniform draw u" param={uParam} />
          <ParamSlider
            label="samples"
            value={logN}
            onChange={setLogN}
            min={1}
            max={4.5}
            step={0.1}
            format={(v) => Math.round(10 ** v).toLocaleString()}
          />
        </>
      }
      readout={
        <>
          <Readout label="u" value={formatNumber(u)} />
          <Readout label="x = F⁻¹(u)" value={formatNumber(x)} />
          <Readout label="samples inside the plotted range" value={`${(100 * histogram.inRange).toFixed(1)}%`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          series={cdfChart.series}
          segments={cdfChart.segments}
          xRange={[lo, hi]}
          yRange={[0, 1]}
          xLabel="x"
          yLabel="u = F(x)"
          handles={handles}
        />
        <XYChart
          height={300}
          series={histogram.series}
          xRange={[lo, hi]}
          yRange={[0, undefined]}
          xLabel="x"
          yLabel="density"
        />
      </div>
    </Interactive>
  )
}
