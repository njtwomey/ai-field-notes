import { useMemo } from 'react'
import { extent, histogram, mean, quantile, variance } from 'aifn/stats'
import { linspace, toFlat } from 'aifn/tensor'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from './format'
import type { FrameProps } from './frame'

/** Reference moments shown beside the sample's own. */
export type ReferenceMoments = { mean?: number; variance?: number }

export type SamplesViewProps = FrameProps &
  (
    | {
        /** Continuous draws, drawn as a density histogram. */
        kind?: 'histogram'
        samples: ArrayLike<number>
        /** The reference density, drawn as a line over the histogram. */
        density?: (x: number) => number
        /** The reference cdf; when given, the Kolmogorov–Smirnov distance is shown. */
        cdf?: (x: number) => number
        /** Histogram range; default the 0.5% to 99.5% sample quantiles (heavy tails would flatten the plot). */
        range?: [number, number]
        bins?: number
        reference?: ReferenceMoments
        xLabel?: string
      }
    | {
        /** Integer draws, drawn as relative frequencies against a reference mass function. */
        kind: 'discrete'
        samples: ArrayLike<number>
        pmf?: (k: number) => number
        reference?: ReferenceMoments
        xLabel?: string
      }
    | {
        /** Two-dimensional draws, drawn as a scatter, with optional reference marks (e.g. a covariance ellipse). */
        kind: 'scatter'
        x: ArrayLike<number>
        y: ArrayLike<number>
        overlay?: XYSeries[]
        xRange?: [number, number]
        yRange?: [number, number]
        xLabel?: string
        yLabel?: string
        /** Most points drawn; further draws are counted but not drawn. Default 4000. */
        maxPoints?: number
      }
  )

/** Non-finite values (e.g. a density's pole at an end of its support) become gaps in the line. */
const finiteOrGap = (v: number) => (Number.isFinite(v) ? v : NaN)

const withExact = (value: number, exact: number | undefined) =>
  `${formatValue(value)}${exact !== undefined ? ` (exact ${formatValue(exact)})` : ''}`

function MomentReadouts({ samples, reference }: { samples: ArrayLike<number>; reference?: ReferenceMoments }) {
  const m = useMemo(() => ({ mean: mean(samples), variance: variance(samples, { sample: true }) }), [samples])
  return (
    <>
      <Readout label="n" value={samples.length} />
      <Readout label="mean" value={withExact(m.mean, reference?.mean)} />
      <Readout label="variance" value={withExact(m.variance, reference?.variance)} />
    </>
  )
}

/**
 * Draws from a sampler against the distribution they should follow, as one figure: a density histogram under the
 * reference density, a relative-frequency plot against a mass function, or a scatter for pairs. Readouts give the
 * sample moments against the exact ones and, when a cdf is given, the Kolmogorov–Smirnov distance with its 5%
 * critical value 1.36/√n.
 */
export function SamplesView(props: SamplesViewProps) {
  if (props.kind === 'discrete') return <DiscreteSamples {...props} />
  if (props.kind === 'scatter') return <ScatterSamples {...props} />
  return <HistogramSamples {...props} />
}

function HistogramSamples({
  samples,
  density,
  cdf,
  range,
  bins = 60,
  reference,
  xLabel = 'x',
  title = 'Samples against the density',
  readouts,
  ...frame
}: Extract<SamplesViewProps, { samples: ArrayLike<number> }> & { kind?: 'histogram' }) {
  const [lo0, hi0] = range ?? []
  const { series, ks } = useMemo(() => {
    const [lo, hi] = [lo0 ?? quantile(samples, 0.005), hi0 ?? quantile(samples, 0.995)]
    const h = histogram(samples, { bins, range: [lo, hi] })
    // Density over all n draws, so the bars integrate to the share that falls in the range.
    const scale = (samples.length - h.dropped) / samples.length
    const out: XYSeries[] = [
      {
        name: 'samples',
        type: 'bar',
        x: Array.from(h.counts, (_, i) => (h.edges[i] + h.edges[i + 1]) / 2),
        y: Array.from(h.density, (d) => d * scale),
      },
    ]
    if (density) {
      const xs = toFlat(linspace(lo, hi, 301))
      out.push({ name: 'density', type: 'line', x: xs, y: xs.map((x) => finiteOrGap(density(x))) })
    }
    // TODO(tests): the Kolmogorov–Smirnov statistic belongs in aifn (a goodness-of-fit test); computed here until then.
    let d = NaN
    if (cdf) {
      const sorted = Float64Array.from(samples).sort()
      const n = sorted.length
      d = 0
      for (let i = 0; i < n; i++) {
        const f = cdf(sorted[i])
        d = Math.max(d, Math.abs((i + 1) / n - f), Math.abs(i / n - f))
      }
    }
    return { series: out, ks: d }
  }, [samples, density, cdf, lo0, hi0, bins])
  return (
    <Figure
      title={title}
      {...frame}
      readouts={
        <>
          {readouts}
          <MomentReadouts samples={samples} reference={reference} />
          {cdf && (
            <Readout
              label="KS distance"
              value={`${formatValue(ks)} (5% critical ${formatValue(1.36 / Math.sqrt(samples.length))})`}
            />
          )}
        </>
      }
    >
      <XYChart series={series} xLabel={xLabel} yLabel="density" />
    </Figure>
  )
}

function DiscreteSamples({
  samples,
  pmf,
  reference,
  xLabel = 'k',
  title = 'Frequencies against the mass function',
  readouts,
  ...frame
}: Extract<SamplesViewProps, { kind: 'discrete' }>) {
  const series = useMemo(() => {
    const [lo, hi] = extent(samples)
    // One bin per integer: edges at the half-integers.
    const h = histogram(samples, { bins: toFlat(linspace(lo - 0.5, hi + 0.5, hi - lo + 2)) })
    const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
    const out: XYSeries[] = [
      { name: 'samples', type: 'bar', x: ks, y: Array.from(h.counts, (c) => c / samples.length) },
    ]
    if (pmf) out.push({ name: 'pmf', type: 'scatter', x: ks, y: ks.map(pmf) })
    return out
  }, [samples, pmf])
  return (
    <Figure
      title={title}
      {...frame}
      readouts={
        <>
          {readouts}
          <MomentReadouts samples={samples} reference={reference} />
        </>
      }
    >
      <XYChart series={series} xLabel={xLabel} yLabel="probability" integerX />
    </Figure>
  )
}

function ScatterSamples({
  x,
  y,
  overlay,
  xRange,
  yRange,
  xLabel = 'x',
  yLabel = 'y',
  maxPoints = 4000,
  title = 'Draws',
  readouts,
  ...frame
}: Extract<SamplesViewProps, { kind: 'scatter' }>) {
  const series = useMemo(() => {
    const n = Math.min(x.length, maxPoints)
    const out: XYSeries[] = [
      {
        name: 'samples',
        type: 'scatter',
        x: Array.from({ length: n }, (_, i) => x[i]),
        y: Array.from({ length: n }, (_, i) => y[i]),
      },
    ]
    return overlay ? [...out, ...overlay] : out
  }, [x, y, overlay, maxPoints])
  const equal = xRange !== undefined && yRange !== undefined
  return (
    <Figure
      title={title}
      {...frame}
      readouts={
        <>
          {readouts}
          <Readout label="n" value={x.length} />
          {x.length > maxPoints && <Readout label="drawn" value={maxPoints} />}
        </>
      }
    >
      <XYChart series={series} xLabel={xLabel} yLabel={yLabel} xRange={xRange} yRange={yRange} equalAspect={equal} />
    </Figure>
  )
}
