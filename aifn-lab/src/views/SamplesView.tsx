import { useMemo, type ReactNode } from 'react'
import { extent, histogram, quantile } from 'aifn/probability/stats'
import { ksTest } from 'aifn/probability/tests'
import { linspace, mean, tensor, toFlat, variance } from 'aifn/foundation/tensor'
import { PanelSlot } from '@lab/layout'
import { Bars, Curve, Plot, Points, Readout, useAxis } from '@lab/viz'
import { formatValue } from './format'
import { histogramBars } from './histogram'

/** Reference moments shown beside the sample's own. */
export type ReferenceMoments = { mean?: number; variance?: number }

export type SamplesPanelProps =
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
      /** Reference layers drawn over the draws (e.g. a covariance ellipse as a `Curve`). */
      children?: ReactNode
      xRange?: [number, number]
      yRange?: [number, number]
      xLabel?: string
      yLabel?: string
      /** Most points drawn; further draws are counted but not drawn. Default 4000. */
      maxPoints?: number
    }

/** Non-finite values (e.g. a density's pole at an end of its support) become gaps in the line. */
const finiteOrGap = (v: number) => (Number.isFinite(v) ? v : NaN)

const withExact = (value: number, exact: number | undefined) =>
  `${formatValue(value)}${exact !== undefined ? ` (exact ${formatValue(exact)})` : ''}`

function MomentReadouts({ samples, reference }: { samples: ArrayLike<number>; reference?: ReferenceMoments }) {
  const m = useMemo(() => {
    const x = tensor(samples)
    return { mean: mean(x), variance: variance(x, null, false, 1) }
  }, [samples])
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
 * sample moments against the exact ones and, when a cdf is given, the Kolmogorov–Smirnov distance with its p-value
 * (`ksTest` from `aifn/probability/tests`).
 */
export function SamplesPanel(props: SamplesPanelProps) {
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
}: Extract<SamplesPanelProps, { samples: ArrayLike<number> }> & { kind?: 'histogram' }) {
  const [lo0, hi0] = range ?? []
  const { bars, line, ks } = useMemo(() => {
    const [lo, hi] = [lo0 ?? quantile(samples, 0.005), hi0 ?? quantile(samples, 0.995)]
    const h = histogram(samples, { bins, range: [lo, hi] })
    const b = histogramBars(h)
    // Density over all n draws, so the bars integrate to the share that falls in the range.
    const scale = (samples.length - h.dropped) / samples.length
    const xs = density ? toFlat(linspace(lo, hi, 301)) : null
    const ks = cdf && samples.length > 0 ? ksTest(samples, cdf) : null
    return {
      bars: { x: b.x, edges: b.edges, y: b.density.map((d) => d * scale) },
      line: xs && density ? { x: xs, y: xs.map((x) => finiteOrGap(density(x))) } : null,
      ks,
    }
  }, [samples, density, cdf, lo0, hi0, bins])
  const xAxis = useAxis({ label: xLabel })
  const yAxis = useAxis({ label: 'density' })
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <MomentReadouts samples={samples} reference={reference} />
          {cdf && (
            <Readout
              label="KS distance"
              value={ks ? `${formatValue(ks.statistic)} (p = ${formatValue(ks.pValue)})` : '–'}
            />
          )}
        </>
      </PanelSlot>
      <Plot x={xAxis} y={yAxis}>
        <Bars name="samples" x={bars.x} y={bars.y} edges={bars.edges} />
        {line && <Curve name="density" x={line.x} y={line.y} />}
      </Plot>
    </>
  )
}

function DiscreteSamples({ samples, pmf, reference, xLabel = 'k' }: Extract<SamplesPanelProps, { kind: 'discrete' }>) {
  const { ks, freq } = useMemo(() => {
    const [lo, hi] = extent(samples)
    // One bin per integer: edges at the half-integers.
    const h = histogram(samples, { bins: toFlat(linspace(lo - 0.5, hi + 0.5, hi - lo + 2)) })
    const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
    return { ks, freq: toFlat(h.counts).map((c) => c / samples.length) }
  }, [samples])
  const mass = useMemo(() => (pmf ? ks.map(pmf) : null), [ks, pmf])
  const xAxis = useAxis({ label: xLabel, integer: true })
  const yAxis = useAxis({ label: 'probability' })
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <MomentReadouts samples={samples} reference={reference} />
        </>
      </PanelSlot>
      <Plot x={xAxis} y={yAxis}>
        <Bars name="samples" x={ks} y={freq} />
        {mass && <Points name="pmf" x={ks} y={mass} />}
      </Plot>
    </>
  )
}

function ScatterSamples({
  x,
  y,
  children,
  xRange,
  yRange,
  xLabel = 'x',
  yLabel = 'y',
  maxPoints = 4000,
}: Extract<SamplesPanelProps, { kind: 'scatter' }>) {
  const drawn = useMemo(() => {
    const n = Math.min(x.length, maxPoints)
    return { x: Array.from({ length: n }, (_, i) => x[i]), y: Array.from({ length: n }, (_, i) => y[i]) }
  }, [x, y, maxPoints])
  const equal = xRange !== undefined && yRange !== undefined
  const xAxis = useAxis({ label: xLabel, range: xRange })
  const yAxis = useAxis({ label: yLabel, range: yRange, equal: equal ? xAxis : undefined })
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <Readout label="n" value={x.length} />
          {x.length > maxPoints && <Readout label="drawn" value={maxPoints} />}
        </>
      </PanelSlot>
      <Plot x={xAxis} y={yAxis}>
        <Points name="samples" x={drawn.x} y={drawn.y} />
        {children}
      </Plot>
    </>
  )
}
