import { useMemo } from 'react'
import type { Distribution, Multivariate, Univariate } from 'aifn-compute/probability/distributions'
import { stream } from 'aifn-compute/foundation/random'
import { histogram } from 'aifn-compute/probability/stats'
import { reshape, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { PanelSlot } from 'aifn-render/layout'
import { Bars, Curve, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'
import { distributionRange } from './distribution-range'
import { formatValue } from './format'
import { histogramBars } from './histogram'
import { registerKind, registerView } from './registry'

export type DistributionPanelProps = {
  /** A univariate distribution (a batch draws one curve per member, up to eight), or a bivariate one (event [2]). */
  distribution: Distribution
  /** The x range (univariate) or [lo, hi] for both axes (bivariate); default from the quantiles or the moments. */
  range?: [number, number]
  /** The y range of a bivariate view; default from the moments. */
  yRange?: [number, number]
  /** Draws shown against the density (default 2000; 0 for none). Only for an unbatched distribution. */
  samples?: number
  /** Seed of the stream the draws come from (default 1). */
  seed?: number | string
  xLabel?: string
  yLabel?: string
  /**
   * Refit the axes as the distribution changes (default true). False, for parameter exploration: x holds `range` (pass
   * the range of the family's default parameters), the density's y axis grows to hold every curve seen (`union`), and
   * the cdf stays on [0, 1]; `axisKey` (e.g. the family) refits.
   */
  rescaleOnChange?: boolean
  axisKey?: string | number
}

/** A Value as numbers (a number becomes a one-element list). */
const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}

/** A derived quantity for a readout: '—' when the family has no closed form (the method throws). */
function attempt(f: () => unknown): string {
  try {
    const values = numbers(f() as Value)
    return values.map(formatValue).join(', ')
  } catch {
    return '—'
  }
}

/** Tick labels at integers only, for a discrete variable. */

/** Non-finite values become gaps in lines. */
const gap = (v: number) => (Number.isFinite(v) ? v : NaN)

/**
 * A distribution object as one figure. Univariate: the density (or mass) with a histogram of draws under it, the cdf
 * against the empirical cdf of the draws, and readouts of the moments, entropy, mode, support, shapes and the
 * Kolmogorov–Smirnov distance of the draws. A batch draws one curve per member. Bivariate (event shape [2]): the
 * density as a heatmap with draws over it, and the mean and covariance. Quantities with no closed form show '—'.
 */
export function DistributionPanel(props: DistributionPanelProps) {
  const d = props.distribution
  if (d.eventShape.length === 1 && d.eventShape[0] === 2 && d.batchShape.length === 0)
    return <BivariatePanel {...props} distribution={d as Multivariate} />
  if (d.eventShape.length !== 0) {
    return (
      <div className="text-sm text-muted-foreground">
        DistributionPanel draws univariate and bivariate distributions; this one has event shape [
        {d.eventShape.join(', ')}].
      </div>
    )
  }
  return <UnivariatePanel {...props} distribution={d as Univariate} />
}

/** The default title of a distribution's figure: its name and parameters. */
const titleOf = (d: Distribution) =>
  d.eventShape.length === 0 ? `${d.name} (${Object.keys(d.params).join(', ')})` : d.name

const isDistribution = (o: unknown): o is Distribution =>
  typeof o === 'object' &&
  o !== null &&
  typeof (o as Distribution).prob === 'function' &&
  typeof (o as Distribution).sample === 'function' &&
  Array.isArray((o as Distribution).eventShape)

registerKind('distribution', isDistribution)
registerView<Distribution>({
  key: 'distribution/density',
  kind: 'distribution',
  description:
    'The density or mass with a histogram of draws, the cdf against the empirical cdf, and the moments (univariate); the density as a heatmap with draws (bivariate).',
  title: titleOf,
  render: (d) => <DistributionPanel distribution={d} />,
})

function UnivariatePanel({
  distribution: d,
  range,
  samples = 2000,
  seed = 1,
  xLabel = 'x',
  rescaleOnChange = true,
  axisKey,
}: DistributionPanelProps & { distribution: Univariate }) {
  const batch = d.batchShape
  const members = batch.reduce((a, b) => a * b, 1)
  const shown = Math.min(members, 8)
  const [lo, hi] = useMemo(() => range ?? distributionRange(d), [d, range])
  const xs = useMemo(() => {
    if (d.discrete) {
      const a = Math.ceil(lo)
      const b = Math.max(a, Math.min(Math.floor(hi), a + 400))
      return Array.from({ length: b - a + 1 }, (_, i) => a + i)
    }
    return Array.from({ length: 401 }, (_, i) => lo + ((hi - lo) * i) / 400)
  }, [d.discrete, lo, hi])
  // Evaluate on a grid shaped [n, 1, …, 1] so that it broadcasts against the batch: the result is [n, ...batch].
  const curves = useMemo(() => {
    const grid = reshape(tensor(xs), [xs.length, ...batch.map(() => 1)]) as Tensor
    const column = (v: Value) => {
      const flat = numbers(v)
      return Array.from({ length: shown }, (_, m) => xs.map((_, i) => gap(flat[i * members + m] ?? flat[i])))
    }
    return { density: column(d.prob(grid)), cdf: column(d.cdf(grid)) }
  }, [d, xs, batch, members, shown])
  const draws = useMemo(() => {
    if (samples <= 0 || members !== 1) return null
    return numbers(d.sample(stream(seed), { shape: [samples] }))
  }, [d, samples, seed, members])
  const ks = useMemo(() => {
    if (!draws || d.discrete) return NaN
    const sorted = Float64Array.from(draws).sort()
    const cdf = numbers(d.cdf(tensor(Array.from(sorted))))
    let dist = 0
    for (let i = 0; i < sorted.length; i++)
      dist = Math.max(dist, Math.abs((i + 1) / sorted.length - cdf[i]), Math.abs(i / sorted.length - cdf[i]))
    return dist
  }, [d, draws])
  const bars = useMemo(() => {
    if (!draws) return null
    if (d.discrete) {
      const counts = new Map<number, number>()
      for (const v of draws) counts.set(v, (counts.get(v) ?? 0) + 1)
      return { name: 'draws (frequency)', x: xs, y: xs.map((k) => (counts.get(k) ?? 0) / draws.length) }
    }
    const h = histogram(draws, { bins: 60, range: [lo, hi] })
    const b = histogramBars(h)
    return { name: 'draws (density)', x: b.x, y: toFlat(h.density), edges: b.edges }
  }, [draws, d.discrete, xs, lo, hi])
  const empirical = useMemo(() => {
    if (!draws) return null
    const sorted = Float64Array.from(draws).sort()
    const step = Math.max(1, Math.floor(sorted.length / 400))
    const ex: number[] = []
    const ey: number[] = []
    for (let i = 0; i < sorted.length; i += step) {
      ex.push(sorted[i])
      ey.push((i + 1) / sorted.length)
    }
    return { x: ex, y: ey }
  }, [draws])
  const xAxis = useAxis({ label: xLabel, range: [lo, hi], integer: d.discrete })
  const densityAxis = useAxis({
    label: d.discrete ? 'probability' : 'density',
    hold: rescaleOnChange ? undefined : 'union',
    key: axisKey,
  })
  const cdfAxis = useAxis({ label: 'cdf', range: [0, 1] })
  const memberName = (m: number, own: string) => (shown > 1 ? `member ${m}` : own)
  const support = d.support
  const supportText =
    'lower' in support
      ? `${support.type} [${attempt(() => support.lower)}; ${attempt(() => support.upper)}]`
      : support.type
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <Readout label="mean" value={attempt(() => d.mean())} />
          <Readout label="variance" value={attempt(() => d.variance())} />
          <Readout label="sd" value={attempt(() => d.stddev())} />
          <Readout label="entropy (nats)" value={attempt(() => d.entropy())} />
          <Readout label="mode" value={attempt(() => d.mode())} />
          <Readout label="support" value={supportText} />
          <Readout label="batch × event" value={`[${batch.join(', ')}] × []`} />
          {Number.isFinite(ks) && draws && (
            <Readout
              label="KS distance of draws"
              value={`${formatValue(ks)} (5% critical ${formatValue(1.36 / Math.sqrt(draws.length))})`}
            />
          )}
        </>
      </PanelSlot>
      <Plots rows={2} heights={[55, 45]} hoverGroup>
        <Plot x={xAxis} y={densityAxis}>
          {bars && <Bars name={bars.name} x={bars.x} y={bars.y} edges={bars.edges} muted />}
          {curves.density.map((y, m) =>
            d.discrete ? (
              <Points key={m} name={memberName(m, 'mass')} x={xs} y={y} slot={m} />
            ) : (
              <Curve key={m} name={memberName(m, 'density')} x={xs} y={y} slot={m} />
            ),
          )}
        </Plot>
        <Plot x={xAxis} y={cdfAxis}>
          {curves.cdf.map((y, m) => (
            <Curve key={m} name={memberName(m, 'cdf')} x={xs} y={y} slot={m} />
          ))}
          {empirical && <Curve name="empirical cdf" x={empirical.x} y={empirical.y} dashed slot={1} />}
        </Plot>
      </Plots>
    </>
  )
}

function BivariatePanel({
  distribution: d,
  range,
  yRange,
  samples = 1000,
  seed = 1,
  xLabel = 'x₁',
  yLabel = 'x₂',
}: DistributionPanelProps & { distribution: Multivariate }) {
  const [xr, yr] = useMemo((): [[number, number], [number, number]] => {
    if (range && yRange) return [range, yRange]
    try {
      const m = numbers(d.mean())
      const s = numbers(d.stddev())
      return [range ?? [m[0] - 3.5 * s[0], m[0] + 3.5 * s[0]], yRange ?? [m[1] - 3.5 * s[1], m[1] + 3.5 * s[1]]]
    } catch {
      return [range ?? [0, 1], yRange ?? [0, 1]]
    }
  }, [d, range, yRange])
  const n = 81
  const { gx, gy, z } = useMemo(() => {
    const gx = Array.from({ length: n }, (_, i) => xr[0] + ((xr[1] - xr[0]) * i) / (n - 1))
    const gy = Array.from({ length: n }, (_, i) => yr[0] + ((yr[1] - yr[0]) * i) / (n - 1))
    const points: number[] = []
    for (const y of gy) for (const x of gx) points.push(x, y)
    const p = numbers(d.prob(tensor(points, [n * n, 2])))
    const z = gy.map((_, i) => gx.map((_, j) => gap(p[i * n + j])))
    return { gx, gy, z }
  }, [d, xr, yr])
  const draws = useMemo(() => {
    if (samples <= 0) return null
    const flat = numbers(d.sample(stream(seed), { shape: [samples] }))
    // Only the draws inside the grid, so the axes end at the density's edges.
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i + 1 < flat.length; i += 2) {
      const [a, b] = [flat[i], flat[i + 1]]
      if (a < xr[0] || a > xr[1] || b < yr[0] || b > yr[1]) continue
      x.push(a)
      y.push(b)
    }
    return { x, y }
  }, [d, samples, seed, xr, yr])
  const xAxis = useAxis({ label: xLabel })
  const yAxis = useAxis({ label: yLabel, equal: xAxis })
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <Readout label="mean" value={attempt(() => d.mean())} />
          <Readout label="covariance" value={attempt(() => d.covariance())} />
          <Readout label="entropy (nats)" value={attempt(() => d.entropy())} />
          <Readout label="batch × event" value={`[${d.batchShape.join(', ')}] × [${d.eventShape.join(', ')}]`} />
        </>
      </PanelSlot>
      <Plot x={xAxis} y={yAxis}>
        <Raster x={gx} y={gy} z={z} valueLabel="density" />
        {draws && <Points name="draws" x={draws.x} y={draws.y} slot={1} thin />}
      </Plot>
    </>
  )
}
