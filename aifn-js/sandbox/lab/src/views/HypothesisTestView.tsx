import { useMemo, type ReactNode } from 'react'
import { tensor, toFlat, unwrap, type Value } from 'aifn/foundation/tensor'
import type { Univariate } from 'aifn/probability/distributions'
import { lowerTail, rejectionRegion, upperTail, type TestResult } from 'aifn/probability/tests'
import { PanelSlot } from '@lab/layout'
import { slider } from '@lab/state'
import { Annotation, Area, Bars, Curve, Plot, Plots, Points, Readout, Segments, useAxis } from '@lab/viz'
import { formatValue } from './format'
import { registerView } from './registry'

export type HypothesisTestPanelProps = {
  /** A test's result (`aifn/probability/tests`): its statistic, null law, tail rule, interval and effect size. */
  result: TestResult
  /** The significance level whose rejection region is shaded (default 0.05). */
  alpha?: number
  /** The statistic axis; default from the null law's 0.05% and 99.95% quantiles, widened to hold the statistic. */
  range?: [number, number]
  /** Hold the axes while the inputs change (the key refits them, e.g. the test's name). */
  axisKey?: string | number
  /** A `Plot` drawn as the first row above the null law, e.g. the data the test was computed from. */
  top?: ReactNode
}

const num = (v: Value) => unwrap(v) as number
const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : v > 0 ? '∞' : '−∞')
const fp = (p: number) => (p < 1e-4 ? p.toExponential(2) : formatValue(Number(p.toPrecision(3))))

/** The statistic axis: the null law's central 99.9% (its support where bounded), widened to hold the statistic. */
function rangeOf(law: Univariate, t: number): [number, number] {
  let lo = num(law.quantile(5e-4))
  let hi = num(law.isf(5e-4))
  if (!Number.isFinite(lo)) lo = num(law.quantile(0.01))
  if (!Number.isFinite(hi)) hi = num(law.isf(0.01))
  if (Number.isFinite(t)) {
    const pad = 0.08 * (hi - lo || 1)
    lo = Math.min(lo, t - pad)
    hi = Math.max(hi, t + pad)
  }
  return law.discrete ? [Math.floor(lo) - 0.5, Math.ceil(hi) + 0.5] : [lo, hi]
}

/** Clip a region's intervals to the axis. */
const clip = (r: [number, number][], [lo, hi]: [number, number]) =>
  r.map(([a, b]) => [Math.max(a, lo), Math.min(b, hi)] as [number, number]).filter(([a, b]) => b >= a)

/**
 * A hypothesis test as one figure: the statistic's null law (density, or mass for an exact test), the rejection region
 * at α shaded, the p-value's area (the outcomes at least as extreme as the one observed), the observed statistic, and
 * below it the confidence interval for the estimand against its null value. Readouts give the statistic, degrees of
 * freedom, p-value, the decision at α, the interval and the effect size.
 */
export function HypothesisTestPanel({ result: r, alpha = 0.05, range, axisKey, top }: HypothesisTestPanelProps) {
  const law = r.null
  const t = r.statistic
  const [lo, hi] = useMemo(() => range ?? rangeOf(law, t), [range, law, t])
  const region = useMemo(() => clip(rejectionRegion(r, alpha), [lo, hi]), [r, alpha, lo, hi])

  // The outcomes at least as extreme as t, as intervals (continuous) or a predicate (discrete).
  const extreme = useMemo(() => {
    if (r.tail === 'upper') return [[t, hi]] as [number, number][]
    if (r.tail === 'lower') return [[lo, t]] as [number, number][]
    if (r.tail === 'both') {
      // The doubled smaller tail: the tail beyond t and the opposite tail of equal probability.
      const below = lowerTail(law, t)
      const above = upperTail(law, t)
      return below <= above
        ? ([
            [lo, t],
            [num(law.isf(below)), hi],
          ] as [number, number][])
        : ([
            [lo, num(law.quantile(above))],
            [t, hi],
          ] as [number, number][])
    }
    return [] as [number, number][]
  }, [r.tail, law, t, lo, hi])

  const curve = useMemo(() => {
    if (law.discrete) {
      const ks: number[] = []
      for (let k = Math.ceil(lo); k <= Math.floor(hi) && ks.length < 2001; k++) ks.push(k)
      const p = numbers(law.prob(tensor(ks)))
      const pt = num(law.prob(Math.round(t))) * (1 + 1e-7)
      const inRegion = (k: number) => region.some(([a, b]) => k >= a - 1e-9 && k <= b + 1e-9)
      const inP = (k: number) =>
        r.tail === 'likelihood'
          ? p[ks.indexOf(k)] <= pt
          : extreme.some(([a, b]) => k >= Math.min(a, b) - 1e-9 && k <= Math.max(a, b) + 1e-9)
      const pick = (f: (k: number) => boolean) => ({ x: ks.filter(f), y: ks.flatMap((k, i) => (f(k) ? [p[i]] : [])) })
      return { kind: 'mass' as const, all: { x: ks, y: p }, region: pick(inRegion), pValue: pick(inP) }
    }
    const xs = Array.from({ length: 401 }, (_, i) => lo + ((hi - lo) * i) / 400)
    const y = numbers(law.prob(tensor(xs))).map((v) => (Number.isFinite(v) ? v : NaN))
    // A shaded piece of the density over [a, b], on its own grid so the edges are exact.
    const piece = ([a, b]: [number, number]) => {
      const px = Array.from({ length: 81 }, (_, i) => a + ((b - a) * i) / 80)
      const py = numbers(law.prob(tensor(px))).map((v) => (Number.isFinite(v) ? v : 0))
      return { x: px, y: py }
    }
    return {
      kind: 'density' as const,
      all: { x: xs, y },
      region: region.map(piece),
      pValue: clip(extreme, [lo, hi]).map(piece),
    }
  }, [law, lo, hi, region, extreme, r.tail, t])

  const xAxis = useAxis({ label: `statistic ${r.symbol}`, range: [lo, hi], integer: law.discrete, key: axisKey })
  const yAxis = useAxis({ label: law.discrete ? 'null probability' : 'null density', key: axisKey })

  const ci = r.ci
  const ciRange = useMemo<[number, number] | null>(() => {
    if (!ci) return null
    const ends = [ci.lower, ci.upper, r.estimate ?? NaN, r.nullValue ?? NaN].filter(Number.isFinite)
    if (!ends.length) return null
    const a = Math.min(...ends)
    const b = Math.max(...ends)
    const pad = 0.25 * (b - a || Math.abs(a) || 1)
    return [a - pad, b + pad]
  }, [ci, r.estimate, r.nullValue])
  const ciAxis = useAxis({ label: r.estimand ?? 'estimate', range: ciRange ?? [0, 1], key: axisKey })
  const ciY = useAxis({ range: [-1, 1], label: ' ' })

  const reject = r.pValue <= alpha
  const df = r.df === undefined ? null : Array.isArray(r.df) ? r.df.map(f3).join(', ') : f3(r.df as number)
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <Readout label="test" value={r.method} />
          <Readout label={`statistic ${r.symbol}`} value={f3(t)} />
          {df && <Readout label="df" value={df} />}
          <Readout label="p-value" value={fp(r.pValue)} />
          <Readout label={`at α = ${alpha}`} value={reject ? 'reject H₀' : 'do not reject H₀'} />
          {ci && (
            <Readout
              label={`${Math.round(ci.level * 100)}% CI (${r.estimand ?? 'estimate'})`}
              value={`[${f3(ci.lower)}, ${f3(ci.upper)}]`}
            />
          )}
          {r.effectSize && <Readout label={r.effectSize.name} value={f3(r.effectSize.value)} />}
          <Readout label="n" value={r.n} />
        </>
      </PanelSlot>
      <Plots
        rows={(top ? 1 : 0) + 1 + (ci && ciRange ? 1 : 0)}
        heights={[...(top ? [36] : []), 64, ...(ci && ciRange ? [20] : [])]}
      >
        {top}
        <Plot x={xAxis} y={yAxis}>
          {curve.kind === 'density' ? (
            <>
              {curve.region.map((p, i) => (
                <Area
                  key={`r${i}`}
                  name={`rejection region (α = ${alpha})`}
                  x={p.x}
                  y={p.y}
                  tone="destructive"
                  opacity={0.3}
                  line={false}
                />
              ))}
              {curve.pValue.map((p, i) => (
                <Area key={`p${i}`} name="p-value area" x={p.x} y={p.y} slot={0} opacity={0.45} line={false} />
              ))}
              <Curve name={`null law (${law.name})`} x={curve.all.x} y={curve.all.y} emphasis />
            </>
          ) : (
            <>
              <Bars name={`null law (${law.name})`} x={curve.all.x} y={curve.all.y} muted />
              <Bars
                name={`rejection region (α = ${alpha})`}
                x={curve.region.x}
                y={curve.region.y}
                tone="destructive"
                opacity={0.5}
              />
              <Bars name="p-value outcomes" x={curve.pValue.x} y={curve.pValue.y} slot={0} opacity={0.75} />
            </>
          )}
          {Number.isFinite(t) && <Annotation x={t} text={`${r.symbol} = ${f3(t)}`} />}
        </Plot>
        {ci && ciRange && (
          <Plot x={ciAxis} y={ciY}>
            <Segments
              name={`${Math.round(ci.level * 100)}% CI`}
              segments={[{ from: [Math.max(ci.lower, ciRange[0]), 0], to: [Math.min(ci.upper, ciRange[1]), 0] }]}
              slot={0}
              width={3}
            />
            {r.estimate !== undefined && Number.isFinite(r.estimate) && (
              <Points name="estimate" x={[r.estimate]} y={[0]} emphasis />
            )}
            {r.nullValue !== undefined && <Annotation x={r.nullValue} text="null" dashed />}
          </Plot>
        )}
      </Plots>
    </>
  )
}

registerView<TestResult, { alpha: ReturnType<typeof slider> }>({
  key: 'test-result/null',
  kind: 'test-result',
  description:
    'The null law of the statistic with the rejection region at α, the p-value area and the observed statistic; the confidence interval against the null value; the effect size.',
  title: (r) => r.method,
  options: { alpha: slider(0.001, 0.2, 0.05, { step: 0.001, label: 'α' }) },
  render: (r, o) => <HypothesisTestPanel result={r} alpha={o.alpha} />,
})
