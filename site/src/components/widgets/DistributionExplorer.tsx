import { useMemo } from 'react'
import {
  Curve,
  Density,
  Figure,
  Mass,
  Plot,
  Readout,
  choice,
  formatNumber,
  slider,
  useAxis,
  useFigureState,
  type SliderDef,
} from 'aifn-render'
import { distributionSpec, type Params } from './distribution-specs'

/**
 * The standard figure for a `distribution` note: sliders for every parameter, the pmf or pdf (or the cdf), a dashed
 * line at the mean, and the moments. Discrete distributions draw as bars at the integers; continuous ones as curves.
 */
export function DistributionExplorer({ id, caption }: { id: string; caption?: string }) {
  const spec = distributionSpec(id)
  const probe = useMemo(() => spec.make(Object.fromEntries(spec.params.map((p) => [p.key, p.value]))), [spec])
  const discrete = probe.discrete
  const densityLabel = discrete ? 'pmf' : 'pdf'

  const schema = useMemo(
    () => ({
      ...(Object.fromEntries(
        spec.params.map((p) => [p.key, slider(p.min, p.max, p.value, { step: p.step, label: p.label })]),
      ) as Record<string, SliderDef>),
      view: choice(
        [
          { value: 'density', label: densityLabel },
          { value: 'cdf', label: 'cdf' },
        ],
        'density',
        { label: 'show' },
      ),
    }),
    [spec, densityLabel],
  )
  const state = useFigureState(schema)
  const cdfView = state.view === 'cdf'
  // The parameter fields are built from the spec, so their keys are not known to the type.
  const values = state.values as unknown as Params
  // A value key, so the memo below reruns only when a parameter changes (the values object is new every render).
  const key = spec.params.map((p) => values[p.key]).join(',')

  const { dist, lo, hi, cdf, top } = useMemo(() => {
    const nums = key.split(',').map(Number)
    const params: Params = Object.fromEntries(spec.params.map((p, i) => [p.key, nums[i]]))
    const dist = spec.make(params)
    const [lo, hi] = spec.range(params, dist)
    let cdf: { x: number[]; y: number[] }
    let top = 1
    if (discrete) {
      const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
      // A step function: flat between integers, jumping at each k.
      cdf = { x: ks.flatMap((k) => [k, k + 1]), y: ks.flatMap((k) => [dist.cdf(k), dist.cdf(k)]) }
      if (!cdfView) top = Math.max(...ks.map((k) => dist.prob(k)))
    } else {
      const xs = Array.from({ length: 300 }, (_, i) => lo + ((hi - lo) * i) / 299)
      cdf = { x: xs, y: xs.map((x) => dist.cdf(x)) }
      if (!cdfView) top = Math.max(...xs.map((x) => dist.prob(x)).filter(Number.isFinite))
    }
    return { dist, lo, hi, cdf, top }
  }, [spec, key, discrete, cdfView])

  const x = useAxis({ label: discrete ? 'k' : 'x', range: discrete ? [lo - 0.5, hi + 0.5] : [lo, hi] })
  const y = useAxis({
    label: cdfView ? (discrete ? 'P(X ≤ k)' : 'P(X ≤ x)') : discrete ? 'P(X = k)' : 'density',
    range: cdfView ? [0, 1] : [0, undefined],
  })

  const mean = dist.mean()
  const variance = dist.variance()
  const shown = (v: number) => (Number.isFinite(v) ? formatNumber(v) : v === Infinity ? '∞' : 'undefined')

  return (
    <Figure
      title={`${spec.name} distribution`}
      state={state}
      caption={
        caption ?? `Move the parameters to see how the ${densityLabel} and cdf change. The dashed line is the mean.`
      }
      readouts={
        <>
          <Readout label="mean" value={shown(mean)} />
          <Readout label="variance" value={shown(variance)} />
          <Readout label="standard deviation" value={shown(Math.sqrt(variance))} />
        </>
      }
    >
      <Plot x={x} y={y} height={300}>
        {cdfView ? (
          <Curve name="cdf" x={cdf.x} y={cdf.y} slot={0} />
        ) : discrete ? (
          <Mass name={densityLabel} dist={dist} range={[lo, hi]} slot={0} />
        ) : (
          <Density name={densityLabel} dist={dist} range={[lo, hi]} n={300} slot={0} />
        )}
        {Number.isFinite(mean) && <Curve name="mean" x={[mean, mean]} y={[0, top]} dashed slot={1} />}
      </Plot>
    </Figure>
  )
}
