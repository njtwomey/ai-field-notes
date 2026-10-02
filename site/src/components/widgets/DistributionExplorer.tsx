import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber } from 'aifn-render'
import { defaults, distribution, type Params } from '@/lib/distributions'
import { linspace } from '@/lib/math'

type View = 'density' | 'cdf'

/**
 * The standard figure for a `distribution` note: sliders for every parameter, the pmf or pdf (or the cdf), a dashed
 * line at the mean, and the moments. Discrete distributions draw as stems; continuous ones as curves.
 */
export function DistributionExplorer({ id, caption }: { id: string; caption?: string }) {
  const d = distribution(id)
  const [params, setParams] = useState<Params>(() => defaults(d))
  const [view, setView] = useState<View>('density')
  const densityLabel = d.discrete ? 'pmf' : 'pdf'

  const { series, segments } = useMemo(() => {
    const [lo, hi] = d.range(params)
    if (d.discrete) {
      const ks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
      if (view === 'density') {
        const ps = ks.map((k) => d.density(k, params))
        return {
          series: [{ name: densityLabel, type: 'scatter' as const, x: ks, y: ps }],
          segments: ks.map((k, i) => ({ from: [k, 0] as [number, number], to: [k, ps[i]] as [number, number] })),
        }
      }
      // A step function: flat between integers, jumping at each k.
      const xs = ks.flatMap((k) => [k, k + 1])
      const ys = ks.flatMap((k) => [d.cdf(k, params), d.cdf(k, params)])
      return { series: [{ name: 'cdf', type: 'line' as const, x: xs, y: ys }], segments: [] }
    }

    const xs = linspace(lo, hi, 300)
    const ys = xs.map((x) => (view === 'density' ? d.density(x, params) : d.cdf(x, params)))
    return {
      series: [{ name: view === 'density' ? densityLabel : 'cdf', type: 'line' as const, x: xs, y: ys }],
      segments: [],
    }
  }, [d, params, view, densityLabel])

  const mean = d.mean(params)
  const variance = d.variance(params)
  // A dashed line at the mean, from the axis to the top of the curve (or to 1 for the cdf).
  const top = view === 'cdf' ? 1 : Math.max(...series[0].y)
  const withMean = Number.isFinite(mean)
    ? [...series, { name: 'mean', type: 'line' as const, dashed: true, x: [mean, mean], y: [0, top] }]
    : series
  const shown = (v: number) => (Number.isFinite(v) ? formatNumber(v) : v === Infinity ? '∞' : 'undefined')

  return (
    <Interactive
      title={`${d.name} distribution`}
      caption={
        caption ?? `Move the parameters to see how the ${densityLabel} and cdf change. The dashed line is the mean.`
      }
      controls={
        <>
          {d.params.map((p) => (
            <ParamSlider
              key={p.key}
              label={p.label}
              value={params[p.key]}
              onChange={(v) => setParams((prev) => ({ ...prev, [p.key]: v }))}
              min={p.min}
              max={p.max}
              step={p.step}
            />
          ))}
          <ParamChoice
            label="show"
            value={view}
            onChange={setView}
            options={[
              { value: 'density', label: densityLabel },
              { value: 'cdf', label: 'cdf' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="mean" value={shown(mean)} />
          <Readout label="variance" value={shown(variance)} />
          <Readout label="standard deviation" value={shown(Math.sqrt(variance))} />
        </>
      }
    >
      <XYChart
        height={300}
        series={withMean}
        segments={segments}
        xLabel={d.discrete ? 'k' : 'x'}
        yLabel={view === 'density' ? (d.discrete ? 'P(X = k)' : 'density') : 'P(X ≤ x)'}
        yRange={view === 'cdf' ? [0, 1] : [0, undefined]}
      />
    </Interactive>
  )
}
