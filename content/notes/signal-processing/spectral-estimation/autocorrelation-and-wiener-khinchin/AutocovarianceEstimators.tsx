import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { arProcess, autocovariance } from '../_shared/spectra'

/**
 * Biased and unbiased autocovariance estimates of an AR(1) process against the true r[m] = a^{|m|}/(1 − a²). The
 * unbiased estimator is right on average but its variance explodes at lags near N.
 */
export function AutocovarianceEstimators() {
  const a = useParam(0.8, { min: -0.95, max: 0.95, step: 0.05 })
  const n = useParam(200, { min: 50, max: 1000, step: 50 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const maxLag = n.value - 1

  const r = useMemo(() => {
    const x = arProcess([a.value], n.value, seed.value)
    const biased = autocovariance(x, maxLag)
    const unbiased = biased.map((v, m) => (v * n.value) / (n.value - m))
    const truth = biased.map((_, m) => a.value ** m / (1 - a.value * a.value))
    return { lags: biased.map((_, m) => m), biased, unbiased, truth }
  }, [a.value, n.value, seed.value, maxLag])

  const series: XYSeries[] = [
    { name: 'unbiased (divide by N − m)', type: 'line', x: r.lags, y: r.unbiased, slot: 2 },
    { name: 'biased (divide by N)', type: 'line', x: r.lags, y: r.biased, slot: 0 },
    { name: 'true r[m]', type: 'line', x: r.lags, y: r.truth, slot: 1, dashed: true },
  ]
  const r0 = 1 / (1 - a.value * a.value)

  return (
    <Interactive
      title="Biased against unbiased autocovariance"
      caption="Sample autocovariances of N samples of an AR(1) process x[n] = a x[n−1] + e[n], with the true r[m] = aᵐ / (1 − a²) dashed. The unbiased estimator divides each lag's sum by the N − m terms in it; at large lags it averages only a handful of products and swings wildly. The biased estimator divides by N, shrinking large lags towards zero; it is positive semidefinite and its transform is the periodogram."
      controls={
        <>
          <ParamSlider label="AR coefficient a" param={a} />
          <ParamSlider label="samples N" param={n} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="true r[0] = 1/(1 − a²)" value={formatNumber(r0)} />
          <Readout label="true S(0) = 1/(1 − a)²" value={formatNumber(1 / (1 - a.value) ** 2)} />
          <Readout label="true S(π) = 1/(1 + a)²" value={formatNumber(1 / (1 + a.value) ** 2)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="lag m"
        yLabel="autocovariance"
        xRange={[0, maxLag]}
        yRange={[-2 * r0, 2 * r0]}
        height={300}
      />
    </Interactive>
  )
}
