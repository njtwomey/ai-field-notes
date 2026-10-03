import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { distribution } from '@/lib/distributions'

const poisson = distribution('poisson')
const binomial = distribution('binomial')

/** Binomial(n, λ/n) against Poisson(λ) as n grows, with the total-variation distance and Le Cam's bound. */
export function BinomialLimit() {
  const lambdaParam = useParam(3, { min: 0.5, max: 10, step: 0.5 })
  const lambda = lambdaParam.value
  const [logN, setLogN] = useState(1)
  const n = Math.max(Math.ceil(lambda), Math.round(10 ** logN))

  const result = useMemo(() => {
    const kMax = Math.max(12, Math.ceil(lambda + 5 * Math.sqrt(lambda)))
    const ks = Array.from({ length: kMax + 1 }, (_, k) => k)
    const pois = ks.map((k) => poisson.density(k, { lambda }))
    const binom = ks.map((k) => binomial.density(k, { n, p: lambda / n }))
    // Mass beyond kMax is negligible for the ranges plotted.
    const tv = 0.5 * ks.reduce((s, k) => s + Math.abs(pois[k] - binom[k]), 0)
    const series: XYSeries[] = [
      { name: 'Poisson(λ)', type: 'scatter', x: ks, y: pois, slot: 0 },
      { name: `Binomial(n, λ/n)`, type: 'line', x: ks, y: binom, slot: 1, dashed: true },
    ]
    return { series, tv }
  }, [lambda, n])

  return (
    <Interactive
      title="The law of rare events"
      caption="Split an interval into n pieces, each holding an event with probability λ/n. As n grows, the binomial count (dashed) approaches the Poisson pmf (points). The distance between them is bounded by λ²/n."
      controls={
        <>
          <ParamSlider label="rate λ" param={lambdaParam} />
          <ParamSlider
            label="pieces n"
            value={logN}
            onChange={setLogN}
            min={0.5}
            max={3}
            step={0.05}
            format={(v) => String(Math.max(Math.ceil(lambda), Math.round(10 ** v)))}
          />
        </>
      }
      readout={
        <>
          <Readout label="n" value={n} />
          <Readout label="p = λ/n" value={formatNumber(lambda / n)} />
          <Readout label="total-variation distance" value={formatNumber(result.tv)} />
          <Readout label="Le Cam bound λ²/n" value={formatNumber((lambda * lambda) / n)} />
        </>
      }
    >
      <XYChart height={280} series={result.series} xLabel="k" yLabel="P(X = k)" yRange={[0, undefined]} />
    </Interactive>
  )
}
