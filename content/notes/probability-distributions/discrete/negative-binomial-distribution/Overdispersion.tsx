import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { distribution } from '@/lib/distributions'

const poisson = distribution('poisson')
const negativeBinomial = distribution('negative-binomial')

/** A negative binomial and a Poisson with the same mean, as the dispersion parameter r varies. */
export function Overdispersion() {
  const [mean, setMean] = useState(5)
  const [logR, setLogR] = useState(0.5)
  const r = Number((10 ** logR).toPrecision(2))
  const p = r / (r + mean)

  const result = useMemo(() => {
    const sd = Math.sqrt(mean + (mean * mean) / r)
    const kMax = Math.min(150, Math.max(15, Math.ceil(mean + 5 * sd)))
    const ks = Array.from({ length: kMax + 1 }, (_, k) => k)
    const series: XYSeries[] = [
      { name: 'negative binomial', type: 'scatter', x: ks, y: ks.map((k) => negativeBinomial.density(k, { r, p })) },
      {
        name: 'Poisson, same mean',
        type: 'line',
        x: ks,
        y: ks.map((k) => poisson.density(k, { lambda: mean })),
        slot: 1,
        dashed: true,
      },
    ]
    return { series }
  }, [mean, r, p])

  return (
    <Interactive
      title="Same mean, more spread"
      caption="Both distributions have mean μ. The negative binomial (points) has variance μ + μ²/r, the Poisson (dashed) has variance μ. Small r gives a long right tail and many more zeros. As r grows the negative binomial approaches the Poisson."
      controls={
        <>
          <ParamSlider label="mean μ" value={mean} onChange={setMean} min={0.5} max={20} step={0.5} />
          <ParamSlider
            label="dispersion r"
            value={logR}
            onChange={setLogR}
            min={-0.3}
            max={2}
            step={0.05}
            format={(v) => String(Number((10 ** v).toPrecision(2)))}
          />
        </>
      }
      readout={
        <>
          <Readout label="p = r/(r + μ)" value={formatNumber(p)} />
          <Readout label="variance, negative binomial" value={formatNumber(mean + (mean * mean) / r)} />
          <Readout label="variance, Poisson" value={formatNumber(mean)} />
          <Readout label="P(X = 0), negative binomial" value={formatNumber(negativeBinomial.density(0, { r, p }))} />
          <Readout label="P(X = 0), Poisson" value={formatNumber(Math.exp(-mean))} />
        </>
      }
    >
      <XYChart height={280} series={result.series} xLabel="k" yLabel="P(X = k)" yRange={[0, undefined]} />
    </Interactive>
  )
}
