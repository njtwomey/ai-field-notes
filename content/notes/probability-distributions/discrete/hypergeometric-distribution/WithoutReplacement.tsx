import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { distribution } from '@/lib/distributions'

const hypergeometric = distribution('hypergeometric')
const binomial = distribution('binomial')

/** Drawing without replacement (hypergeometric) against with replacement (binomial) as the population grows. */
export function WithoutReplacement() {
  const [n, setN] = useState(10)
  const [share, setShare] = useState(0.4)
  const [logN, setLogN] = useState(1.3)
  const N = Math.max(n, Math.round(10 ** logN))
  const K = Math.round(share * N)
  const p = K / N

  const result = useMemo(() => {
    const ks = Array.from({ length: n + 1 }, (_, k) => k)
    const hyper = ks.map((k) => hypergeometric.density(k, { N, K, n }))
    const binom = ks.map((k) => binomial.density(k, { n, p }))
    const tv = 0.5 * ks.reduce((s, k) => s + Math.abs(hyper[k] - binom[k]), 0)
    const series: XYSeries[] = [
      { name: 'without replacement (hypergeometric)', type: 'scatter', x: ks, y: hyper },
      { name: 'with replacement (binomial)', type: 'line', x: ks, y: binom, slot: 1, dashed: true },
    ]
    return { series, tv }
  }, [N, K, n, p])

  const fpc = N === 1 ? 0 : (N - n) / (N - 1)
  return (
    <Interactive
      title="Sampling without replacement"
      caption="Draw n items from a population of N, a fraction K/N of which are successes. Without replacement (points) the count is hypergeometric. With replacement (dashed) it is binomial with p = K/N. Both have mean np. Without replacement the variance is smaller by the factor (N − n)/(N − 1). Grow N and the two coincide; shrink N towards n and the hypergeometric narrows until, at N = n, the count is always K."
      controls={
        <>
          <ParamSlider label="draws n" value={n} onChange={setN} min={1} max={40} step={1} />
          <ParamSlider
            label="success fraction K/N"
            value={share}
            onChange={setShare}
            min={0.05}
            max={0.95}
            step={0.05}
          />
          <ParamSlider
            label="population N"
            value={logN}
            onChange={setLogN}
            min={0}
            max={3.3}
            step={0.05}
            format={(v) => String(Math.max(n, Math.round(10 ** v)))}
          />
        </>
      }
      readout={
        <>
          <Readout label="N" value={N} />
          <Readout label="K" value={K} />
          <Readout label="(N − n)/(N − 1)" value={formatNumber(fpc)} />
          <Readout label="variance, hypergeometric" value={formatNumber(hypergeometric.variance({ N, K, n }))} />
          <Readout label="variance, binomial" value={formatNumber(n * p * (1 - p))} />
          <Readout label="total-variation distance" value={formatNumber(result.tv)} />
        </>
      }
    >
      <XYChart
        height={280}
        series={result.series}
        xLabel="successes drawn k"
        yLabel="P(X = k)"
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}
