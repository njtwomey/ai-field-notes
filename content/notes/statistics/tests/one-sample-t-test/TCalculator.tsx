import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { logGamma, normalPdf, studentTCdf } from '@/lib/math/special'
import { studentTQuantile } from '@/lib/math/tests'

/** Student t density with ν degrees of freedom. */
function tPdf(t: number, df: number): number {
  const logC = logGamma((df + 1) / 2) - logGamma(df / 2) - 0.5 * Math.log(df * Math.PI)
  return Math.exp(logC - ((df + 1) / 2) * Math.log(1 + (t * t) / df))
}

/** One-sample t-test from summary statistics, with the observed t against its null distribution. */
export function TCalculator() {
  const [mean, setMean] = useState(5.2)
  const [mu0, setMu0] = useState(4.5)
  const [sd, setSd] = useState(1.2)
  const [n, setN] = useState(16)

  const result = useMemo(() => {
    const df = n - 1
    const se = sd / Math.sqrt(n)
    const t = (mean - mu0) / se
    const p = 2 * (1 - studentTCdf(Math.abs(t), df))
    const q = studentTQuantile(0.975, df)
    const lim = Math.max(5, Math.abs(t) + 1)
    const xs = linspace(-lim, lim, 400)
    const tail = (keep: (x: number) => boolean) => {
      const inside = xs.filter(keep)
      return { x: inside, y: inside.map((x) => tPdf(x, df)) }
    }
    const series: XYSeries[] = [
      { name: `t, ${df} df`, type: 'line', x: xs, y: xs.map((x) => tPdf(x, df)), slot: 0 },
      { name: 'standard normal', type: 'line', x: xs, y: xs.map(normalPdf), slot: 2, dashed: true },
      ...[tail((x) => x <= -Math.abs(t)), tail((x) => x >= Math.abs(t))].map((r): XYSeries => ({
        name: 'p-value',
        type: 'line',
        ...r,
        slot: 0,
        area: true,
      })),
    ]
    return { series, t, df, p, se, lower: mean - q * se, upper: mean + q * se }
  }, [mean, mu0, sd, n])

  return (
    <Interactive
      title="One-sample t-test from summary statistics"
      caption="The curve is the t distribution with n − 1 degrees of freedom, the distribution of the statistic when H₀ is true. The shaded tails beyond ±t are the two-sided p-value. The dashed normal curve has thinner tails; the difference matters for small n."
      controls={
        <>
          <ParamSlider label="sample mean x̄" value={mean} onChange={setMean} min={3} max={7} step={0.05} />
          <ParamSlider label="null value μ₀" value={mu0} onChange={setMu0} min={3} max={7} step={0.05} />
          <ParamSlider label="sample standard deviation s" value={sd} onChange={setSd} min={0.2} max={3} step={0.05} />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={2} max={100} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="t" value={formatNumber(result.t)} />
          <Readout label="df" value={result.df} />
          <Readout label="p (two-sided)" value={formatNumber(result.p)} />
          <Readout label="95% interval" value={`[${formatNumber(result.lower)}, ${formatNumber(result.upper)}]`} />
        </>
      }
    >
      <XYChart height={280} series={result.series} xLabel="t" yLabel="density" yRange={[0, undefined]} />
    </Interactive>
  )
}
