import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { logGamma, normalPdf, studentTCdf, studentTQuantile } from 'aifn-compute/numerics/special'

/** Student t density with ν degrees of freedom. */
function tPdf(t: number, df: number): number {
  const logC = logGamma((df + 1) / 2) - logGamma(df / 2) - 0.5 * Math.log(df * Math.PI)
  return Math.exp(logC - ((df + 1) / 2) * Math.log(1 + (t * t) / df))
}

/** One-sample t-test from summary statistics, with the observed t against its null distribution. */
export function TCalculator() {
  const state = useFigureState({
    mean: int(5.2, { min: 3, max: 7, step: 0.05, label: 'sample mean x̄' }),
    mu0: float(4.5, { min: 3, max: 7, step: 0.05, label: 'null value μ₀' }),
    sd: int(1.2, { min: 0.2, max: 3, step: 0.05, label: 'sample standard deviation s' }),
    n: int(16, { min: 2, max: 100, step: 1, label: 'sample size n' }),
  })

  const result = useMemo(() => {
    const df = state.n - 1
    const se = state.sd / Math.sqrt(state.n)
    const t = (state.mean - state.mu0) / se
    const p = 2 * (1 - studentTCdf(Math.abs(t), df))
    const q = studentTQuantile(0.975, df)
    const lim = Math.max(5, Math.abs(t) + 1)
    const xs = toFlat(linspace(-lim, lim, 400))
    const tail = (keep: (x: number) => boolean) => {
      const inside = xs.filter(keep)
      return { x: inside, y: inside.map((x) => tPdf(x, df)) }
    }
    const series: SeriesSpec[] = [
      { name: `t, ${df} df`, type: 'line', x: xs, y: xs.map((x) => tPdf(x, df)), slot: 0 },
      { name: 'standard normal', type: 'line', x: xs, y: xs.map((v: number) => normalPdf(v)), slot: 2, dashed: true },
      ...[tail((x) => x <= -Math.abs(t)), tail((x) => x >= Math.abs(t))].map((r): SeriesSpec => ({
        name: 'p-value',
        type: 'line',
        ...r,
        slot: 0,
        area: true,
      })),
    ]
    return { series, t, df, p, se, lower: state.mean - q * se, upper: state.mean + q * se }
  }, [state.mean, state.mu0, state.sd, state.n])

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="One-sample t-test from summary statistics"
      state={state}
      caption="The curve is the t distribution with n − 1 degrees of freedom, the distribution of the statistic when H₀ is true. The shaded tails beyond ±t are the two-sided p-value. The dashed normal curve has thinner tails; the difference matters for small n."

      readouts={
        <>
          <Readout label="t" value={formatNumber(result.t)} />
          <Readout label="df" value={result.df} />
          <Readout label="p (two-sided)" value={formatNumber(result.p)} />
          <Readout label="95% interval" value={`[${formatNumber(result.lower)}, ${formatNumber(result.upper)}]`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        {seriesLayers(result.series)}
      </Plot>
    </Figure>
  )
}
