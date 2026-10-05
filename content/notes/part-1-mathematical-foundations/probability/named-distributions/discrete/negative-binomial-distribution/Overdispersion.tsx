import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { NegativeBinomial, Poisson } from 'aifn-compute/probability/distributions'

/** A negative binomial and a Poisson with the same mean, as the dispersion parameter r varies. */
export function Overdispersion() {
  const state = useFigureState({
    mean: float(5, { min: 0.5, max: 20, step: 0.5, label: 'mean μ' }),
    logR: float(0.5, {
      min: -0.3,
      max: 2,
      step: 0.05,
      label: 'dispersion r',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => String(Number((10 ** v).toPrecision(2))),
    }),
  })
  const r = Number((10 ** state.logR).toPrecision(2))
  const p = r / (r + state.mean)

  const result = useMemo(() => {
    const sd = Math.sqrt(state.mean + (state.mean * state.mean) / r)
    const kMax = Math.min(150, Math.max(15, Math.ceil(state.mean + 5 * sd)))
    const ks = Array.from({ length: kMax + 1 }, (_, k) => k)
    const negativeBinomial = NegativeBinomial(r, p)
    const poisson = Poisson(state.mean)
    const series = [
      { name: 'negative binomial', x: ks, y: ks.map((k) => negativeBinomial.prob(k)) },
      {
        name: 'Poisson, same mean',
        x: ks,
        y: ks.map((k) => poisson.prob(k)),
        slot: 1,
        dashed: true,
      },
    ] as const
    return { series }
  }, [state.mean, r, p])

  const xAxis = useAxis({ label: 'k', hold: 'union' })
  const yAxis = useAxis({ label: 'P(X = k)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Same mean, more spread"
      state={state}
      caption="Both distributions have mean μ. The negative binomial (points) has variance μ + μ²/r, the Poisson (dashed) has variance μ. Small r gives a long right tail and many more zeros. As r grows the negative binomial approaches the Poisson."

      readouts={
        <>
          <Readout label="p = r/(r + μ)" value={formatNumber(p)} />
          <Readout
            label="variance, negative binomial"
            value={formatNumber(state.mean + (state.mean * state.mean) / r)}
          />
          <Readout label="variance, Poisson" value={formatNumber(state.mean)} />
          <Readout label="P(X = 0), negative binomial" value={formatNumber(p ** r)} />
          <Readout label="P(X = 0), Poisson" value={formatNumber(Math.exp(-state.mean))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Points {...result.series[0]} />
        <Curve {...result.series[1]} />
      </Plot>
    </Figure>
  )
}
