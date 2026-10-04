import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { arProcess, autocovariance } from '../_shared/spectra'

/**
 * Biased and unbiased autocovariance estimates of an AR(1) process against the true r[m] = a^{|m|}/(1 − a²). The
 * unbiased estimator is right on average but its variance explodes at lags near N.
 */
export function AutocovarianceEstimators() {
  const state = useFigureState({
    a: float(0.8, { min: -0.95, max: 0.95, step: 0.05, label: 'AR coefficient a' }),
    n: int(200, { min: 50, max: 1000, step: 50, label: 'samples N', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const maxLag = state.n - 1

  const r = useMemo(() => {
    const x = arProcess([state.a], state.n, state.seed)
    const biased = autocovariance(x, maxLag)
    const unbiased = biased.map((v, m) => (v * state.n) / (state.n - m))
    const truth = biased.map((_, m) => state.a ** m / (1 - state.a * state.a))
    return { lags: biased.map((_, m) => m), biased, unbiased, truth }
  }, [state.a, state.n, state.seed, maxLag])

  const series = [
    { name: 'unbiased (divide by N − m)', x: r.lags, y: r.unbiased, slot: 2 },
    { name: 'biased (divide by N)', x: r.lags, y: r.biased, slot: 0 },
    { name: 'true r[m]', x: r.lags, y: r.truth, slot: 1, dashed: true },
  ] as const
  const r0 = 1 / (1 - state.a * state.a)

  const xAxis = useAxis({ label: 'lag m', range: [0, maxLag] })
  const yAxis = useAxis({ label: 'autocovariance', range: [-2 * r0, 2 * r0] })
  return (
    <Figure
      title="Biased against unbiased autocovariance"
      state={state}
      caption="Sample autocovariances of N samples of an AR(1) process x[n] = a x[n−1] + e[n], with the true r[m] = aᵐ / (1 − a²) dashed. The unbiased estimator divides each lag's sum by the N − m terms in it; at large lags it averages only a handful of products and swings wildly. The biased estimator divides by N, shrinking large lags towards zero; it is positive semidefinite and its transform is the periodogram."

      readouts={
        <>
          <Readout label="true r[0] = 1/(1 − a²)" value={formatNumber(r0)} />
          <Readout label="true S(0) = 1/(1 − a)²" value={formatNumber(1 / (1 - state.a) ** 2)} />
          <Readout label="true S(π) = 1/(1 + a)²" value={formatNumber(1 / (1 + state.a) ** 2)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
