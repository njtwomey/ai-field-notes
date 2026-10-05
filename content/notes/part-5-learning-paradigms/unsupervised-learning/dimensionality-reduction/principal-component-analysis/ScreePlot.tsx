import { useMemo } from 'react'
import { Bars, Curve, Figure, float, formatNumber, Plot, Readout, setting, useAxis, useFigureState } from 'aifn-render'
import { covariance, eigSymmetric } from '../../_shared/linalg'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 200
const D = 10
const FACTORS = 3

/** Fixed loadings: feature m depends on the three latent factors with weights drawn once. */
const LOADINGS = (() => {
  const r = stream(12)
  return Array.from({ length: D }, () => Array.from({ length: FACTORS }, () => normal(r)))
})()

/** n points x = W z + noise·ε in 10 dimensions, with three latent factors z. */
function sample(noise: number, bigUnits: boolean): number[][] {
  const r = stream(3)
  return Array.from({ length: N }, () => {
    const z = Array.from({ length: FACTORS }, () => normal(r))
    return LOADINGS.map((w, m) => {
      const x = w.reduce((s, wk, k) => s + wk * z[k], 0) + noise * normal(r)
      // Feature 10 recorded in units 20 times smaller, so its values are 20 times larger.
      return bigUnits && m === D - 1 ? 20 * x : x
    })
  })
}

function standardise(x: number[][]): number[][] {
  const mean = Array.from({ length: D }, (_, m) => x.reduce((s, r) => s + r[m], 0) / x.length)
  const sd = Array.from({ length: D }, (_, m) => Math.sqrt(x.reduce((s, r) => s + (r[m] - mean[m]) ** 2, 0) / x.length))
  return x.map((r) => r.map((v, m) => (v - mean[m]) / sd[m]))
}

export function ScreePlot() {
  const state = useFigureState({
    noise: float(0.6, { min: 0.05, max: 3, step: 0.05, label: 'noise standard deviation' }),
    bigUnits: setting(false, 'feature 10 in units 20× smaller'),
    scale: setting(false, 'standardise features'),
  })
  const values = useMemo(() => {
    const x = sample(state.noise, state.bigUnits)
    return eigSymmetric(covariance(state.scale ? standardise(x) : x)).values.map((v) => Math.max(v, 0))
  }, [state.noise, state.bigUnits, state.scale])
  const total = values.reduce((a, b) => a + b, 0)
  const ratio = values.map((v) => v / total)
  const cumulative = ratio.map((_, j) => ratio.slice(0, j + 1).reduce((a, b) => a + b, 0))
  const needed = cumulative.findIndex((c) => c >= 0.9) + 1
  const ks = values.map((_, j) => j + 1)

  const xAxis = useAxis({ label: 'component', hold: 'union' })
  const yAxis = useAxis({ label: 'share of variance', range: [0, 1] })
  return (
    <Figure
      title="Scree plot"
      state={state}
      caption="Ten features generated from three latent factors plus independent noise. The bars are each component's share of the total variance; the line is the cumulative share. With little noise, three components hold almost everything and the bars drop sharply after the third. Record feature 10 in units 20 times smaller and the first component becomes that feature alone, until the features are standardised."

      readouts={
        <>
          <Readout label="first three components explain" value={`${(100 * cumulative[2]).toFixed(1)}%`} />
          <Readout label="components for 90%" value={needed} />
          <Readout label="largest eigenvalue" value={formatNumber(values[0])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Bars name="explained variance ratio" x={ks} y={ratio} slot={0} />
        <Curve name="cumulative" x={ks} y={cumulative} slot={1} />
      </Plot>
    </Figure>
  )
}
