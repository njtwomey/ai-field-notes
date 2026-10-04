import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/** Four zero-mean densities scaled to the same standard deviation σ, with their differential entropies in nats. */
function family(sigma: number) {
  const b = sigma / Math.SQRT2 // Laplace scale: variance 2b²
  const s = (sigma * Math.sqrt(3)) / Math.PI // logistic scale: variance π²s²/3
  const w = sigma * Math.sqrt(3) // uniform half-width: variance w²/3
  return [
    {
      name: 'Gaussian',
      pdf: (x: number) => Math.exp((-0.5 * x * x) / sigma ** 2) / (sigma * Math.sqrt(2 * Math.PI)),
      entropy: 0.5 * Math.log(2 * Math.PI * Math.E * sigma ** 2),
    },
    {
      name: 'logistic',
      pdf: (x: number) => {
        const e = Math.exp(-Math.abs(x) / s)
        return e / (s * (1 + e) ** 2)
      },
      entropy: Math.log(s) + 2,
    },
    { name: 'Laplace', pdf: (x: number) => Math.exp(-Math.abs(x) / b) / (2 * b), entropy: 1 + Math.log(2 * b) },
    { name: 'uniform', pdf: (x: number) => (Math.abs(x) <= w ? 1 / (2 * w) : 0), entropy: Math.log(2 * w) },
  ]
}

/**
 * Densities with equal variance and their entropies. The Gaussian always has the largest entropy, by the same margin
 * for every σ, because changing σ adds log σ to every entropy.
 */
export function FixedVariance() {
  const state = useFigureState({
    sigma: float(1, { min: 0.5, max: 2, step: 0.05, label: 'standard deviation σ' }),
  })
  const members = useMemo(() => family(state.sigma), [state.sigma])
  const xs = useMemo(() => toFlat(linspace(-4 * state.sigma, 4 * state.sigma, 401)), [state.sigma])
  const series: SeriesSpec[] = useMemo(
    () => members.map((m, i) => ({ name: m.name, type: 'line', x: xs, y: xs.map(m.pdf), slot: i })),
    [members, xs],
  )
  const xAxis = useAxis({ label: 'x', hold: 'union' })
  const yAxis = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="Same variance, different entropy"
      state={state}
      caption="Four densities with mean 0 and the same standard deviation σ. Among all densities on the real line with a given variance, the Gaussian has the largest entropy; the others are more concentrated somewhere, which lowers their entropy. Changing σ shifts every entropy by the same log σ, so the ranking and the gaps never change."

      readouts={
        <>
          {members.map((m) => (
            <Readout key={m.name} label={`${m.name} entropy`} value={formatNumber(m.entropy)} />
          ))}
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
