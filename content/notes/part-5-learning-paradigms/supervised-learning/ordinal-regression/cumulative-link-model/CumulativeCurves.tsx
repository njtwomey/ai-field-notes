import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  MathText,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { sigmoid } from 'aifn/numerics/special'

const ETA = toFlat(linspace(-6, 6, 241))
const THETA = [-2, 0, 1.5]

/**
 * Cumulative probabilities P(y ≤ k | η) = σ(θ_k − η) of a four-class cumulative logit model, as functions of the linear
 * predictor. The curves are horizontal shifts of one sigmoid, and the class probabilities are the vertical gaps between
 * neighbouring curves.
 */
export function CumulativeCurves() {
  const state = useFigureState({
    eta: float(0.5, { min: -6, max: 6, step: 0.05, label: 'linear predictor η' }),
  })

  const series = useMemo<SeriesSpec[]>(
    () =>
      THETA.map((t, k) => ({
        name: `P(y ≤ ${k + 1})`,
        type: 'line',
        x: ETA,
        y: ETA.map((e) => sigmoid(t - e)),
        slot: k,
      })),
    [],
  )

  const cum = [0, ...THETA.map((t) => sigmoid(t - state.eta)), 1]
  const probs = cum.slice(1).map((c, k) => c - cum[k])
  // The gaps at the cursor, drawn as thin segments: each one is a class probability.
  const segments: Segment[] = cum.slice(1).map((c, k) => ({ from: [state.eta, cum[k]], to: [state.eta, c] }))

  const xAxis = useAxis({ label: 'linear predictor η', range: [-6, 6] })
  const yAxis = useAxis({ label: 'cumulative probability', range: [0, 1] })
  return (
    <Figure
      title="Cumulative curves and the gaps between them"
      state={state}
      caption={
        <MathText text="Each curve is $P(y \le k \mid \eta) = \sigma(\theta_k - \eta)$ for thresholds $\theta = (-2, 0, 1.5)$. The three curves are the same sigmoid shifted sideways, so they never cross. At the cursor, the gap between neighbouring curves is a class probability: the lowest gap is $P(y = 1)$ and the gap above the top curve is $P(y = 4)$. Drag the cursor along $\eta$." />
      }

      readouts={probs.map((p, k) => (
        <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
      ))}
    >
      <Plot x={xAxis} y={yAxis} height={300} ariaLabel={'Cumulative probability curves against the linear predictor'}>
        {seriesLayers(series)}
        <Segments segments={segments} />
        <Handle {...state.handle('eta', { label: 'η' })} />
      </Plot>
    </Figure>
  )
}
