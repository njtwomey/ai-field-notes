import { useMemo } from 'react'
import { MathText } from 'aifn-render'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { linspace, sigmoid } from '@/lib/math'

const ETA = linspace(-6, 6, 241)
const THETA = [-2, 0, 1.5]

/**
 * Cumulative probabilities P(y ≤ k | η) = σ(θ_k − η) of a four-class cumulative logit model, as functions of the linear
 * predictor. The curves are horizontal shifts of one sigmoid, and the class probabilities are the vertical gaps between
 * neighbouring curves.
 */
export function CumulativeCurves() {
  const eta = useParam(0.5, { min: -6, max: 6, step: 0.05 })

  const series = useMemo<XYSeries[]>(
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

  const cum = [0, ...THETA.map((t) => sigmoid(t - eta.value)), 1]
  const probs = cum.slice(1).map((c, k) => c - cum[k])
  // The gaps at the cursor, drawn as thin segments: each one is a class probability.
  const segments: Segment[] = cum.slice(1).map((c, k) => ({ from: [eta.value, cum[k]], to: [eta.value, c] }))
  const handles: Handle[] = [{ kind: 'x', at: eta.value, label: 'η', onDrag: eta.set }]

  return (
    <Interactive
      title="Cumulative curves and the gaps between them"
      caption={
        <MathText text="Each curve is $P(y \le k \mid \eta) = \sigma(\theta_k - \eta)$ for thresholds $\theta = (-2, 0, 1.5)$. The three curves are the same sigmoid shifted sideways, so they never cross. At the cursor, the gap between neighbouring curves is a class probability: the lowest gap is $P(y = 1)$ and the gap above the top curve is $P(y = 4)$. Drag the cursor along $\eta$." />
      }
      controls={<ParamSlider label="linear predictor η" param={eta} />}
      readout={probs.map((p, k) => (
        <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
      ))}
    >
      <XYChart
        series={series}
        segments={segments}
        handles={handles}
        xRange={[-6, 6]}
        yRange={[0, 1]}
        xLabel="linear predictor η"
        yLabel="cumulative probability"
        height={300}
        ariaLabel="Cumulative probability curves against the linear predictor"
      />
    </Interactive>
  )
}
