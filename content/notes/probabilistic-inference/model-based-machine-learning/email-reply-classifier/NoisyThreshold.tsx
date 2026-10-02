import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { Phi, grid } from '../_shared/gaussian'

const S = grid(-10, 10, 201)

/** Why the classifier adds noise before the threshold: the step becomes a probit curve whose width is the noise. */
export function NoisyThreshold() {
  const threshold = useParam(2, { min: -6, max: 6, step: 0.1 })
  const noiseVar = useParam(10, { min: 0.1, max: 30, step: 0.1 })
  const weightVar = useParam(0, { min: 0, max: 10, step: 0.1 })

  const series = useMemo((): XYSeries[] => {
    const sd = Math.sqrt(noiseVar.value)
    const sdAll = Math.sqrt(noiseVar.value + weightVar.value)
    return [
      { name: 'no noise (step)', type: 'line', x: S, y: S.map((s) => (s > threshold.value ? 1 : 0)), muted: true },
      { name: 'noisy score', type: 'line', x: S, y: S.map((s) => Phi((s - threshold.value) / sd)), slot: 0 },
      {
        name: 'noisy score, uncertain weights',
        type: 'line',
        x: S,
        y: S.map((s) => Phi((s - threshold.value) / sdAll)),
        slot: 1,
        dashed: true,
      },
    ]
  }, [threshold.value, noiseVar.value, weightVar.value])

  return (
    <Interactive
      title="From a hard threshold to a reply probability"
      caption="The reply probability as a function of an email's score. Without noise, the model says reply exactly when the score passes the threshold, and one mislabelled email has probability zero. Gaussian noise of variance σ² turns the step into Φ((score − θ)/σ). Uncertainty about the weights adds its variance to σ² and flattens the curve further, so an unfamiliar email gets a less extreme prediction. Drag the threshold line."
      controls={
        <>
          <ParamSlider label="threshold θ" param={threshold} format={(v) => v.toFixed(1)} />
          <ParamSlider label="noise variance σ²" param={noiseVar} format={(v) => v.toFixed(1)} />
          <ParamSlider label="score variance from the weights" param={weightVar} format={(v) => v.toFixed(1)} />
        </>
      }
      readout={
        <>
          <Readout label="P(reply | score 0)" value={formatNumber(Phi(-threshold.value / Math.sqrt(noiseVar.value)))} />
          <Readout
            label="P(reply | score 5)"
            value={formatNumber(Phi((5 - threshold.value) / Math.sqrt(noiseVar.value)))}
          />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="score"
        yLabel="P(reply)"
        xRange={[-10, 10]}
        yRange={[0, 1]}
        handles={[{ kind: 'x', at: threshold.value, label: 'threshold', onDrag: (x) => threshold.set(x) }]}
      />
    </Interactive>
  )
}
