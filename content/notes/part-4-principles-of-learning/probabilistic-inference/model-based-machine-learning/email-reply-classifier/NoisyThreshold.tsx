import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { Phi, grid } from '../_shared/gaussian'

const S = grid(-10, 10, 201)

/** Why the classifier adds noise before the threshold: the step becomes a probit curve whose width is the noise. */
export function NoisyThreshold() {
  const state = useFigureState({
    threshold: slider(-6, 6, 2, { step: 0.1, label: 'threshold θ', format: (v) => v.toFixed(1) }),
    noiseVar: float(10, { min: 0.1, max: 30, step: 0.1, label: 'noise variance σ²', format: (v) => v.toFixed(1) }),
    weightVar: float(0, {
      min: 0,
      max: 10,
      step: 0.1,
      label: 'score variance from the weights',
      format: (v) => v.toFixed(1),
    }),
  })

  const series = useMemo(() => {
    const sd = Math.sqrt(state.noiseVar)
    const sdAll = Math.sqrt(state.noiseVar + state.weightVar)
    return [
      { name: 'no noise (step)', x: S, y: S.map((s) => (s > state.threshold ? 1 : 0)), muted: true },
      { name: 'noisy score', x: S, y: S.map((s) => Phi((s - state.threshold) / sd)), slot: 0 },
      {
        name: 'noisy score, uncertain weights',
        x: S,
        y: S.map((s) => Phi((s - state.threshold) / sdAll)),
        slot: 1,
        dashed: true,
      },
    ] as const
  }, [state.threshold, state.noiseVar, state.weightVar])

  const xAxis = useAxis({ label: 'score', range: [-10, 10] })
  const yAxis = useAxis({ label: 'P(reply)', range: [0, 1] })
  return (
    <Figure
      title="From a hard threshold to a reply probability"
      state={state}
      caption="The reply probability as a function of an email's score. Without noise, the model says reply exactly when the score passes the threshold, and one mislabelled email has probability zero. Gaussian noise of variance σ² turns the step into Φ((score − θ)/σ). Uncertainty about the weights adds its variance to σ² and flattens the curve further, so an unfamiliar email gets a less extreme prediction. Drag the threshold line."

      readouts={
        <>
          <Readout label="P(reply | score 0)" value={formatNumber(Phi(-state.threshold / Math.sqrt(state.noiseVar)))} />
          <Readout
            label="P(reply | score 5)"
            value={formatNumber(Phi((5 - state.threshold) / Math.sqrt(state.noiseVar)))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('threshold', { label: 'threshold' })} />
      </Plot>
    </Figure>
  )
}
