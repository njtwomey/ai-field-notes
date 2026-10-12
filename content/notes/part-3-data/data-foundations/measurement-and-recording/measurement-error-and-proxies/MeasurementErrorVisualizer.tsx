import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function MeasurementErrorVisualizer() {
  const state = useFigureState({
    viewMode: choice<'regression_lines' | 'reliability_decay'>(
      [
        { value: 'regression_lines', label: 'Regression Dilution (True beta=1.0 vs Attenuated beta_hat)' },
        { value: 'reliability_decay', label: 'Reliability Ratio lambda = Var(X*) / (Var(X*) + Var(U))' },
      ],
      'regression_lines',
      { label: 'attenuation perspective' },
    ),
    noiseSigma: slider(0.0, 2.0, 0.8, {
      step: 0.1,
      label: 'measurement error standard deviation sigma_u (latent sigma_x = 1.0)',
    }),
  })

  const results = useMemo(() => {
    const varX = 1.0
    const varU = Math.pow(state.noiseSigma, 2)
    const reliability = varX / (varX + varU)
    const betaTrue = 1.0
    const betaAttenuated = betaTrue * reliability
    const biasLossPct = (1.0 - reliability) * 100.0

    // For regression lines: plot y across x in [-2.5, 2.5]
    const xGrid = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    const trueY = xGrid.map((x) => betaTrue * x)
    const observedY = xGrid.map((x) => betaAttenuated * x)

    // For reliability decay: sweep noise sigma [0..2.5]
    const sigmaGrid: number[] = []
    const lambdaCurve: number[] = []
    for (let s = 0.0; s <= 2.5; s += 0.1) {
      sigmaGrid.push(s)
      lambdaCurve.push(1.0 / (1.0 + s * s))
    }

    return {
      xGrid,
      trueY,
      observedY,
      sigmaGrid,
      lambdaCurve,
      betaTrue,
      betaAttenuated,
      reliability,
      biasLossPct,
    }
  }, [state.noiseSigma])

  const axisX = useAxis({
    label: state.viewMode === 'regression_lines' ? 'Observed Predictor W = X* + U' : 'Measurement Noise sigma_u',
    range: state.viewMode === 'regression_lines' ? [-2.6, 2.6] : [0, 2.6],
  })

  const axisY = useAxis({
    label: state.viewMode === 'regression_lines' ? 'Target Outcome Y' : 'Reliability Ratio lambda',
    range: state.viewMode === 'regression_lines' ? [-2.6, 2.6] : [0, 1.05],
  })

  return (
    <Figure
      title="Classical Measurement Error and Attenuation Bias (Regression Dilution)"
      state={state}
      hoverReadout={false}
      caption="Random additive measurement error in features (W = X* + U) attenuates regression coefficients toward zero. Even if a feature has a strong true causal effect (beta=1.0), high sensor noise dilutes the learned slope, leading models and practitioners to falsely dismiss vital predictors."
      readouts={
        <>
          <Readout label="True Latent Effect (beta)" value="1.00" />
          <Readout
            label="Observed Learned Slope"
            value={`${results.betaAttenuated.toFixed(2)} (${results.biasLossPct.toFixed(0)}% dilution)`}
          />
          <Readout label="Reliability Ratio (lambda)" value={`${results.reliability.toFixed(3)}`} />
          <Readout
            label="Attenuated Attenuation"
            value={results.biasLossPct > 40 ? 'Severe false negative risk' : 'Mild shrinkage'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Predictive Effect Attenuation vs True Structural Law">
        {state.viewMode === 'regression_lines' ? (
          <>
            <Curve
              name="True Latent Relationship (beta=1.0)"
              x={results.xGrid}
              y={results.trueY}
              slot={0}
              width={2.5}
            />
            <Curve
              name="Attenuated Observed Regression Line"
              x={results.xGrid}
              y={results.observedY}
              slot={2}
              dashed
              width={2.5}
            />
          </>
        ) : (
          <>
            <Curve name="Reliability Ratio Decay" x={results.sigmaGrid} y={results.lambdaCurve} slot={0} width={2.5} />
            <Points name="Current Operating Point" x={[state.noiseSigma]} y={[results.reliability]} slot={1} size={8} />
          </>
        )}
      </Plot>
    </Figure>
  )
}
