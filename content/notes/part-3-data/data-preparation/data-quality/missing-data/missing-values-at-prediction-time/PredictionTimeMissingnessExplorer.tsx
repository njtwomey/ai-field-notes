import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function PredictionTimeMissingnessExplorer() {
  const state = useFigureState({
    strategy: choice<'native' | 'frozen_imputer' | 'fallback_cascade'>(
      [
        { value: 'native', label: 'tree native split routing (XGBoost)' },
        { value: 'frozen_imputer', label: 'frozen training imputer + indicator' },
        { value: 'fallback_cascade', label: 'dedicated fallback model cascade' },
      ],
      'native',
      { label: 'serving strategy' },
    ),
    productionMissingRate: slider(0.0, 0.5, 0.15, { step: 0.05, label: 'production missingness rate' }),
    inputX: slider(-2.0, 2.0, 1.2, { step: 0.2, label: 'query feature X₁ value' }),
  })

  const results = useMemo(() => {
    // True underlying target: Y = 1.5 * X1 + 2.0 * X2
    // Suppose X2 is the vulnerable feature that goes missing in production
    // Training baseline: X1 ~ N(0, 1), X2 ~ N(0, 1), trained with 5% missingness
    const x1 = state.inputX
    const trueX2 = 0.8
    const trueY = 1.5 * x1 + 2.0 * trueX2 // ~ 1.5 * x1 + 1.6

    // Under production missingness, X2 is unavailable
    // Evaluate predictions under the three strategies:
    let predY = 0
    let errorStd = 0
    let latencyMs = 0
    let riskLevel = 'Low'

    if (state.strategy === 'native') {
      // Tree native routing: if trained with minimal missingness, default branch is assigned
      // If default direction was "left" (mean X2 ~ 0), it predicts based on default leaf
      // When X2 goes missing, prediction reverts to default split
      predY = 1.5 * x1 + 2.0 * 0.1 // default branch roughly assumes near-zero X2
      // If production missing rate is much higher than training (5%), risk increases
      errorStd = Math.abs(trueY - predY) * (0.8 + state.productionMissingRate)
      latencyMs = 0.4
      riskLevel = state.productionMissingRate > 0.25 ? 'High (silent drift)' : 'Moderate'
    } else if (state.strategy === 'frozen_imputer') {
      // Frozen training median: X2 replaced by frozen train median (0.0) + indicator M=1
      // Model linear adjustment with beta_M:
      predY = 1.5 * x1 + 0.0 + 0.3 // beta_M offset
      errorStd = Math.abs(trueY - predY) * 0.7
      latencyMs = 2.1
      riskLevel = 'Low (stateless & stable)'
    } else {
      // Dedicated fallback model: trained explicitly without X2!
      // Optimally marginalized: E[Y | X1] = 1.5 * X1 + 2.0 * E[X2 | X1] = 1.5 * X1
      predY = 1.5 * x1 + 0.0
      errorStd = 0.6 // calibrated residual variance of marginalized model
      latencyMs = 0.6
      riskLevel = 'Minimal (calibrated variance)'
    }

    // Prediction curve across X1 in [-2, 2]
    const xGrid: number[] = []
    const trueCurve: number[] = []
    const predCurve: number[] = []
    const upperCiCurve: number[] = []
    const lowerCiCurve: number[] = []

    for (let x = -2.0; x <= 2.05; x += 0.2) {
      xGrid.push(x)
      const ty = 1.5 * x + 2.0 * trueX2
      let py = 0
      if (state.strategy === 'native') py = 1.5 * x + 0.2
      else if (state.strategy === 'frozen_imputer') py = 1.5 * x + 0.3
      else py = 1.5 * x

      trueCurve.push(ty)
      predCurve.push(py)
      upperCiCurve.push(py + 1.96 * errorStd)
      lowerCiCurve.push(py - 1.96 * errorStd)
    }

    const queryPointX = [x1]
    const queryPointY = [predY]

    return {
      trueY,
      predY,
      errorAbs: Math.abs(trueY - predY),
      errorStd,
      latencyMs,
      riskLevel,
      xGrid,
      trueCurve,
      predCurve,
      upperCiCurve,
      lowerCiCurve,
      queryPointX,
      queryPointY,
    }
  }, [state.strategy, state.productionMissingRate, state.inputX])

  const xAxis = useAxis({ label: 'observed query feature X₁', range: [-2.0, 2.0] })
  const yAxis = useAxis({ label: 'model prediction ŷ vs ground truth y', range: [-3.0, 5.0] })

  return (
    <Figure
      title="Prediction-time missingness strategies and fallback calibration"
      state={state}
      hoverReadout={false}
      caption="When vulnerable feature X₂ is missing at prediction time, different production strategies respond differently. Native tree routing routes queries down pre-compiled default branches with zero latency overhead, but risks silent divergence if production missingness exceeds training rates. Frozen training imputers ensure stateless repeatability. Dedicated fallback cascades invoke a model trained without X₂, guaranteeing calibrated marginal uncertainty."
      readouts={
        <>
          <Readout label="serving prediction ŷ" value={results.predY.toFixed(2)} />
          <Readout label="ground truth y" value={results.trueY.toFixed(2)} />
          <Readout label="absolute error |y - ŷ|" value={results.errorAbs.toFixed(2)} />
          <Readout label="serving latency" value={`${results.latencyMs.toFixed(1)} ms`} />
          <Readout label="operational risk" value={results.riskLevel} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Serving prediction response across feature values">
        <Curve
          name="ground truth response (when X₂ is observed)"
          x={results.xGrid}
          y={results.trueCurve}
          slot={2}
          dashed
          width={1.5}
        />
        <Curve name="serving model prediction ŷ" x={results.xGrid} y={results.predCurve} slot={0} width={2.5} />
        <Curve name="uncertainty upper bound" x={results.xGrid} y={results.upperCiCurve} slot={1} thin width={1} />
        <Curve name="uncertainty lower bound" x={results.xGrid} y={results.lowerCiCurve} slot={1} thin width={1} />
        <Points name="current query prediction" x={results.queryPointX} y={results.queryPointY} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
