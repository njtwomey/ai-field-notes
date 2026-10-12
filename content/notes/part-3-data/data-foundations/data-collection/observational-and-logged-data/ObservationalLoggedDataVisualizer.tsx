import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ObservationalLoggedDataVisualizer() {
  const state = useFigureState({
    estimatorMode: choice<'naive_observational' | 'ips_debiased'>(
      [
        { value: 'naive_observational', label: 'Naive Observational Difference E[Y | A=1] - E[Y | A=0]' },
        { value: 'ips_debiased', label: 'Causal Inverse Propensity Weighted (IPS / Doubly Robust)' },
      ],
      'naive_observational',
      { label: 'estimation strategy' },
    ),
    confoundingSeverity: slider(0.0, 1.2, 0.7, {
      step: 0.1,
      label: 'logging policy confounding strength gamma (assignment bias)',
    }),
  })

  const results = useMemo(() => {
    // 10 user strata with baseline severity X in [1..10]
    // Baseline outcome: Y0 = 2.0 * X
    // True treatment effect: tau = -4.0 (treatment lowers outcome by 4.0 units)
    // Y1 = Y0 + tau = 2.0 * X - 4.0
    const xGrid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const gamma = state.confoundingSeverity

    const propensities: number[] = []
    let treatedWeightedY = 0
    let controlWeightedY = 0
    let totalTreatedW = 0
    let totalControlW = 0

    let treatedRawY = 0
    let controlRawY = 0
    let totalTreatedCount = 0
    let totalControlCount = 0

    xGrid.forEach((x) => {
      // Propensity to treat increases with baseline severity:
      // When gamma = 0, p = 0.5 (randomized trial)
      const logit = gamma * (x - 5.5)
      const p = 1.0 / (1.0 + Math.exp(-logit))
      propensities.push(p)

      const y0 = 2.0 * x
      const y1 = y0 - 4.0 // true causal effect is -4.0

      // In expectation across large cohort at severity x:
      // p individuals get treatment A=1, (1-p) get control A=0
      treatedRawY += p * y1
      totalTreatedCount += p

      controlRawY += (1 - p) * y0
      totalControlCount += 1 - p

      // IPS weighting: weight treated by 1/p, control by 1/(1-p)
      treatedWeightedY += (p * y1) / Math.max(0.02, p)
      totalTreatedW += p / Math.max(0.02, p)

      controlWeightedY += ((1 - p) * y0) / Math.max(0.02, 1 - p)
      totalControlW += (1 - p) / Math.max(0.02, 1 - p)
    })

    const meanTreatedObs = treatedRawY / Math.max(0.001, totalTreatedCount)
    const meanControlObs = controlRawY / Math.max(0.001, totalControlCount)
    const naiveATE = meanTreatedObs - meanControlObs

    const meanTreatedIPS = treatedWeightedY / Math.max(0.001, totalTreatedW)
    const meanControlIPS = controlWeightedY / Math.max(0.001, totalControlW)
    const ipsATE = meanTreatedIPS - meanControlIPS

    const trueATE = -4.0
    const estimatedATE = state.estimatorMode === 'naive_observational' ? naiveATE : ipsATE
    const signFlip = estimatedATE > 0 && trueATE < 0

    return {
      xGrid,
      propensities,
      trueATE,
      estimatedATE,
      naiveATE,
      ipsATE,
      signFlip,
    }
  }, [state.confoundingSeverity, state.estimatorMode])

  const axisX = useAxis({
    label: 'Baseline Severity / Risk Score X (1 to 10)',
    range: [0.8, 10.2],
  })

  const axisY = useAxis({
    label: 'Treatment Assignment Propensity P(A=1 | X)',
    range: [0, 1.05],
  })

  return (
    <Figure
      title="Observational Confounding and Counterfactual Causal Adjustment"
      state={state}
      hoverReadout={false}
      caption="The true intervention effect is beneficial (tau = -4.0). Under confounded logging policies, sicker users receive treatment disproportionately. Naive correlation shows a sign inversion (appearing harmful), whereas inverse propensity scoring (IPS) recovers the true causal benefit."
      readouts={
        <>
          <Readout label="True Causal ATE" value="-4.00 (Beneficial)" />
          <Readout
            label="Estimated Effect"
            value={`${results.estimatedATE > 0 ? '+' : ''}${results.estimatedATE.toFixed(2)} (${results.signFlip ? 'Spurious Sign Inversion!' : 'Unbiased'})`}
          />
          <Readout
            label="Confounding Bias"
            value={`${Math.abs(results.estimatedATE - results.trueATE).toFixed(2)} units distortion`}
          />
          <Readout
            label="Randomization Status"
            value={state.confoundingSeverity < 0.1 ? 'Unconfounded RCT' : 'Confounded Logging Policy'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Logging Policy Treatment Probability by Severity">
        <Curve
          name="Logging Policy Propensity pi_0(A=1 | X)"
          x={results.xGrid}
          y={results.propensities}
          slot={0}
          width={2.5}
        />
        <Points name="Strata Propensity Points" x={results.xGrid} y={results.propensities} slot={1} size={7} />
      </Plot>
    </Figure>
  )
}
