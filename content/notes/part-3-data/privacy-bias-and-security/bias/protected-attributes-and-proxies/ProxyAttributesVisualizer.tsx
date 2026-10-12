import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ProxyAttributesVisualizer() {
  const state = useFigureState({
    unawarenessPolicy: choice<'blind' | 'aware_penalized' | 'naive_aware'>(
      [
        { value: 'blind', label: 'Blind Model (Protected attribute A removed: "Fairness through Unawareness")' },
        { value: 'naive_aware', label: 'Unconstrained Model (A included directly as explicit predictor)' },
        {
          value: 'aware_penalized',
          label: 'Aware Penalized Model (A used during training to enforce demographic parity)',
        },
      ],
      'blind',
      { label: 'training regime' },
    ),
    proxyCorrelation: slider(0.0, 0.95, 0.7, {
      step: 0.05,
      label: 'correlation between proxy feature X_proxy and protected attribute A (rho)',
    }),
  })

  const results = useMemo(() => {
    const rhoGrid: number[] = []
    const diGrid: number[] = []

    // Base true qualification difference or historical disparity between groups
    const baseDisparity = 0.4

    for (let r = 0.0; r <= 0.95; r += 0.05) {
      rhoGrid.push(r)
      let di = 1.0
      if (state.unawarenessPolicy === 'aware_penalized') {
        // Active mitigation maintains near parity across all correlation levels
        di = Math.max(0.85, 1.0 - 0.1 * r)
      } else if (state.unawarenessPolicy === 'blind') {
        // When blind, proxy correlation reconstructs A: disparity scales with r^2
        const leak = r * r * baseDisparity
        const p1 = 0.8
        const p0 = Math.max(0.1, p1 - leak)
        di = p0 / p1
      } else {
        // Direct inclusion leaks maximum disparity regardless of proxy
        const leak = baseDisparity
        const p1 = 0.8
        const p0 = p1 - leak
        di = p0 / p1
      }
      diGrid.push(di)
    }

    const currentRho = state.proxyCorrelation
    let currentDi = 1.0
    if (state.unawarenessPolicy === 'aware_penalized') {
      currentDi = Math.max(0.85, 1.0 - 0.1 * currentRho)
    } else if (state.unawarenessPolicy === 'blind') {
      const leak = currentRho * currentRho * baseDisparity
      const p1 = 0.8
      const p0 = Math.max(0.1, p1 - leak)
      currentDi = p0 / p1
    } else {
      const p1 = 0.8
      const p0 = p1 - baseDisparity
      currentDi = p0 / p1
    }

    const proxyR2 = currentRho * currentRho
    const fourFifthsCompliant = currentDi >= 0.8

    return {
      rhoGrid,
      diGrid,
      currentRho,
      currentDi,
      proxyR2,
      fourFifthsCompliant,
    }
  }, [state.unawarenessPolicy, state.proxyCorrelation])

  const axisX = useAxis({
    label: 'correlation rho = Corr(X_proxy, A)',
    range: [0, 1],
  })
  const axisY = useAxis({
    label: 'disparate impact ratio P(Y=1 | A=0) / P(Y=1 | A=1)',
    range: [0.4, 1.05],
  })

  return (
    <Figure
      title="Fairness through unawareness versus proxy feature leakage"
      state={state}
      hoverReadout={false}
      caption="Evaluating the fallacy of 'fairness through unawareness'. Merely deleting protected attribute A from training data fails to prevent discriminatory impact if remaining features X_proxy correlate with A (e.g. ZIP code, education history, credit tier). As proxy correlation rho rises, a blind model reconstructs A with coefficient of determination R² = rho², driving the disparate impact ratio below the US EEOC four-fifths threshold (0.80). Paradoxically, enforcing fair treatment requires collecting A to audit and constrain demographic parity."
      readouts={
        <>
          <Readout
            label="disparate impact ratio"
            value={`${results.currentDi.toFixed(2)} (${results.currentDi >= 0.8 ? 'Compliant' : 'Adverse Impact'})`}
          />
          <Readout
            label="proxy reconstructibility (R²)"
            value={`${(results.proxyR2 * 100).toFixed(1)}% of A variance explained`}
          />
          <Readout
            label="EEOC four-fifths rule"
            value={results.fourFifthsCompliant ? 'Passed (DI ≥ 0.80)' : 'Violated (DI < 0.80)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Disparate impact ratio as proxy correlation rises">
        <Curve
          name="disparate impact curve"
          x={results.rhoGrid}
          y={results.diGrid}
          slot={results.fourFifthsCompliant ? 0 : 1}
          width={2.5}
        />
        <Points
          name="current proxy operating point"
          x={[results.currentRho]}
          y={[results.currentDi]}
          slot={results.fourFifthsCompliant ? 0 : 1}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
