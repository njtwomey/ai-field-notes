import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DatasetBiasTaxonomyVisualizer() {
  const state = useFigureState({
    biasType: choice<'historical' | 'representation' | 'measurement' | 'aggregation'>(
      [
        { value: 'representation', label: 'Representation Bias (Sampling imbalance: Buolamwini & Gebru 2018)' },
        { value: 'measurement', label: 'Measurement Bias (Faulty proxy label Y != true construct Y*)' },
        { value: 'aggregation', label: 'Aggregation Bias (Homogeneous model across heterogeneous groups)' },
        { value: 'historical', label: 'Historical Bias (World state disparity: Suresh & Guttag 2021)' },
      ],
      'representation',
      { label: 'dataset bias mechanism' },
    ),
    interventionLevel: slider(0, 100, 50, {
      step: 10,
      label: 'targeted mitigation intervention strength (%)',
    }),
  })

  const results = useMemo(() => {
    // Generate disparity curves across intervention strength t in [0, 100]
    const tGrid: number[] = []
    const disparityGrid: number[] = []

    for (let t = 0; t <= 100; t += 10) {
      tGrid.push(t)
      let disp = 0
      if (state.biasType === 'representation') {
        // Disparity drops sharply with stratified resampling & target collection
        disp = 35.0 * Math.exp(-0.03 * t) + 2.0
      } else if (state.biasType === 'measurement') {
        // Proxy measurement calibration requires metric redesign
        disp = 28.0 * (1.0 - 0.008 * t)
      } else if (state.biasType === 'aggregation') {
        // Multi-task or group-conditional modeling reduces aggregation gap
        disp = 32.0 * Math.exp(-0.025 * t) + 4.0
      } else {
        // Historical bias cannot be solved by data cleaning alone
        disp = 40.0 - 0.15 * t
      }
      disparityGrid.push(disp)
    }

    const t = state.interventionLevel
    let currentDisparity = 0
    let stage = ''
    let recommendedMitigation = ''

    if (state.biasType === 'representation') {
      currentDisparity = 35.0 * Math.exp(-0.03 * t) + 2.0
      stage = 'Data Collection / Sampling Stage (Mismatched P(X))'
      recommendedMitigation = 'Stratified oversampling, diverse benchmarking, targeted active learning'
    } else if (state.biasType === 'measurement') {
      currentDisparity = 28.0 * (1.0 - 0.008 * t)
      stage = 'Feature & Label Definition Stage (Proxy gap Y != Y*)'
      recommendedMitigation = 'Construct validity review, differential measurement calibration'
    } else if (state.biasType === 'aggregation') {
      currentDisparity = 32.0 * Math.exp(-0.025 * t) + 4.0
      stage = "Model Formulation & Pooling Stage (Simpson's paradox)"
      recommendedMitigation = 'Mixture-of-experts, group-conditioned heads, multi-task learning'
    } else {
      currentDisparity = 40.0 - 0.15 * t
      stage = 'Pre-Data Social Generation Stage (World state disparity)'
      recommendedMitigation = 'Outcome reweighting, counterfactual causal bounds, policy constraints'
    }

    return {
      tGrid,
      disparityGrid,
      currentT: t,
      currentDisparity,
      stage,
      recommendedMitigation,
    }
  }, [state.biasType, state.interventionLevel])

  const axisX = useAxis({
    label: 'targeted mitigation intervention strength (%)',
    range: [0, 100],
  })
  const axisY = useAxis({
    label: 'subgroup error rate disparity gap (percentage points)',
    range: [0, 45],
  })

  return (
    <Figure
      title="Taxonomy of dataset bias and stages of entry across the ML lifecycle"
      state={state}
      hoverReadout={false}
      caption="Deconstructing sources of dataset bias (Suresh & Guttag 2021). Representation bias enters during cohort sampling when minority demographics are under-sampled, causing sharp error spikes on underrepresented populations (e.g. Buolamwini & Gebru 2018). Measurement bias enters when chosen feature/label proxies deviate systematically across groups. Aggregation bias occurs when distinct subgroup relationships are collapsed into a single global model. Crucially, historical bias resides in the real world: perfect sampling and measurement will still replicate past social disparities unless explicitly counter-weighted."
      readouts={
        <>
          <Readout label="lifecycle entry stage" value={results.stage} />
          <Readout label="subgroup error gap" value={`${results.currentDisparity.toFixed(1)}% discrepancy`} />
          <Readout label="recommended remedy" value={results.recommendedMitigation} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Disparity reduction under targeted stage interventions">
        <Curve
          name="subgroup disparity curve"
          x={results.tGrid}
          y={results.disparityGrid}
          slot={state.biasType === 'historical' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current intervention point"
          x={[results.currentT]}
          y={[results.currentDisparity]}
          slot={state.biasType === 'historical' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
