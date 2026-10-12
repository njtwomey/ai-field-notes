import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function KAnonymityVisualizer() {
  const state = useFigureState({
    generalisationLevel: choice<'raw' | 'moderate' | 'broad' | 'high_dim_collapse'>(
      [
        { value: 'moderate', label: 'Moderate Generalisation (Age [10y bins], ZIP 3-digit: k=4)' },
        { value: 'broad', label: 'Broad Generalisation (Age [30y bins], ZIP state: k=10)' },
        { value: 'high_dim_collapse', label: 'High-Dimension Collapse (Curse of dimensionality: Aggarwal 2005)' },
        { value: 'raw', label: 'Raw Unprotected Records (Exact age, 5-digit ZIP: k=1)' },
      ],
      'moderate',
      { label: 'anonymisation transformation' },
    ),
    enforceLdiversity: choice<'no' | 'yes'>(
      [
        { value: 'yes', label: 'Enforce l-diversity (l >= 2 sensitive values per group)' },
        { value: 'no', label: 'No l-diversity (vulnerable to homogeneity attacks)' },
      ],
      'yes',
      { label: 'l-diversity constraint' },
    ),
    targetK: slider(2, 12, 4, { step: 1, label: 'minimum k threshold' }),
  })

  const results = useMemo(() => {
    // Pareto curve of Privacy k vs Retained Machine Learning Utility (%)
    const kGrid: number[] = []
    const utilityCurve: number[] = []

    for (let k = 1; k <= 15; k++) {
      kGrid.push(k)
      // Exponential loss of utility as k increases (Aggarwal 2005)
      const u = Math.max(5, 100 * Math.exp(-0.22 * (k - 1)))
      utilityCurve.push(u)
    }

    let achievedK = 1
    let diversityL = 1
    let utilityPct = 100
    let homogeneityRisk = 'Critical (100% Certainty)'

    if (state.generalisationLevel === 'raw') {
      achievedK = 1
      diversityL = 1
      utilityPct = 99.0
      homogeneityRisk = 'Critical (Unique Records)'
    } else if (state.generalisationLevel === 'moderate') {
      achievedK = Math.max(state.targetK, 4)
      diversityL = state.enforceLdiversity === 'yes' ? 3 : 1
      utilityPct = state.enforceLdiversity === 'yes' ? 68.0 : 78.5
      homogeneityRisk = state.enforceLdiversity === 'yes' ? 'Safe (l >= 3)' : 'Homogeneity Attack (l = 1)'
    } else if (state.generalisationLevel === 'broad') {
      achievedK = Math.max(state.targetK, 10)
      diversityL = 4
      utilityPct = 38.0
      homogeneityRisk = 'Safe (High Entropy)'
    } else {
      // Aggarwal high-dimensional collapse
      achievedK = 25
      diversityL = 6
      utilityPct = 8.5 // Most cells suppressed to global root
      homogeneityRisk = 'Safe but Zero Analytical Utility'
    }

    // Points representing equivalence classes (Group size on Y, Group index on X)
    const groupX: number[] = []
    const groupSizes: number[] = []
    const N_GROUPS = 8

    for (let g = 0; g < N_GROUPS; g++) {
      groupX.push(g)
      let sz = achievedK
      if (state.generalisationLevel === 'moderate') {
        sz = achievedK + (g % 3)
      } else if (state.generalisationLevel === 'broad') {
        sz = achievedK + (g % 5) * 2
      } else if (state.generalisationLevel === 'raw') {
        sz = 1
      } else {
        sz = 30 + g * 4
      }
      groupSizes.push(sz)
    }

    return {
      kGrid,
      utilityCurve,
      achievedK,
      diversityL,
      utilityPct,
      homogeneityRisk,
      groupX,
      groupSizes,
    }
  }, [state.generalisationLevel, state.enforceLdiversity, state.targetK])

  const axisX = useAxis({
    label: 'anonymity threshold k (equivalence group minimum cardinality)',
    range: [1, 15],
  })
  const axisY = useAxis({
    label: 'retained dataset utility for downstream ML (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="k-anonymity, l-diversity, and the curse of dimensionality in disclosure control"
      state={state}
      hoverReadout={false}
      caption="Evaluating k-anonymity and l-diversity in tabular disclosure control. While k-anonymity groups records into clusters of size >= k via attribute generalization and suppression, identical sensitive values within a cluster remain vulnerable to homogeneity attacks. Enforcing l-diversity resolves homogeneity, but as attribute dimensionality increases, Aggarwal's curse of dimensionality forces generalization hierarchies to collapse, destroying downstream machine learning utility."
      readouts={
        <>
          <Readout label="achieved k-anonymity" value={`k = ${results.achievedK} records`} />
          <Readout label="sensitive diversity l" value={`l = ${results.diversityL} distinct values`} />
          <Readout label="retained ML utility" value={`${results.utilityPct.toFixed(1)}%`} />
          <Readout label="disclosure risk" value={results.homogeneityRisk} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="The fundamental Pareto frontier: Privacy k vs Analytical Utility">
        <Curve
          name="utility decay curve as k increases"
          x={results.kGrid}
          y={results.utilityCurve}
          slot={0}
          width={2.5}
        />
        <Points
          name="current operating anonymity and utility"
          x={[Math.min(15, results.achievedK)]}
          y={[results.utilityPct]}
          slot={1}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
