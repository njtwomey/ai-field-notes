import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataRightsUnlearningVisualizer() {
  const state = useFigureState({
    unlearningStrategy: choice<'database_only' | 'exact_scratch' | 'sisa_sharded' | 'influence_approx'>(
      [
        { value: 'sisa_sharded', label: 'SISA Sharded Retraining (Exact unlearning at 10% compute: Bourtoule 2021)' },
        { value: 'exact_scratch', label: 'Exact Full Retraining (100% compute overhead per batch)' },
        { value: 'influence_approx', label: 'Approximate Influence Scrubbing (Hessian inversion: Koh 2017)' },
        { value: 'database_only', label: 'Database Deletion Only (Model retains memory: Non-compliant)' },
      ],
      'sisa_sharded',
      { label: 'erasure execution method' },
    ),
    deletionRequests: slider(10, 200, 50, {
      step: 10,
      label: 'cumulative erasure requests (DSAR rows)',
    }),
  })

  const results = useMemo(() => {
    // Generate Pareto trade-off curve across batch request sizes K in [10, 200]
    const kGrid: number[] = []
    const computeCostGrid: number[] = []

    for (let k = 10; k <= 200; k += 10) {
      kGrid.push(k)
      let cost = 0
      if (state.unlearningStrategy === 'database_only') {
        cost = 0.5
      } else if (state.unlearningStrategy === 'exact_scratch') {
        cost = 100.0 // Full retraining required
      } else if (state.unlearningStrategy === 'sisa_sharded') {
        // SISA: roughly 1/S of training cost per affected shard
        cost = Math.min(100, 6.0 + (k / 200) * 14.0)
      } else {
        // Influence scrubbing: fast gradient step
        cost = 2.5 + (k / 200) * 4.0
      }
      computeCostGrid.push(cost)
    }

    const K = state.deletionRequests
    let currentCost = 0
    let miaVulnerability = 0
    let complianceStatus = 'Compliant'

    if (state.unlearningStrategy === 'database_only') {
      currentCost = 0.5
      miaVulnerability = 96.5 // Subject still easily re-identified from model weights!
      complianceStatus = 'Illegal Non-Compliance (Parametric Memory Retained)'
    } else if (state.unlearningStrategy === 'exact_scratch') {
      currentCost = 100.0
      miaVulnerability = 50.0 // Baseline random guess
      complianceStatus = 'Full Exact Compliance (Provably Erased)'
    } else if (state.unlearningStrategy === 'sisa_sharded') {
      currentCost = 6.0 + (K / 200) * 14.0
      miaVulnerability = 50.0 // SISA guarantees exact retraining of affected shard
      complianceStatus = 'Full Exact Compliance (Sharded Partitioning)'
    } else {
      currentCost = 2.5 + (K / 200) * 4.0
      miaVulnerability = 53.5 // Residual trace minimal
      complianceStatus = 'Certified Approximate Compliance'
    }

    return {
      kGrid,
      computeCostGrid,
      currentCost,
      miaVulnerability,
      complianceStatus,
      currentK: K,
    }
  }, [state.unlearningStrategy, state.deletionRequests])

  const axisX = useAxis({
    label: 'cumulative deletion requests (data points to unlearn)',
    range: [0, 200],
  })
  const axisY = useAxis({
    label: 'relative computational unlearning cost (% of full training)',
    range: [0, 110],
  })

  return (
    <Figure
      title="Data rights, erasure requests, and machine unlearning architectures"
      state={state}
      hoverReadout={false}
      caption="Evaluating machine unlearning architectures under Article 17 erasure requests. Merely deleting rows from database tables leaves parametric memory intact, exposing subjects to membership inference attacks (MIA accuracy > 95%). Full retraining from scratch guarantees unlearning but incurs 100% compute cost per request. The SISA (Sharded, Isolated, Sliced, Aggregated) architecture achieves provably exact unlearning while reducing retraining compute by ~85%."
      readouts={
        <>
          <Readout label="computational cost" value={`${results.currentCost.toFixed(1)}% of full training`} />
          <Readout
            label="membership inference vulnerability"
            value={`${results.miaVulnerability.toFixed(1)}% MIA accuracy (50% = random)`}
          />
          <Readout label="regulatory compliance" value={results.complianceStatus} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Unlearning compute cost as deletion requests scale">
        <Curve
          name="computational cost curve (% full training)"
          x={results.kGrid}
          y={results.computeCostGrid}
          slot={state.unlearningStrategy === 'exact_scratch' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current deletion request operating point"
          x={[results.currentK]}
          y={[results.currentCost]}
          slot={state.unlearningStrategy === 'exact_scratch' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
