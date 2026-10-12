import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SchemaEvolutionVisualizer() {
  const state = useFigureState({
    changeCategory: choice<'column_rename' | 'unit_change' | 'enum_change' | 'backfill'>(
      [
        { value: 'unit_change', label: 'Unit Scale Change (e.g. seconds to milliseconds: 1,000x numerical shift)' },
        { value: 'column_rename', label: 'Renamed / Removed Column (Silent null defaults: Breck et al. 2017)' },
        { value: 'enum_change', label: 'Enumeration / Category Mutation (New category becomes unmapped zero-vector)' },
        { value: 'backfill', label: 'Historical Backfill Discontinuity (Point-in-time boundary mismatch)' },
      ],
      'unit_change',
      { label: 'schema mutation type' },
    ),
    affectedTraffic: slider(10, 100, 50, {
      step: 10,
      label: 'fraction of production traffic affected by schema change (%)',
    }),
  })

  const results = useMemo(() => {
    // Generate prediction distortion curves across affected traffic p in [10, 100]%
    const pGrid: number[] = []
    const distortionGrid: number[] = []

    for (let p = 10; p <= 100; p += 10) {
      pGrid.push(p)
      let dist = 0

      if (state.changeCategory === 'unit_change') {
        // Massive numerical distortion directly scales with affected traffic
        dist = Math.min(100, p * 0.95)
      } else if (state.changeCategory === 'column_rename') {
        // Missing column imputed with default zeros/means degrades model
        dist = Math.min(100, p * 0.65)
      } else if (state.changeCategory === 'enum_change') {
        // OOV categories lose categorical signal
        dist = Math.min(100, p * 0.42)
      } else {
        // Backfill boundary creates step-function discontinuity
        dist = Math.min(100, p * 0.55)
      }

      distortionGrid.push(dist)
    }

    const p = state.affectedTraffic
    let currentDist = 0
    let compatibilityStatus = ''
    let contractDefense = ''

    if (state.changeCategory === 'unit_change') {
      currentDist = Math.min(100, p * 0.95)
      compatibilityStatus = 'Breaking Change (Silent 1,000x numerical magnitude shift)'
      contractDefense = 'Semantic Contracts: Data quality assertion validates physical units & ranges'
    } else if (state.changeCategory === 'column_rename') {
      currentDist = Math.min(100, p * 0.65)
      compatibilityStatus = 'Breaking Change (Downstream readers observe 100% null values)'
      contractDefense = 'Schema Registry: Enforce backward compatibility (reject unaliased drops)'
    } else if (state.changeCategory === 'enum_change') {
      currentDist = Math.min(100, p * 0.42)
      compatibilityStatus = 'Partially Compatible (New enum values map to unknown category)'
      contractDefense = 'Enum Schemas: Strict domain constraints alert on unmapped tokens'
    } else {
      currentDist = Math.min(100, p * 0.55)
      compatibilityStatus = 'Temporal Boundary Skew (Historical training data differs from live stream)'
      contractDefense = 'Point-in-Time Audits: Verify backfilled timestamps align with original logging'
    }

    return {
      pGrid,
      distortionGrid,
      currentP: p,
      currentDist,
      compatibilityStatus,
      contractDefense,
    }
  }, [state.changeCategory, state.affectedTraffic])

  const axisX = useAxis({
    label: 'affected production traffic (%)',
    range: [10, 100],
  })
  const axisY = useAxis({
    label: 'model prediction error rate / distortion (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Schema evolution: mutation failure modes and downstream model distortion"
      state={state}
      hoverReadout={false}
      caption="Evaluating the impact of schema mutations on downstream machine learning pipelines (Polyzotis et al. 2018, Breck et al. 2017). Unlike databases where schema changes produce explicit SQL errors, machine learning pipelines fail silently. Unit changes (e.g. converting duration from seconds to milliseconds) shift numerical scales by 1,000x, pushing inputs far beyond the model's support. Enforcing schema registries and semantic contracts prevents unversioned schema alterations from entering production."
      readouts={
        <>
          <Readout label="compatibility classification" value={results.compatibilityStatus} />
          <Readout label="prediction distortion" value={`${results.currentDist.toFixed(1)}% error rate`} />
          <Readout label="governance defense" value={results.contractDefense} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Model prediction distortion vs affected traffic fraction">
        <Curve
          name="prediction distortion curve (%)"
          x={results.pGrid}
          y={results.distortionGrid}
          slot={results.currentDist > 50 ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current traffic operating point"
          x={[results.currentP]}
          y={[results.currentDist]}
          slot={results.currentDist > 50 ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
