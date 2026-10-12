import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DatasetSearchVisualizer() {
  const state = useFigureState({
    discoveryType: choice<'joinable' | 'unionable' | 'fanout'>(
      [
        { value: 'joinable', label: 'Joinable Table Search (MinHash Containment & Overlap)' },
        { value: 'unionable', label: 'Unionable Table Search (Schema & Semantic Alignment)' },
        { value: 'fanout', label: 'Data Augmentation Risk (1-to-N Join Fan-Out Inflation)' },
      ],
      'joinable',
      { label: 'discovery paradigm' },
    ),
    containmentCutoff: slider(0.2, 0.95, 0.7, {
      step: 0.05,
      label: 'containment threshold C(Q, T)',
    }),
    fanoutMultiplier: slider(1.0, 5.0, 2.5, {
      step: 0.5,
      label: 'join fan-out multiplier (rows per key)',
    }),
  })

  const { discoveryType, containmentCutoff, fanoutMultiplier } = state

  const results = useMemo(() => {
    if (discoveryType === 'joinable') {
      // 5 candidate data lake tables evaluated against query key (postal_code)
      // T1: Regional Demographics (Containment 0.95, Jaccard 0.82)
      // T2: Meteorological Weather (Containment 0.78, Jaccard 0.65)
      // T3: Real Estate Valuations (Containment 0.72, Jaccard 0.58)
      // T4: Internal Employee Roster (Containment 0.14, Jaccard 0.06)
      // T5: Mobile Telemetry Beacons (Containment 0.05, Jaccard 0.02)
      const tables = [1, 2, 3, 4, 5]
      const containmentScores = [0.95, 0.78, 0.72, 0.14, 0.05]
      const threshLine = tables.map(() => containmentCutoff)

      const qualified = containmentScores.filter((s) => s >= containmentCutoff).length

      return {
        x: tables,
        y: containmentScores,
        baseline: threshLine,
        qualified,
        diagnostic: `${qualified} tables pass containment threshold; suitable for feature enrichment joins`,
      }
    } else if (discoveryType === 'unionable') {
      // 5 candidate tables evaluated for unionability (semantic type compatibility & schema match)
      // T1: 2025 Transaction Archive (Unionability 0.98)
      // T2: Partner Retail Stream (Unionability 0.85)
      // T3: Mobile App Event Dump (Unionability 0.62)
      // T4: Inventory Stock Ledger (Unionability 0.22)
      // T5: System Access Logs (Unionability 0.04)
      const tables = [1, 2, 3, 4, 5]
      const unionScores = [0.98, 0.85, 0.62, 0.22, 0.04]
      const threshLine = tables.map(() => 0.75)

      return {
        x: tables,
        y: unionScores,
        baseline: threshLine,
        qualified: unionScores.filter((s) => s >= 0.75).length,
        diagnostic: 'Tables 1 and 2 are union-compatible; safe to append rows to expand training sample size',
      }
    } else {
      // Fanout risk across 5 join stages (Base N = 10k rows)
      // J1: 1-to-1 customer demographics (10k rows)
      // J2: 1-to-1 account metadata (10k rows)
      // J3: 1-to-N transaction history (10k * fanoutMultiplier)
      // J4: 1-to-N support tickets (10k * fanoutMultiplier * 1.5)
      // J5: 1-to-N web clicks (10k * fanoutMultiplier * 2.2)
      const stages = [1, 2, 3, 4, 5]
      const baseRow = 10
      const rowCounts = [
        baseRow,
        baseRow,
        baseRow * fanoutMultiplier,
        baseRow * fanoutMultiplier * 1.4,
        baseRow * fanoutMultiplier * 2.0,
      ]
      const originalLine = stages.map(() => baseRow)

      return {
        x: stages,
        y: rowCounts,
        baseline: originalLine,
        qualified: 0,
        diagnostic:
          fanoutMultiplier > 1.5
            ? `WARNING: Row multiplication risk (${(fanoutMultiplier * 2).toFixed(1)}x expansion). Generates duplicate entity keys across splits!`
            : 'Clean 1-to-1 join profile',
      }
    }
  }, [discoveryType, containmentCutoff, fanoutMultiplier])

  const axisX = useAxis({
    label:
      discoveryType === 'fanout'
        ? 'Join Augmentation Sequence (1: Base, 2: Meta, 3: Transactions, 4: Tickets, 5: Clicks)'
        : 'Candidate Data Lake Tables (T1 to T5)',
    range: [0.5, 5.5],
  })

  const axisY = useAxis({
    label:
      discoveryType === 'fanout'
        ? 'Resulting Table Row Count (Thousands)'
        : discoveryType === 'joinable'
          ? 'Key Containment Score C(Q, T)'
          : 'Schema Unionability Alignment Score',
    range: discoveryType === 'fanout' ? [0, 110] : [0, 1.05],
  })

  return (
    <Figure
      title="Dataset Discovery & Table Augmentation Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating automated data discovery across joinable table containment, unionable schema compatibility, and the critical hazard of 1-to-N join fan-out row multiplication."
      readouts={
        <>
          <Readout label="Discovery Status" value={results.diagnostic} />
          {discoveryType === 'joinable' && (
            <Readout
              label="Qualified Join Tables"
              value={`${results.qualified} tables (C ≥ ${containmentCutoff.toFixed(2)})`}
            />
          )}
          {discoveryType === 'unionable' && (
            <Readout label="Union Candidates" value="Tables 1 & 2 share aligned entity schemas" />
          )}
          {discoveryType === 'fanout' && (
            <Readout
              label="Split Contamination Hazard"
              value={
                fanoutMultiplier > 1.5
                  ? 'CRITICAL: Must group-split by primary entity ID'
                  : 'Low risk of entity leakage'
              }
              color={fanoutMultiplier > 1.5 ? 'var(--red-600)' : 'var(--blue-600)'}
            />
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Discovery Metric Profile">
        <Curve
          name="Threshold / Original Scale Reference"
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Points
          name="Evaluated Table Metrics"
          x={results.x}
          y={results.y}
          slot={discoveryType === 'fanout' && fanoutMultiplier > 1.5 ? 1 : 0}
          size={8}
        />
        {discoveryType === 'fanout' && (
          <Curve name="Expanded Row Count Path" x={results.x} y={results.y} slot={0} width={2.2} />
        )}
      </Plot>
    </Figure>
  )
}
