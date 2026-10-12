import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DatasetVersioningVisualizer() {
  const state = useFigureState({
    versioningModel: choice<'content_hash' | 'time_travel' | 'mutable'>(
      [
        {
          value: 'time_travel',
          label: 'Transaction Log Time-Travel (Delta / Iceberg commit log: Polyzotis et al. 2018)',
        },
        { value: 'content_hash', label: 'Content-Addressable Hashes (SHA-256 Merkle DAG: Zaharia et al. 2018)' },
        { value: 'mutable', label: 'Unversioned Overwrites (Direct database updates: destroys auditability)' },
      ],
      'time_travel',
      { label: 'dataset versioning architecture' },
    ),
    asOfDaysAgo: slider(0, 90, 30, {
      step: 5,
      label: 'as-of time-travel query horizon (days in the past)',
    }),
  })

  const results = useMemo(() => {
    // Generate cumulative storage footprint across snapshot commits C in [1, 10]
    const cGrid: number[] = []
    const storageFootprintGrid: number[] = []

    for (let c = 1; c <= 10; c += 1) {
      cGrid.push(c)
      let storage = 0
      if (state.versioningModel === 'mutable') {
        // Flat constant size, but historical versions permanently lost
        storage = 100.0
      } else if (state.versioningModel === 'time_travel') {
        // Copy-on-write / Parquet delta sharing: only changed row groups stored
        storage = 100.0 + (c - 1) * 8.5
      } else {
        // Content-hash deduplication: shared chunks deduplicated via SHA-256
        storage = 100.0 + (c - 1) * 12.0
      }
      storageFootprintGrid.push(storage)
    }

    const days = state.asOfDaysAgo
    let reproducibility = ''
    let lineageStatus = ''
    let storageOverhead = ''

    if (state.versioningModel === 'mutable') {
      reproducibility = '0% Reproducible (Historical state permanently overwritten)'
      lineageStatus = 'Broken Lineage: Unable to reproduce model trained on past data'
      storageOverhead = '1.0x (No history preserved)'
    } else if (state.versioningModel === 'time_travel') {
      reproducibility = '100% Bit-Exact Reproducibility'
      lineageStatus = `Valid As-Of Query: Restored snapshot from ${days} days ago via commit log`
      storageOverhead = '1.7x (Copy-on-write Parquet file sharing across 90 days)'
    } else {
      reproducibility = '100% Cryptographically Certified'
      lineageStatus = `Merkle DAG Verified: SHA-256 commit hash pinned to model artifact`
      storageOverhead = '1.9x (Chunk deduplication across dataset revisions)'
    }

    return {
      cGrid,
      storageFootprintGrid,
      currentCommit: Math.min(10, Math.max(1, Math.round(10 - days / 10))),
      reproducibility,
      lineageStatus,
      storageOverhead,
    }
  }, [state.versioningModel, state.asOfDaysAgo])

  const axisX = useAxis({
    label: 'dataset revision commits over time',
    range: [1, 10],
  })
  const axisY = useAxis({
    label: 'cumulative storage footprint (GB)',
    range: [80, 240],
  })

  return (
    <Figure
      title="Dataset versioning, copy-on-write storage, and model lineage provenance"
      state={state}
      hoverReadout={false}
      caption="Evaluating dataset versioning architectures for machine learning reproducibility (Polyzotis et al. 2018, Zaharia et al. 2018). In production, models must be auditably linked to the exact training data that produced them. Unversioned mutable databases overwrite records, permanently destroying reproducibility. Modern lakehouses (Delta Lake, Apache Iceberg) use append-only transaction logs to enable 'as-of' time-travel queries, preserving historical snapshots via copy-on-write file sharing without duplicating full terabyte datasets."
      readouts={
        <>
          <Readout label="reproducibility guarantee" value={results.reproducibility} />
          <Readout label="lineage provenance state" value={results.lineageStatus} />
          <Readout label="storage efficiency" value={results.storageOverhead} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Storage footprint growth across dataset revisions">
        <Curve
          name="storage footprint curve (GB)"
          x={results.cGrid}
          y={results.storageFootprintGrid}
          slot={state.versioningModel === 'mutable' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="active snapshot commit point"
          x={[results.currentCommit]}
          y={[results.storageFootprintGrid[results.currentCommit - 1]]}
          slot={state.versioningModel === 'mutable' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
