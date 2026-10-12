import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SemiStructuredDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<
      'flattening_sparsity_explosion' | 'schema_drift_polymorphism' | 'array_aggregation_tradeoff'
    >(
      [
        { value: 'flattening_sparsity_explosion', label: 'Dot-Path Flattening: Key Diversity vs Matrix Sparsity' },
        {
          value: 'schema_drift_polymorphism',
          label: 'Schema Inference: Polymorphic Types & Lattice Conflicts (Armbrust 2015)',
        },
        {
          value: 'array_aggregation_tradeoff',
          label: 'Nested Arrays: Row Explosion vs Deep Set Pooling (Zaheer 2017)',
        },
      ],
      'flattening_sparsity_explosion',
      { label: 'semi-structured transformation' },
    ),
    recordVolume: slider(100, 1000, 400, {
      step: 50,
      label: 'ingested documents N',
    }),
    heterogeneityFactor: slider(2, 20, 8, {
      step: 1,
      label: 'attribute / key diversity factor K',
    }),
  })

  const { visualizerMode, recordVolume, heterogeneityFactor } = state

  const results = useMemo(() => {
    if (visualizerMode === 'flattening_sparsity_explosion') {
      const factors = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
      const totalColumns = factors.map((f) => Math.round(5 + f * 12 + Math.pow(f, 1.4)))
      const avgKeysPerDoc = 8
      const sparsityPercent = totalColumns.map((cols) => {
        const present = Math.min(avgKeysPerDoc + heterogeneityFactor * 0.2, cols)
        return Math.min(99.5, Math.max(0, (1 - present / cols) * 100))
      })

      const currCols = Math.round(5 + heterogeneityFactor * 12 + Math.pow(heterogeneityFactor, 1.4))
      const currPresent = Math.min(avgKeysPerDoc + heterogeneityFactor * 0.2, currCols)
      const currSparsity = (1 - currPresent / currCols) * 100

      return {
        x: factors,
        y: sparsityPercent,
        baseline: factors.map(() => 50),
        stat1: `Documents N = ${recordVolume}`,
        stat2: `Union schema columns: ${currCols}`,
        stat3: `Matrix sparsity: ${currSparsity.toFixed(1)}%`,
        warning:
          currSparsity > 85
            ? 'Severe sparsity explosion: Flattening arbitrary JSON maps generates thousands of columns with >85% null values'
            : 'Moderate sparsity: Manageable dot-path flattening, but tree-based models suffer partition degradation',
        axisXLabel: 'Key Diversity Factor K (Nested Object Schema Variance)',
        axisYLabel: 'Tabular Sparsity Rate (% Null Cells)',
        xRange: [1.5, 20.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'schema_drift_polymorphism') {
      const batchSizes = [50, 100, 200, 300, 500, 750, 1000]
      const pPerDoc = 0.002 * heterogeneityFactor
      const conflictProbability = batchSizes.map((bs) => {
        const noConflict = Math.pow(1 - pPerDoc, bs)
        return Math.min(100, Math.max(0, (1 - noConflict) * 100))
      })

      const currConflict = (1 - Math.pow(1 - pPerDoc, recordVolume)) * 100

      return {
        x: batchSizes,
        y: conflictProbability,
        baseline: batchSizes.map(() => 0),
        stat1: `Batch size N = ${recordVolume}`,
        stat2: `Per-record type drift rate: ${(pPerDoc * 100).toFixed(2)}%`,
        stat3: `Lattice conflict risk: ${currConflict.toFixed(1)}%`,
        warning:
          currConflict > 70
            ? 'High risk of schema collision: downstream typed engines (Arrow/Spark) will throw type mismatches or cast to string'
            : 'Low risk: Schema-on-read inference produces stable union types across modest batch sizes',
        axisXLabel: 'Ingested Documents Batch Size N',
        axisYLabel: 'Probability of Polymorphic Type Collision (%)',
        xRange: [40, 1020] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      const docCounts = [100, 200, 300, 400, 600, 800, 1000]
      const L = heterogeneityFactor
      const explodedRows = docCounts.map((n) => n * L)
      const pooledRows = docCounts.map((n) => n)

      const currExploded = recordVolume * L
      const currPooled = recordVolume

      return {
        x: docCounts,
        y: explodedRows,
        baseline: pooledRows,
        stat1: `Average array length L = ${L} items`,
        stat2: `Exploded rows: ${currExploded.toLocaleString()}`,
        stat3: `Pooled rows: ${currPooled.toLocaleString()}`,
        warning:
          L >= 10
            ? 'Combinatorial row explosion: Exploding arrays multiplies dataset size by 10x and duplicates entity features'
            : 'Permutation-invariant pooling (mean/sum/Deep Sets) maintains fixed rectangular tensor dimensions',
        axisXLabel: 'Parent Document Count N',
        axisYLabel: 'Output Tabular Row Count',
        xRange: [80, 1020] as [number, number],
        yRange: [0, Math.max(...explodedRows) * 1.05] as [number, number],
      }
    }
  }, [visualizerMode, recordVolume, heterogeneityFactor])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Semi-Structured Data: Schema Inference, Flattening Sparsity, and Array Pooling Dynamics"
      state={state}
      hoverReadout={false}
      caption="Dynamics of semi-structured preprocessing: Trade-offs between dot-path flattening sparsity explosion, polymorphic schema unification conflicts, and nested array explosion vs permutation-invariant pooling."
      readouts={
        <>
          <Readout label="Record Scope" value={results.stat1} />
          <Readout label="Structure Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'flattening_sparsity_explosion'
                ? 'Sparsity'
                : visualizerMode === 'schema_drift_polymorphism'
                  ? 'Conflict Risk'
                  : 'Pooled Rows'
            }
            value={results.stat3}
            color={visualizerMode === 'array_aggregation_tradeoff' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Architecture Warning" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Semi-Structured Representation Profile">
        <Curve
          name={
            visualizerMode === 'flattening_sparsity_explosion'
              ? '50% Sparsity Reference Threshold'
              : visualizerMode === 'schema_drift_polymorphism'
                ? 'Zero-Conflict Baseline'
                : 'Permutation-Invariant Pooled Row Count'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'flattening_sparsity_explosion'
              ? 'Pivoted Matrix Sparsity Rate'
              : visualizerMode === 'schema_drift_polymorphism'
                ? 'Type Collision Probability'
                : 'Exploded Row Count (N x L)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
