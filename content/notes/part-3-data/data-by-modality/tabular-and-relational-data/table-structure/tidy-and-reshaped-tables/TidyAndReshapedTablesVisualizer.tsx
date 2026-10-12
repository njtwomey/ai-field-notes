import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TidyAndReshapedTablesVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'wide_long_transform' | 'group_leakage' | 'pivot_sparsity'>(
      [
        { value: 'wide_long_transform', label: 'Wide vs Long Layout: Row Count vs Dimensionality (Wickham 2014)' },
        { value: 'group_leakage', label: 'Long Format Row-Splitting vs Patient Group Leakage' },
        { value: 'pivot_sparsity', label: 'Pivot Sparsity & Structural NaN Explosion' },
      ],
      'wide_long_transform',
      { label: 'structural perspective' },
    ),
    entityCount: slider(10, 100, 25, {
      step: 5,
      label: 'unique entities / patients N_entities',
    }),
    timeHorizon: slider(2, 12, 4, {
      step: 1,
      label: 'repeated observation periods T',
    }),
  })

  const { visualizerMode, entityCount, timeHorizon } = state

  const results = useMemo(() => {
    if (visualizerMode === 'wide_long_transform') {
      // Time horizons from 2 to 12
      const horizons = [2, 3, 4, 5, 6, 8, 10, 12]
      // Wide layout: row count stays constant = entityCount, columns = horizons + 1
      const wideRows = horizons.map(() => entityCount)
      // Long layout: row count scales multiplicatively = entityCount * T
      const longRows = horizons.map((t) => entityCount * t)

      const currWide = entityCount
      const currLong = entityCount * timeHorizon

      return {
        x: horizons,
        y: longRows,
        baseline: wideRows,
        stat1: `Entities N = ${entityCount}, Periods T = ${timeHorizon}`,
        stat2: `Wide rows: ${currWide} (Unit: Entity i)`,
        stat3: `Long rows: ${currLong} (Unit: Entity-Period i x t)`,
        warning:
          timeHorizon >= 8
            ? 'Long layout multiplies row volume by 8x; ideal for sequential/panel models, GBDTs require wide aggregation'
            : 'Pivoting transforms the fundamental unit of observation between instances and episodes',
        axisXLabel: 'Repeated Measurement Periods per Entity (T)',
        axisYLabel: 'Dataset Total Row Count (Observations)',
        xRange: [1.5, 12.5] as [number, number],
        yRange: [0, Math.max(...longRows) * 1.05] as [number, number],
      }
    } else if (visualizerMode === 'group_leakage') {
      // Test split fraction: 10% to 50%
      const splitFractions = [10, 20, 25, 30, 40, 50]
      const T = timeHorizon

      // Naive uniform row splitting on long format: probability that an entity has rows in BOTH train and test!
      // Leakage rate = 1 - (p_train^T + p_test^T)
      const leakageRate = splitFractions.map((f) => {
        const p_test = f / 100
        const p_train = 1 - p_test
        const unshared = Math.pow(p_train, T) + Math.pow(p_test, T)
        return (1 - unshared) * 100
      })
      const groupSplitClean = splitFractions.map(() => 0)

      const currPTest = 0.2
      const currPTrain = 0.8
      const currLeakage = (1 - (Math.pow(currPTrain, T) + Math.pow(currPTest, T))) * 100

      return {
        x: splitFractions,
        y: leakageRate,
        baseline: groupSplitClean,
        stat1: `Observation count T = ${T} per entity`,
        stat2: `Naive row-split leakage: ${currLeakage.toFixed(1)}% entities`,
        stat3: 'Group-aware split: 0% leakage',
        warning: 'CRITICAL TRAP: Random row splits on long tables leak entity identity across train/test folds',
        axisXLabel: 'Evaluation Test Split Fraction (%)',
        axisYLabel: 'Entity Overlap Contamination Rate (%)',
        xRange: [5, 55] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Key cardinality K from 2 to 20
      const keys = [2, 4, 6, 8, 10, 12, 16, 20]
      // Observation density per entity (e.g. 3 observations out of K)
      const density = 2.5
      // Sparsity (% of NaN cells in pivoted wide matrix): 1 - (density / K)
      const sparsity = keys.map((k) => Math.max(0, (1 - density / k) * 100))
      const safeBaseline = keys.map(() => 20.0)

      return {
        x: keys,
        y: sparsity,
        baseline: safeBaseline,
        stat1: 'Pivoting on categorical keys',
        stat2: `At K=20 keys: ${sparsity[7].toFixed(1)}% NaN cells`,
        stat3: 'High-cardinality sparse matrix',
        warning: 'Pivoting high-cardinality keys creates massive, sparse tables dominated by missing values',
        axisXLabel: 'Pivoted Category Key Cardinality (K)',
        axisYLabel: 'Pivoted Matrix Sparsity Rate (% Missing Values)',
        xRange: [1, 21] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [visualizerMode, entityCount, timeHorizon])

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
      title="Tidy Data Reshaping & Unit-of-Observation Inspector"
      state={state}
      hoverReadout={false}
      caption="Evaluating table structural reshaping: Multiplicative row expansion between wide and long layouts, catastrophic entity leakage under naive long table row splits, and pivot matrix sparsity."
      readouts={
        <>
          <Readout label="Layout Mode" value={results.stat1} />
          <Readout label="Observation Scale" value={results.stat2} />
          <Readout label="Integrity Dynamic" value={results.stat3} />
          <Readout
            label="Structural Directive"
            value={results.warning}
            color={visualizerMode === 'group_leakage' ? 'var(--red-600)' : 'var(--blue-600)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Table Reshaping Trajectory">
        <Curve
          name={
            visualizerMode === 'wide_long_transform'
              ? 'Wide Table Static Entity Row Count'
              : visualizerMode === 'group_leakage'
                ? 'Group-Aware Entity Split (0% Leakage)'
                : 'Acceptable Baseline Sparsity'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'wide_long_transform'
              ? 'Long Table Expanded Row Volume (N x T)'
              : visualizerMode === 'group_leakage'
                ? 'Naive Random Row-Split Contamination'
                : 'Pivoted Matrix Sparsity Rate'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Coordinates" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
