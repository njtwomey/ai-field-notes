import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataValuationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'valuation_distribution' | 'loo_vs_shapley' | 'influence_test'>(
      [
        { value: 'valuation_distribution', label: 'Data Shapley Distribution & Noise Identification (Ghorbani 2019)' },
        { value: 'loo_vs_shapley', label: 'Leave-One-Out (LOO) Failure on Redundant Duplicates' },
        { value: 'influence_test', label: 'Influence Function Attribution to Test Point (Koh 2017)' },
      ],
      'valuation_distribution',
      { label: 'valuation framework' },
    ),
    noisePercentage: slider(5, 30, 15, {
      step: 5,
      label: 'synthetic label noise rate in corpus (%)',
    }),
    duplicateCount: slider(1, 5, 2, {
      step: 1,
      label: 'identical duplicate exemplar copies k',
    }),
  })

  const { visualizerMode, noisePercentage, duplicateCount } = state

  const results = useMemo(() => {
    if (visualizerMode === 'valuation_distribution') {
      // 20 ranked percentiles from 1 to 100
      const ranks = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 93, 96, 98, 100]
      const noise = noisePercentage

      // Shapley values: top instances have positive value, middle are zero, bottom (noise) are negative
      const shapleyValues = ranks.map((r) => {
        if (r <= 20) return 0.08 * Math.exp(-r / 7)
        if (r <= 100 - noise) return 0.002 * (1 - (r - 20) / 70)
        // Negative values for corrupted label instances in bottom noise tail
        const depth = (r - (100 - noise)) / noise
        return -0.06 * Math.pow(depth, 1.4)
      })

      const zeroLine = ranks.map(() => 0)

      // Evaluation stats
      const negativePoints = shapleyValues.filter((v) => v < 0).length
      const potentialGain = (noise * 0.28).toFixed(1)

      return {
        x: ranks,
        y: shapleyValues.map((v) => v * 1000), // milli-units
        baseline: zeroLine,
        stat1: `Noise tail: ${noise}% of corpus`,
        stat2: `Toxic points detected: ${negativePoints} cohorts`,
        stat3: `Pruning toxic points: +${potentialGain}% test accuracy`,
        warning:
          noise > 20
            ? 'Heavy label contamination: negative Shapley tail drags down optimization'
            : 'Clear valuation separation: high-value anchors separate sharply from noise',
        axisXLabel: 'Dataset Centile Rank (1: Highest Value, 100: Lowest Value)',
        axisYLabel: 'Data Shapley Marginal Contribution phi_i (x 10^-3)',
        xRange: [0, 102] as [number, number],
        yRange: [-65, 85] as [number, number],
      }
    } else if (visualizerMode === 'loo_vs_shapley') {
      // Duplicate count from 1 to 5
      const counts = [1, 2, 3, 4, 5]
      const trueBaseValue = 10.0 // Value of the underlying unique information

      // LOO value: if k = 1, removing it loses 10.0. If k >= 2, removing 1 leaves k-1, so LOO drops to 0!
      const looValues = counts.map((k) => (k === 1 ? trueBaseValue : 0))
      // Shapley value: fairly splits the value among all k copies: 10 / k
      const shapleyValues = counts.map((k) => trueBaseValue / k)

      const currLOO = duplicateCount === 1 ? trueBaseValue : 0
      const currShapley = trueBaseValue / duplicateCount

      return {
        x: counts,
        y: shapleyValues,
        baseline: looValues,
        stat1: `Duplicates k = ${duplicateCount}`,
        stat2: `Shapley value: ${currShapley.toFixed(2)} (fair split)`,
        stat3: `LOO value: ${currLOO.toFixed(2)} (complete blind spot)`,
        warning:
          duplicateCount > 1
            ? 'CRITICAL LOO FLAW: Leave-One-Out assigns 0 value to critical data whenever duplicates exist'
            : 'Single copy: LOO and Shapley coincide',
        axisXLabel: 'Identical Duplicate Copies Present (k)',
        axisYLabel: 'Assigned Value per Copy',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [-1, 12] as [number, number],
      }
    } else {
      // 10 training samples evaluated against a test query
      // Samples 1 to 5: helpful (negative influence = decreases test loss)
      // Samples 6 to 10: harmful/confusing (positive influence = increases test loss)
      const samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const influenceValues = [-8.5, -6.2, -4.8, -3.1, -1.5, 1.2, 2.8, 4.4, 6.7, 9.2]
      const neutralLine = samples.map(() => 0)

      return {
        x: samples,
        y: influenceValues,
        baseline: neutralLine,
        stat1: 'Top helper: -8.5 influence',
        stat2: 'Top spoiler: +9.2 influence',
        stat3: 'Approximation: iHVP via LiSSA',
        warning: 'Harmful training samples often reveal subtle misclassifications or opposing class artifacts',
        axisXLabel: 'Ranked Training Exemplar (1-5: Helpful, 6-10: Harmful)',
        axisYLabel: 'Loss Influence I_loss(z_train, z_test)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [-10, 11] as [number, number],
      }
    }
  }, [visualizerMode, noisePercentage, duplicateCount])

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
      title="Data Valuation, Shapley Values & Influence Function Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating data valuation mechanics: Data Shapley distributions separating high-value anchors from label noise, the redundancy failure mode of Leave-One-Out (LOO), and test-point influence attribution."
      readouts={
        <>
          <Readout label="Valuation Mode" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Comparative Leverage" value={results.stat3} />
          <Readout
            label="Methodological Warning"
            value={results.warning}
            color={visualizerMode === 'loo_vs_shapley' && duplicateCount > 1 ? 'var(--red-600)' : 'var(--blue-600)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Valuation Trajectory">
        <Curve
          name={
            visualizerMode === 'valuation_distribution'
              ? 'Zero Value Threshold'
              : visualizerMode === 'loo_vs_shapley'
                ? 'Leave-One-Out (LOO) Assigned Value'
                : 'Neutral Influence Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'valuation_distribution'
              ? 'Data Shapley Value Curve'
              : visualizerMode === 'loo_vs_shapley'
                ? 'Data Shapley Fair Value'
                : 'Per-Sample Test Influence'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Evaluated Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
