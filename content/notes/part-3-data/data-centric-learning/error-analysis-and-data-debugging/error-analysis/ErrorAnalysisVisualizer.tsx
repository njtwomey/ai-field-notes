import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ErrorAnalysisVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'pareto_taxonomy' | 'sampling_efficiency' | 'fix_roi'>(
      [
        { value: 'pareto_taxonomy', label: 'Error Taxonomy & Pareto Distribution (Ng 2018)' },
        { value: 'sampling_efficiency', label: 'Sampling Strategy vs Tail Error Discovery' },
        { value: 'fix_roi', label: 'Intervention ROI: Data Curation vs Model Tweaks' },
      ],
      'pareto_taxonomy',
      { label: 'analysis perspective' },
    ),
    auditSampleSize: slider(25, 300, 100, {
      step: 25,
      label: 'audit sample size N_errors',
    }),
    confidenceLevel: choice<'high' | 'marginal' | 'all'>(
      [
        { value: 'all', label: 'All Misclassifications' },
        { value: 'high', label: 'High-Confidence Blunders (p > 0.90)' },
        { value: 'marginal', label: 'Boundary Indecision (0.45 < p < 0.55)' },
      ],
      'all',
      { label: 'prediction slice filter' },
    ),
  })

  const { visualizerMode, auditSampleSize, confidenceLevel } = state

  const results = useMemo(() => {
    if (visualizerMode === 'pareto_taxonomy') {
      // 6 failure categories
      const categories = [1, 2, 3, 4, 5, 6]
      let shares: number[]
      if (confidenceLevel === 'high') {
        // High confidence errors are dominated by label errors & spurious shortcuts
        shares = [44, 28, 14, 8, 4, 2]
      } else if (confidenceLevel === 'marginal') {
        // Marginal errors are dominated by boundary ambiguity and capacity
        shares = [8, 12, 19, 15, 31, 15]
      } else {
        // All errors
        shares = [29, 22, 18, 14, 11, 6]
      }

      // Pareto cumulative sum
      let running = 0
      const cum = shares.map((s) => {
        running += s
        return running
      })

      const topShare = shares[0]
      const top3Share = cum[2]
      // Standard error of proportion at chosen sample size: sqrt(p * (1-p) / N)
      const marginErr = 1.96 * Math.sqrt(((topShare / 100) * (1 - topShare / 100)) / auditSampleSize) * 100

      return {
        x: categories,
        y: shares,
        baseline: cum,
        stat1: `Leading failure mode: ${topShare}%`,
        stat2: `Top-3 causes cover ${top3Share}% of errors`,
        stat3: `95% CI: ±${marginErr.toFixed(1)}% (N=${auditSampleSize})`,
        warning:
          topShare >= 40
            ? 'Concentrated failure bottleneck: targeted data fix eliminates over 40% of errors'
            : 'Distributed failure profile: multi-pronged data and pipeline interventions required',
        axisXLabel:
          'Taxonomy Category (1: Label Error, 2: Spurious Shortcut, 3: Rare Slice, 4: Sensor/Noise, 5: Ambiguous, 6: Model Capacity)',
        axisYLabel: 'Proportion of Total Errors (%)',
        xRange: [0.5, 6.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'sampling_efficiency') {
      // Sample size from 25 to 300
      const sizes = [25, 50, 75, 100, 150, 200, 250, 300]
      // Tail failure modes discovered:
      // Stratified slice sampling discovers tail modes rapidly
      const stratifiedDiscovery = sizes.map((n) => Math.min(100, 100 * (1 - Math.exp(-n / 45))))
      // Uniform random sampling discovers tail modes much slower
      const randomDiscovery = sizes.map((n) => Math.min(100, 100 * (1 - Math.exp(-n / 130))))

      const currentStrat = Math.min(100, 100 * (1 - Math.exp(-auditSampleSize / 45)))
      const currentRand = Math.min(100, 100 * (1 - Math.exp(-auditSampleSize / 130)))

      return {
        x: sizes,
        y: stratifiedDiscovery,
        baseline: randomDiscovery,
        stat1: `Stratified yield: ${currentStrat.toFixed(0)}% tail modes`,
        stat2: `Random yield: ${currentRand.toFixed(0)}% tail modes`,
        stat3: `Discovery acceleration: ${(currentStrat / Math.max(1, currentRand)).toFixed(1)}x`,
        warning: 'Uniform random sampling overlooks critical tail slices; use stratified error sampling',
        axisXLabel: 'Audit Sample Size (Inspected Mistakes)',
        axisYLabel: 'Rare Failure Mode Discovery Coverage (%)',
        xRange: [20, 310] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Interventions: 1 to 5
      // 1: Relabel Mislabeled Ground Truth (Data)
      // 2: Augment Underrepresented Slices (Data)
      // 3: Feature Engineering / Regularization (Pipeline)
      // 4: Deeper Architecture / Scaling (Model)
      // 5: Hyperparameter Grid Search (Tuning)
      const interventions = [1, 2, 3, 4, 5]
      // Expected Error Rate Reduction (%)
      const reduction = [28, 22, 14, 9, 3]
      // Engineering Effort in Days
      const effortDays = [3, 5, 8, 14, 7]
      // Efficiency ratio (Reduction % per Eng Day)
      const efficiency = reduction.map((r, i) => (r / effortDays[i]) * 10)

      return {
        x: interventions,
        y: reduction,
        baseline: efficiency,
        stat1: 'Data cleaning ROI: 9.3% error drop/day',
        stat2: 'Model tuning ROI: 0.4% error drop/day',
        stat3: 'Data interventions offer ~20x leverage',
        warning: 'Engineers default to architecture tweaks; data fixes produce drastically higher ROI',
        axisXLabel: 'Intervention (1: Clean Labels, 2: Collect Slice, 3: Features, 4: Arch Scale, 5: Grid Search)',
        axisYLabel: 'Error Reduction Impact vs ROI Factor',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [0, 35] as [number, number],
      }
    }
  }, [visualizerMode, auditSampleSize, confidenceLevel])

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
      title="Systematic Error Analysis & Failure Attribution Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating error analysis methodology: Pareto category attribution across confidence slices, sampling scheme efficiency for rare failure modes, and engineering ROI of data debugging versus model re-architecting."
      readouts={
        <>
          <Readout label="Attribution Metric" value={results.stat1} />
          <Readout label="Coverage / Yield" value={results.stat2} />
          <Readout label="Statistical Bounds" value={results.stat3} />
          <Readout
            label="Engineering Direction"
            value={results.warning}
            color={visualizerMode === 'fix_roi' ? 'var(--blue-600)' : 'var(--emerald-600)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Failure Analysis Trajectory">
        <Curve
          name={
            visualizerMode === 'pareto_taxonomy'
              ? 'Cumulative Pareto Error Share'
              : visualizerMode === 'sampling_efficiency'
                ? 'Uniform Random Sampling Baseline'
                : 'ROI Efficiency (Impact / Eng-Day)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'pareto_taxonomy'
              ? 'Discrete Category Share'
              : visualizerMode === 'sampling_efficiency'
                ? 'Stratified Slice Sampling Coverage'
                : 'Error Reduction Potential (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Observation Bins" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
