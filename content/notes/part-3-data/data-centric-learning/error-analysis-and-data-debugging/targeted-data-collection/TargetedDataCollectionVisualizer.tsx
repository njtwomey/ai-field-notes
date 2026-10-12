import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TargetedDataCollectionVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'collection_efficiency' | 'shortcut_attenuation' | 'flywheel_yield'>(
      [
        { value: 'collection_efficiency', label: 'Targeted vs Uniform Collection Yield (Rare Slice)' },
        { value: 'shortcut_attenuation', label: 'Counterfactual Injection & Shortcut Mitigation (Kaushik 2020)' },
        { value: 'flywheel_yield', label: 'Active Deployment Trigger Utility & Signal Density' },
      ],
      'collection_efficiency',
      { label: 'collection perspective' },
    ),
    slicePrevalence: slider(0.5, 10.0, 2.0, {
      step: 0.5,
      label: 'rare slice natural prevalence pi_k (%)',
    }),
    counterfactualRatio: slider(0, 50, 20, {
      step: 5,
      label: 'counterfactual sample share in slice (%)',
    }),
  })

  const { visualizerMode, slicePrevalence, counterfactualRatio } = state

  const results = useMemo(() => {
    if (visualizerMode === 'collection_efficiency') {
      // Desired slice sample size: 50 to 500
      const desiredSizes = [50, 100, 150, 200, 250, 300, 400, 500]
      const pi = slicePrevalence / 100

      // Uniform random collection required: N_slice / pi
      const uniformRequired = desiredSizes.map((n) => Math.round(n / pi))
      // Targeted collection required: N_slice * (1 + overhead ~1.15)
      const targetedRequired = desiredSizes.map((n) => Math.round(n * 1.15))

      const currentDesired = 200
      const currUniform = Math.round(currentDesired / pi)
      const currTargeted = Math.round(currentDesired * 1.15)
      const savings = (((currUniform - currTargeted) / currUniform) * 100).toFixed(0)

      return {
        x: desiredSizes,
        y: uniformRequired,
        baseline: targetedRequired,
        stat1: `Prevalence pi_k: ${slicePrevalence.toFixed(1)}%`,
        stat2: `Targeted yield: ${currTargeted} labelled`,
        stat3: `Uniform cost: ${currUniform} labelled (${savings}% savings)`,
        warning:
          slicePrevalence < 2.0
            ? 'Extreme tail slice: uniform random collection is economically non-viable'
            : 'Targeted query harvesting concentrates annotation budget on failing cohorts',
        axisXLabel: 'Target Slice Volume Required (Examples)',
        axisYLabel: 'Total Ingestion & Labelling Volume (Examples)',
        xRange: [40, 520] as [number, number],
        yRange: [0, Math.max(...uniformRequired) * 1.05] as [number, number],
      }
    } else if (visualizerMode === 'shortcut_attenuation') {
      // Counterfactual ratio: 0% to 50%
      const ratios = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
      // Reliance on spurious shortcut: drops from 95% down to 8%
      const shortcutReliance = ratios.map((r) => 95 * Math.exp(-r / 14))
      // Out-of-distribution (OOD) accuracy: rises from 40% to 88%
      const oodAccuracy = ratios.map((r) => 40 + 48 * (1 - Math.exp(-r / 12)))

      const currReliance = 95 * Math.exp(-counterfactualRatio / 14)
      const currOOD = 40 + 48 * (1 - Math.exp(-counterfactualRatio / 12))

      return {
        x: ratios,
        y: oodAccuracy,
        baseline: shortcutReliance,
        stat1: `Counterfactual injection: ${counterfactualRatio}%`,
        stat2: `OOD robust accuracy: ${currOOD.toFixed(1)}%`,
        stat3: `Spurious shortcut reliance: ${currReliance.toFixed(1)}%`,
        warning:
          counterfactualRatio < 10
            ? 'Model exploits non-causal background shortcuts; high vulnerability under shift'
            : 'Sufficient counterexamples: decision boundary aligns with invariant causal features',
        axisXLabel: 'Counterfactual / Anti-Shortcut Share in Training Set (%)',
        axisYLabel: 'Performance Metric Percentage (%)',
        xRange: [0, 52] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Production triggers: 1 to 4
      // 1: Random Stream Sampling
      // 2: Softmax Margin Uncertainty (|p1 - p2| < 0.1)
      // 3: Latent Embedding OOD Anomaly
      // 4: Direct User Friction (Undo / Re-query / Negative Feedback)
      const triggers = [1, 2, 3, 4]
      // Information density (% of triggered items that resolve error modes)
      const informativeYield = [3.5, 38.0, 56.0, 78.0]
      const randomBaseline = triggers.map(() => 3.5)

      return {
        x: triggers,
        y: informativeYield,
        baseline: randomBaseline,
        stat1: 'User friction yield: 78.0% error density',
        stat2: 'Uncertainty trigger yield: 38.0%',
        stat3: 'Random baseline: 3.5% informative',
        warning: 'Deploy user friction and latent anomaly triggers to maximize data flywheel ROI',
        axisXLabel: 'Trigger Mechanism (1: Random, 2: Uncertainty, 3: OOD Anomaly, 4: User Friction)',
        axisYLabel: 'High-Value Diagnostic Yield (%)',
        xRange: [0.5, 4.5] as [number, number],
        yRange: [0, 90] as [number, number],
      }
    }
  }, [visualizerMode, slicePrevalence, counterfactualRatio])

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
      title="Targeted Data Collection & Flywheel Diagnostics"
      state={state}
      hoverReadout={false}
      caption="Evaluating targeted acquisition mechanisms: Sample efficiency across rare slice prevalences, counterfactual data injection to break spurious shortcuts, and production trigger yields."
      readouts={
        <>
          <Readout label="Acquisition Setting" value={results.stat1} />
          <Readout label="Targeted Output" value={results.stat2} />
          <Readout label="Comparative Leverage" value={results.stat3} />
          <Readout
            label="Flywheel Directive"
            value={results.warning}
            color={
              (visualizerMode === 'collection_efficiency' && slicePrevalence < 2.0) ||
              (visualizerMode === 'shortcut_attenuation' && counterfactualRatio < 10)
                ? 'var(--red-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Targeted Acquisition Trajectory">
        <Curve
          name={
            visualizerMode === 'collection_efficiency'
              ? 'Targeted Harvesting Requirement'
              : visualizerMode === 'shortcut_attenuation'
                ? 'Spurious Shortcut Reliance'
                : 'Uniform Random Stream Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'collection_efficiency'
              ? 'Uniform Random Sampling Requirement'
              : visualizerMode === 'shortcut_attenuation'
                ? 'OOD Generalization Accuracy'
                : 'High-Value Training Signal Density'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Metrics" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
