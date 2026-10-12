import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SyntheticDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'model_collapse' | 'tstr_utility' | 'privacy_tradeoff'>(
      [
        { value: 'model_collapse', label: 'Recursive Model Collapse & Tail Erasure (Shumailov 2024)' },
        { value: 'tstr_utility', label: 'Train on Synthetic, Test on Real (TSTR) Frontier' },
        { value: 'privacy_tradeoff', label: 'Differential Privacy Epsilon vs Synthetic Utility (Dwork 2014)' },
      ],
      'model_collapse',
      { label: 'synthesis perspective' },
    ),
    recursiveGen: slider(0, 8, 2, {
      step: 1,
      label: 'recursive synthetic training generation g (0: Real Data)',
    }),
    dpEpsilon: slider(0.5, 10.0, 3.0, {
      step: 0.5,
      label: 'differential privacy epsilon budget',
    }),
  })

  const { visualizerMode, recursiveGen, dpEpsilon } = state

  const results = useMemo(() => {
    if (visualizerMode === 'model_collapse') {
      // 17 domain support bins from -4.0 to +4.0
      const xs = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
      const g = recursiveGen

      // Original Real distribution (g=0): standard normal mu=0, sigma=1.0
      const baseSigma = 1.0
      const origDensities = xs.map(
        (x) => (1 / (baseSigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow(x / baseSigma, 2)) * 100,
      )

      // Contracted distribution at generation g: sigma_g = 1.0 * (0.82)^g
      const currSigma = Math.max(0.18, 1.0 * Math.pow(0.82, g))
      // Mean drift delta = 0.08 * g
      const currMu = 0.08 * g
      const collapsedDensities = xs.map((x) => {
        const exponent = -0.5 * Math.pow((x - currMu) / currSigma, 2)
        return (1 / (currSigma * Math.sqrt(2 * Math.PI))) * Math.exp(exponent) * 100
      })

      // Tail probability (|x| > 2.0)
      const tailCover = Math.max(0.5, 100 * Math.pow(0.62, g))
      const varianceRatio = (Math.pow(currSigma, 2) * 100).toFixed(1)

      return {
        x: xs,
        y: collapsedDensities,
        baseline: origDensities,
        stat1: `Generation g = ${g} ${g === 0 ? '(Original Real Ground Truth)' : '(Recursive Synthetic)'}`,
        stat2: `Residual variance: ${varianceRatio}% of original`,
        stat3: `Tail event coverage: ${tailCover.toFixed(1)}%`,
        warning:
          g >= 5
            ? 'CATASTROPHIC MODEL COLLAPSE: Distribution has collapsed into a degenerate mode'
            : g >= 2
              ? 'Early model collapse: long-tail rare concepts and minority expressions erased'
              : 'Uncontaminated distribution: full support coverage maintained',
        axisXLabel: 'Latent Semantic Data Manifold Coordinate x',
        axisYLabel: 'Probability Density Distribution p_g(x) [%]',
        xRange: [-4.2, 4.2] as [number, number],
        yRange: [0, 230] as [number, number],
      }
    } else if (visualizerMode === 'tstr_utility') {
      // Synthetic data mix ratio from 0% (pure real) to 100% (pure synthetic)
      const synMix = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      // High-quality calibrated generator TSTR accuracy: drops gently from 88% down to 81%
      const highQAcc = synMix.map((m) => 88.0 - 7.0 * Math.pow(m / 100, 1.3))
      // Uncalibrated / naive generator TSTR accuracy: plummets from 88% down to 52%
      const lowQAcc = synMix.map((m) => 88.0 - 36.0 * Math.pow(m / 100, 0.9))

      return {
        x: synMix,
        y: highQAcc,
        baseline: lowQAcc,
        stat1: '100% Real (TRTR): 88.0% accuracy',
        stat2: '100% Synthetic (TSTR): 81.0% accuracy',
        stat3: 'Fidelity retention: 92.0% of real',
        warning: 'High-quality conditional synthesis retains >90% of downstream predictive utility',
        axisXLabel: 'Synthetic Data Share in Training Set (%)',
        axisYLabel: 'Real Test Set Evaluation Accuracy (%)',
        xRange: [0, 100] as [number, number],
        yRange: [45, 95] as [number, number],
      }
    } else {
      // DP epsilon budget from 0.5 to 10.0
      const epsilons = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
      // Synthetic data utility (% of unconstrained performance)
      const utility = epsilons.map((e) => 92.0 * (1 - Math.exp(-e / 1.8)))
      // Privacy leakage / Membership inference risk (%)
      const privacyRisk = epsilons.map((e) => 5.0 + 72.0 * (1 - Math.exp(-e / 3.5)))

      const currUtil = 92.0 * (1 - Math.exp(-dpEpsilon / 1.8))
      const currRisk = 5.0 + 72.0 * (1 - Math.exp(-dpEpsilon / 3.5))

      return {
        x: epsilons,
        y: utility,
        baseline: privacyRisk,
        stat1: `Privacy budget epsilon = ${dpEpsilon.toFixed(1)}`,
        stat2: `Downstream utility: ${currUtil.toFixed(1)}%`,
        stat3: `Empirical re-id risk: ${currRisk.toFixed(1)}%`,
        warning:
          dpEpsilon < 1.5
            ? 'High privacy protection, but heavy Gaussian noise injection degrades statistical utility'
            : dpEpsilon > 6.0
              ? 'Weak privacy protection: memorization permits training record reconstruction'
              : 'Balanced DP regime: provable privacy guarantees with robust downstream utility',
        axisXLabel: 'Differential Privacy Parameter Epsilon',
        axisYLabel: 'Utility & Re-Identification Risk (%)',
        xRange: [0, 10.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, recursiveGen, dpEpsilon])

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
      title="Synthetic Data Dynamics, Model Collapse & Privacy Inspector"
      state={state}
      hoverReadout={false}
      caption="Evaluating synthetic data generation mechanics: Autophagous recursive model collapse and tail event erasure across generations, Train on Synthetic Test on Real (TSTR) utility frontiers, and Differential Privacy trade-offs."
      readouts={
        <>
          <Readout label="Generation / Setting" value={results.stat1} />
          <Readout label="Preservation State" value={results.stat2} />
          <Readout label="Integrity Metric" value={results.stat3} />
          <Readout
            label="Methodological Directive"
            value={results.warning}
            color={
              (visualizerMode === 'model_collapse' && recursiveGen >= 5) ||
              (visualizerMode === 'privacy_tradeoff' && dpEpsilon > 6.0)
                ? 'var(--red-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Synthesis Characteristic Curve">
        <Curve
          name={
            visualizerMode === 'model_collapse'
              ? 'Original Real Ground Truth (Gen 0)'
              : visualizerMode === 'tstr_utility'
                ? 'Uncalibrated Generator Baseline'
                : 'Membership Inference Re-ID Risk'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'model_collapse'
              ? 'Recursive Generation Density p_g(x)'
              : visualizerMode === 'tstr_utility'
                ? 'Calibrated Generator TSTR Accuracy'
                : 'Statistical Downstream Utility'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating States" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
