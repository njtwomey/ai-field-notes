import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function FeedbackLoopsVisualizer() {
  const state = useFigureState({
    feedbackMode: choice<'exposure_share' | 'catalog_entropy'>(
      [
        { value: 'exposure_share', label: 'Category A Exposure Concentration (%) over Retraining Rounds' },
        { value: 'catalog_entropy', label: 'Catalog Diversity Entropy H(p) = -sum p*log(p)' },
      ],
      'exposure_share',
      { label: 'feedback dynamic' },
    ),
    explorationEpsilon: slider(0.0, 0.25, 0.05, {
      step: 0.01,
      label: 'randomized exploration injection rate (epsilon)',
    }),
  })

  const results = useMemo(() => {
    // 10 retraining rounds / epochs
    const rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const eps = state.explorationEpsilon

    // Initial slight exposure advantage: p_A = 0.55
    let currentShare = 0.55
    const shareCurve: number[] = []
    const entropyCurve: number[] = []

    rounds.forEach(() => {
      shareCurve.push(currentShare * 100.0)

      // Shannon entropy between Category A and B
      const pA = Math.max(0.001, Math.min(0.999, currentShare))
      const pB = 1.0 - pA
      const entropy = -(pA * Math.log2(pA) + pB * Math.log2(pB))
      entropyCurve.push(entropy)

      // Next round update:
      // Without exploration (eps=0), p_next = p^1.4 / (p^1.4 + (1-p)^1.4) (positive feedback runaway!)
      // With exploration, (1 - eps) * runaway + eps * 0.5
      const runawayP =
        Math.pow(currentShare, 1.45) / (Math.pow(currentShare, 1.45) + Math.pow(1.0 - currentShare, 1.45))
      currentShare = (1.0 - eps) * runawayP + eps * 0.5
    })

    const finalShare = shareCurve[shareCurve.length - 1]
    const finalEntropy = entropyCurve[entropyCurve.length - 1]
    const isDegenerate = finalShare > 85.0

    const baselineBalanced = rounds.map(() => 50.0)

    return {
      rounds,
      shareCurve,
      entropyCurve,
      baselineBalanced,
      finalShare,
      finalEntropy,
      isDegenerate,
      plotY: state.feedbackMode === 'exposure_share' ? shareCurve : entropyCurve,
    }
  }, [state.explorationEpsilon, state.feedbackMode])

  const axisX = useAxis({
    label: 'Model Retraining Iteration Round (1 to 10)',
    range: [0.8, 10.2],
  })

  const axisY = useAxis({
    label:
      state.feedbackMode === 'exposure_share'
        ? 'Favored Category Exposure Share (%)'
        : 'Shannon Diversity Entropy (Bits)',
    range: state.feedbackMode === 'exposure_share' ? [45, 100] : [0, 1.05],
  })

  return (
    <Figure
      title="Runaway Feedback Loops and Performative Distribution Shift"
      state={state}
      hoverReadout={false}
      caption="When an AI system trains on historical interaction data influenced by its own past predictions, small initial random advantages compound exponentially. Without deliberate randomized exploration (epsilon), the system collapses into an uninformative monoculture."
      readouts={
        <>
          <Readout label="Initial Exposure" value="55.0% Category A" />
          <Readout
            label="Round 10 Exposure"
            value={`${results.finalShare.toFixed(1)}% (${results.isDegenerate ? 'Runaway Monoculture!' : 'Stabilized Pluralism'})`}
          />
          <Readout label="Catalog Diversity" value={`${results.finalEntropy.toFixed(2)} bits (Max = 1.00)`} />
          <Readout
            label="Exploration Status"
            value={
              state.explorationEpsilon > 0.08
                ? 'Active exploration prevents runaway'
                : 'Vulnerable to feedback collapse'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Compounding Feedback Dynamics Across Deployment Cycles">
        <Curve name="Model Exposure Trajectory" x={results.rounds} y={results.plotY} slot={0} width={2.5} />
        {state.feedbackMode === 'exposure_share' && (
          <Curve
            name="Balanced Organic Ground Truth (50%)"
            x={results.rounds}
            y={results.baselineBalanced}
            slot={2}
            dashed
            width={2}
          />
        )}
        <Points name="Retraining Epochs" x={results.rounds} y={results.plotY} slot={1} size={7} />
      </Plot>
    </Figure>
  )
}
