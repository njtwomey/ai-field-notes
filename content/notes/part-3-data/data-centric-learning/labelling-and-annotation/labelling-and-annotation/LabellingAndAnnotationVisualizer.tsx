import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function LabellingAndAnnotationVisualizer() {
  const state = useFigureState({
    strategy: choice<'economics' | 'active_learning' | 'dawid_skene'>(
      [
        { value: 'active_learning', label: 'Active Learning (Uncertainty Sampling Efficiency)' },
        { value: 'economics', label: 'Label Economics (Quality vs Cost per Strategy)' },
        { value: 'dawid_skene', label: 'Dawid-Skene Aggregation vs Majority Vote' },
      ],
      'active_learning',
      { label: 'annotation evaluation lens' },
    ),
    labelBudget: slider(500, 5000, 2000, {
      step: 500,
      label: 'annotated sample count budget',
    }),
    annotatorNoise: slider(0.05, 0.45, 0.15, {
      step: 0.05,
      label: 'annotator error rate (1 - accuracy)',
    }),
  })

  const { strategy, labelBudget, annotatorNoise } = state

  const results = useMemo(() => {
    if (strategy === 'active_learning') {
      // Accuracy vs Number of Labeled Examples (N = 500 to 5000)
      const samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
      // Active learning saturates much faster
      const activeAcc = samples.map((n) => 95.0 - 45.0 * Math.exp(-n / 1000))
      // Random sampling scales slowly
      const randomAcc = samples.map((n) => 95.0 - 45.0 * Math.exp(-n / 2500))

      const currentActive = 95.0 - 45.0 * Math.exp(-labelBudget / 1000)
      const currentRandom = 95.0 - 45.0 * Math.exp(-labelBudget / 2500)

      return {
        x: samples,
        y: activeAcc,
        baseline: randomAcc,
        activeVal: currentActive,
        randomVal: currentRandom,
        gain: currentActive - currentRandom,
      }
    } else if (strategy === 'economics') {
      // 4 annotation strategies:
      // 1: Unassisted 1x Human (Cost 1.0x, Accuracy 82%)
      // 2: 3x Human Majority Vote (Cost 3.0x, Accuracy 91%)
      // 3: Model-Assisted Pre-label (Cost 1.4x, Accuracy 93%)
      // 4: Active Learning + Adjudication (Cost 1.8x, Accuracy 96%)
      const xIdx = [1, 2, 3, 4]
      const costScores = [1.0, 3.0, 1.4, 1.8]
      const qualityScores = [82.0, 91.0, 93.0, 96.0]

      return {
        x: xIdx,
        y: qualityScores,
        baseline: costScores.map((c) => c * 30), // scaled cost line
        activeVal: 96.0,
        randomVal: 82.0,
        gain: 14.0,
      }
    } else {
      // Dawid-Skene accuracy vs Majority Voting as crowd noise increases
      const noiseGrid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
      // Majority vote accuracy for 3 raters: P(at least 2 correct) = 3p^2(1-p) + p^3 where p = 1 - noise
      const majAcc = noiseGrid.map((err) => {
        const p = 1 - err
        return (3 * p * p * (1 - p) + p * p * p) * 100
      })
      // Dawid-Skene models individual confusions; outperforms majority vote by 3-8% at high noise
      const dsAcc = noiseGrid.map((err) => {
        const p = 1 - err
        const base = 3 * p * p * (1 - p) + p * p * p
        return Math.min(99.5, base + 0.08 * Math.sqrt(err)) * 100
      })

      const currentNoiseIdx = Math.round((annotatorNoise - 0.05) / 0.05)
      const curDS = dsAcc[currentNoiseIdx] || 90.0
      const curMaj = majAcc[currentNoiseIdx] || 85.0

      return {
        x: noiseGrid.map((n) => n * 100),
        y: dsAcc,
        baseline: majAcc,
        activeVal: curDS,
        randomVal: curMaj,
        gain: curDS - curMaj,
      }
    }
  }, [strategy, labelBudget, annotatorNoise])

  const axisX = useAxis({
    label:
      strategy === 'active_learning'
        ? 'Number of Human Labels Acquired'
        : strategy === 'economics'
          ? 'Workflow (1: 1x Human, 2: 3x Crowd Vote, 3: Model-Assisted, 4: Active + Adjudication)'
          : 'Individual Annotator Error Rate (%)',
    range: strategy === 'active_learning' ? [400, 5100] : strategy === 'economics' ? [0.5, 4.5] : [4, 46],
  })

  const axisY = useAxis({
    label: strategy === 'economics' ? 'Consensus Accuracy (%) / Cost Scale' : 'Model Validation Accuracy (%)',
    range: strategy === 'economics' ? [60, 102] : [50, 100],
  })

  return (
    <Figure
      title="Data Annotation Architecture & Efficiency Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating data labeling architectures: Active learning achieves higher model accuracy with far fewer labels than random sampling; Dawid-Skene probabilistic aggregation outperforms majority vote under noisy crowds."
      readouts={
        <>
          {strategy === 'active_learning' && (
            <>
              <Readout label="Active Learning Accuracy" value={`${results.activeVal.toFixed(1)}%`} />
              <Readout label="Random Baseline Accuracy" value={`${results.randomVal.toFixed(1)}%`} />
              <Readout
                label="Sample Efficiency Gain"
                value={`+${results.gain.toFixed(1)}% pts`}
                color="var(--blue-600)"
              />
            </>
          )}
          {strategy === 'economics' && (
            <>
              <Readout label="Top Strategy Quality" value="96.0% (Active + Adjudication)" />
              <Readout label="Cost Optimization" value="Model-assisted pre-labeling halves multi-annotator cost" />
            </>
          )}
          {strategy === 'dawid_skene' && (
            <>
              <Readout label="Dawid-Skene Consensus" value={`${results.activeVal.toFixed(1)}% accuracy`} />
              <Readout label="Simple Majority Vote" value={`${results.randomVal.toFixed(1)}% accuracy`} />
              <Readout label="Aggregation Advantage" value={`+${results.gain.toFixed(1)}% pts`} />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Annotation Strategy Profile">
        <Curve
          name={
            strategy === 'active_learning'
              ? 'Active Learning Curve'
              : strategy === 'dawid_skene'
                ? 'Dawid-Skene Aggregation'
                : 'Quality Metric'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            strategy === 'active_learning'
              ? 'Uniform Random Sampling'
              : strategy === 'dawid_skene'
                ? 'Majority Vote Baseline'
                : 'Cost Scale'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Points
          name="Operating Point"
          x={[strategy === 'active_learning' ? labelBudget : strategy === 'dawid_skene' ? annotatorNoise * 100 : 4]}
          y={[results.activeVal]}
          slot={1}
          size={8}
        />
      </Plot>
    </Figure>
  )
}
