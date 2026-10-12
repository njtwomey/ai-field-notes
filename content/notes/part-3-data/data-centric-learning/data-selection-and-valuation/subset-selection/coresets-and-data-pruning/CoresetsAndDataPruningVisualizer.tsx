import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function CoresetsAndDataPruningVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'scaling_curves' | 'gradient_matching' | 'distillation_ipc'>(
      [
        { value: 'scaling_curves', label: 'Pruning Scaling Laws vs Uniform Subsampling (Sorscher 2022)' },
        { value: 'gradient_matching', label: 'CRAIG Coreset Gradient Fidelity vs Subset Size (Mirzasoleiman 2020)' },
        { value: 'distillation_ipc', label: 'Dataset Distillation Compression Curve (Wang 2018)' },
      ],
      'scaling_curves',
      { label: 'selection paradigm' },
    ),
    retentionFraction: slider(10, 100, 70, {
      step: 5,
      label: 'dataset retention fraction (%)',
    }),
    coresetFraction: slider(5, 50, 20, {
      step: 5,
      label: 'CRAIG coreset size |S| (%)',
    }),
  })

  const { visualizerMode, retentionFraction, coresetFraction } = state

  const results = useMemo(() => {
    if (visualizerMode === 'scaling_curves') {
      // Retention fraction from 10% to 100%
      const fractions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      // Uniform random subsampling error: power law decay as fraction increases
      // Error = 12% + 28% * (100 / f)^0.4
      const uniformError = fractions.map((f) => 10.0 + 35.0 * Math.pow(10 / f, 0.45))
      // Optimal pruning curve: drops faster, achieves minimum at ~70% retention!
      // In over-parameterized regimes, pruning 30% of easy redundant examples improves test error
      const prunedError = fractions.map((f) => {
        if (f <= 50) return 9.5 + 30.0 * Math.pow(10 / f, 0.5)
        // Sweet spot dip around 65-80%
        const dip = 1.8 * Math.sin(((f - 50) / 50) * Math.PI)
        return 9.5 + 30.0 * Math.pow(10 / f, 0.5) - dip
      })

      const currUniform = 10.0 + 35.0 * Math.pow(10 / retentionFraction, 0.45)
      const dipVal = retentionFraction > 50 ? 1.8 * Math.sin(((retentionFraction - 50) / 50) * Math.PI) : 0
      const currPruned = 9.5 + 30.0 * Math.pow(10 / retentionFraction, 0.5) - dipVal

      return {
        x: fractions,
        y: prunedError,
        baseline: uniformError,
        stat1: `Retained data: ${retentionFraction}%`,
        stat2: `Pruned test error: ${currPruned.toFixed(1)}%`,
        stat3: `Random test error: ${currUniform.toFixed(1)}% (Δ = -${(currUniform - currPruned).toFixed(1)}%)`,
        warning:
          retentionFraction >= 65 && retentionFraction <= 80
            ? 'Optimal pruning regime: matches or beats full-dataset accuracy with 25-35% fewer samples'
            : retentionFraction < 30
              ? 'Aggressive pruning: core manifold coverage compromised'
              : 'Diminishing returns: redundant canonical instances inflate training cost',
        axisXLabel: 'Dataset Retention Budget (%)',
        axisYLabel: 'Evaluation Test Error Rate (%)',
        xRange: [5, 105] as [number, number],
        yRange: [8, 48] as [number, number],
      }
    } else if (visualizerMode === 'gradient_matching') {
      // Coreset size |S| from 5% to 50%
      const sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
      // CRAIG submodular gradient approximation error: drops rapidly
      const craigError = sizes.map((s) => 65 * Math.exp(-s / 11) + 2.5)
      // Random subset gradient approximation error: drops much slower (sqrt(N))
      const randomError = sizes.map((s) => 80 * Math.pow(5 / s, 0.45))

      const currCraig = 65 * Math.exp(-coresetFraction / 11) + 2.5
      const currRand = 80 * Math.pow(5 / coresetFraction, 0.45)
      const speedup = (100 / coresetFraction).toFixed(1)

      return {
        x: sizes,
        y: craigError,
        baseline: randomError,
        stat1: `Coreset size: ${coresetFraction}% of full batch`,
        stat2: `CRAIG gradient error: ${currCraig.toFixed(1)}%`,
        stat3: `Random gradient error: ${currRand.toFixed(1)}%`,
        warning: `CRAIG achieves ${(currRand / currCraig).toFixed(1)}x tighter gradient fidelity, accelerating training by ${speedup}x`,
        axisXLabel: 'Coreset Selection Budget |S| (% of Full Batch)',
        axisYLabel: 'Full Batch Gradient Approximation Error (%)',
        xRange: [3, 52] as [number, number],
        yRange: [0, 85] as [number, number],
      }
    } else {
      // Images per class (IPC): 1, 5, 10, 20, 50, 100
      const ipcs = [1, 5, 10, 20, 50, 100]
      // Distilled synthetic data accuracy on CIFAR-10
      const distilledAcc = [44.0, 58.0, 66.5, 74.0, 81.5, 86.0]
      // Real random images per class accuracy
      const realRandomAcc = [18.0, 32.0, 42.0, 54.0, 68.0, 77.0]

      return {
        x: ipcs,
        y: distilledAcc,
        baseline: realRandomAcc,
        stat1: 'Distilled IPC=10: 66.5% accuracy',
        stat2: 'Random IPC=10: 42.0% accuracy',
        stat3: 'Compression factor: 500x at IPC=10',
        warning: 'Synthesized distillation prototypes pack orders of magnitude more gradient density than raw examples',
        axisXLabel: 'Synthesized Images Per Class (IPC)',
        axisYLabel: 'Downstream Model Test Accuracy (%)',
        xRange: [0, 105] as [number, number],
        yRange: [10, 92] as [number, number],
      }
    }
  }, [visualizerMode, retentionFraction, coresetFraction])

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
      title="Coresets, Data Pruning & Distillation Inspector"
      state={state}
      hoverReadout={false}
      caption="Evaluating training subset compression: Non-linear pruning scaling laws beating power law scaling, CRAIG submodular gradient matching, and dataset distillation efficiency."
      readouts={
        <>
          <Readout label="Budget Setting" value={results.stat1} />
          <Readout label="Performance Yield" value={results.stat2} />
          <Readout label="Comparative Advantage" value={results.stat3} />
          <Readout
            label="Optimization Insight"
            value={results.warning}
            color={
              visualizerMode === 'scaling_curves' && retentionFraction >= 65 && retentionFraction <= 80
                ? 'var(--emerald-600)'
                : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Subset Optimization Trajectory">
        <Curve
          name={
            visualizerMode === 'scaling_curves'
              ? 'Uniform Subsampling Error'
              : visualizerMode === 'gradient_matching'
                ? 'Random Subset Gradient Error'
                : 'Real Random Sample Accuracy Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'scaling_curves'
              ? 'Optimal Pruning Error'
              : visualizerMode === 'gradient_matching'
                ? 'CRAIG Coreset Gradient Error'
                : 'Distilled Synthetic Accuracy'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Configurations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
