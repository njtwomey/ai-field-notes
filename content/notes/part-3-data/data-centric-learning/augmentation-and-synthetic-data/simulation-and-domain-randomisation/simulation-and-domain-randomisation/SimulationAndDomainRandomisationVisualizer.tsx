import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SimulationAndDomainRandomisationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'randomization_breadth' | 'adversarial_alignment' | 'ablation_synergy'>(
      [
        { value: 'randomization_breadth', label: 'Domain Randomisation Breadth vs Real-World Transfer (Tobin 2017)' },
        { value: 'adversarial_alignment', label: 'Domain Adversarial Adaptation Feature Alignment (Ganin 2016)' },
        { value: 'ablation_synergy', label: 'Visual vs Dynamics Randomisation Factor Synergy' },
      ],
      'randomization_breadth',
      { label: 'sim-to-real regime' },
    ),
    randomizationRange: slider(1, 10, 6, {
      step: 1,
      label: 'simulation parameter randomization breadth sigma',
    }),
    adversarialLambda: slider(0.0, 1.0, 0.5, {
      step: 0.1,
      label: 'gradient reversal adaptation weight lambda',
    }),
  })

  const { visualizerMode, randomizationRange, adversarialLambda } = state

  const results = useMemo(() => {
    if (visualizerMode === 'randomization_breadth') {
      // Randomization range sigma from 0 to 10
      const sigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Synthetic simulation training accuracy: decays slowly as variance expands task difficulty
      const simAcc = sigmas.map((s) => Math.max(70, 99 - s * 2.2))
      // Real-world zero-shot transfer accuracy: starts at ~12% (photorealism gap), peaks at sigma=6-7, drops if over-randomized
      const realAcc = sigmas.map((s) => {
        if (s === 0) return 12.0
        if (s <= 6) return 12.0 + 76.0 * (1 - Math.exp(-s / 2.2))
        // Peak and slight decay past 7
        return 88.0 - Math.pow(s - 6.5, 2) * 1.8
      })

      const currSim = Math.max(70, 99 - randomizationRange * 2.2)
      const currReal =
        randomizationRange <= 6
          ? 12.0 + 76.0 * (1 - Math.exp(-randomizationRange / 2.2))
          : 88.0 - Math.pow(randomizationRange - 6.5, 2) * 1.8

      return {
        x: sigmas,
        y: realAcc,
        baseline: simAcc,
        stat1: `Randomization breadth: ${randomizationRange} / 10`,
        stat2: `Real-world zero-shot accuracy: ${currReal.toFixed(1)}%`,
        stat3: `Sim-to-real gap: ${(currSim - currReal).toFixed(1)}%`,
        warning:
          randomizationRange <= 2
            ? 'Reality Gap: clean simulator models fail catastrophically on physical hardware'
            : randomizationRange >= 9
              ? 'Over-randomisation: extreme unphysical variance degrades policy capability'
              : 'Optimal Domain Randomisation: real world appears as one more simulation variation',
        axisXLabel: 'Randomisation Parameter Breadth sigma (Textures, Lighting, Dynamics)',
        axisYLabel: 'Task Performance Accuracy (%)',
        xRange: [-0.5, 10.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'adversarial_alignment') {
      // Training steps 0 to 100
      const steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      const l = adversarialLambda

      // Domain discrepancy without adaptation (remains high ~78%)
      const unadaptedDiscrepancy = steps.map(() => 78.0)
      // Domain discrepancy with gradient reversal: drops smoothly
      const adaptedDiscrepancy = steps.map((s) => 78.0 * Math.exp(-s * (0.015 + 0.035 * l)))

      const finalDiscrepancy = (78.0 * Math.exp(-100 * (0.015 + 0.035 * l))).toFixed(1)

      return {
        x: steps,
        y: adaptedDiscrepancy,
        baseline: unadaptedDiscrepancy,
        stat1: `Adaptation weight lambda = ${l.toFixed(1)}`,
        stat2: `Residual domain gap: ${finalDiscrepancy}%`,
        stat3: 'Gradient Reversal Layer (GRL)',
        warning: 'Adversarial feature alignment forces latent space to be invariant to sim vs real origin',
        axisXLabel: 'Domain Adaptation Steps (% of Training)',
        axisYLabel: 'Empirical Domain Discrepancy (d_H) [%]',
        xRange: [0, 100] as [number, number],
        yRange: [0, 90] as [number, number],
      }
    } else {
      // 4 ablation regimes:
      // 1: Photorealistic Sim Only (No DR)
      // 2: Visual Randomisation Only (Textures & Lights)
      // 3: Dynamics Randomisation Only (Mass, Friction, Latency)
      // 4: Joint Visual + Dynamics Randomisation
      const regimes = [1, 2, 3, 4]
      const successRates = [14.0, 48.0, 38.0, 89.5]
      const simUpperLimit = regimes.map(() => 98.0)

      return {
        x: regimes,
        y: successRates,
        baseline: simUpperLimit,
        stat1: 'Visual only: 48.0% real success',
        stat2: 'Dynamics only: 38.0% real success',
        stat3: 'Joint Visual+Dynamics: 89.5% success',
        warning: 'Physical robotic deployment requires simultaneous visual and physical parameter randomisation',
        axisXLabel: 'Randomisation Setup (1: None, 2: Visual, 3: Dynamics, 4: Joint)',
        axisYLabel: 'Real-World Task Success Rate (%)',
        xRange: [0.5, 4.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, randomizationRange, adversarialLambda])

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
      title="Simulation-to-Real Transfer & Domain Randomisation Inspector"
      state={state}
      hoverReadout={false}
      caption="Quantifying sim-to-real transfer: Domain randomisation variance vs real-world zero-shot accuracy, domain adversarial alignment via gradient reversal, and multi-factor ablation synergy."
      readouts={
        <>
          <Readout label="Transfer State" value={results.stat1} />
          <Readout label="Target Accuracy" value={results.stat2} />
          <Readout label="Gap Metric" value={results.stat3} />
          <Readout
            label="Engineering Directive"
            value={results.warning}
            color={
              visualizerMode === 'randomization_breadth' && randomizationRange <= 2
                ? 'var(--red-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Sim-to-Real Trajectory">
        <Curve
          name={
            visualizerMode === 'randomization_breadth'
              ? 'Simulated Training Accuracy'
              : visualizerMode === 'adversarial_alignment'
                ? 'Unadapted Domain Discrepancy Baseline'
                : 'Ideal Simulation Performance Ceiling'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'randomization_breadth'
              ? 'Real-World Zero-Shot Transfer Accuracy'
              : visualizerMode === 'adversarial_alignment'
                ? 'Adapted Domain Discrepancy'
                : 'Empirical Real-World Success Rate'
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
