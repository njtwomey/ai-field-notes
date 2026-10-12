import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function CurriculaAndDataMixturesVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'pacing_curriculum' | 'domain_mixture' | 'repetition_epochs'>(
      [
        { value: 'pacing_curriculum', label: 'Curriculum Pacing Functions (Bengio 2009)' },
        { value: 'domain_mixture', label: 'Heterogeneous Domain Mixture & Temperature Scaling' },
        { value: 'repetition_epochs', label: 'Multi-Epoch Repetition Scaling & Decay (Muennighoff 2023)' },
      ],
      'pacing_curriculum',
      { label: 'regime perspective' },
    ),
    pacingType: choice<'root' | 'linear' | 'step'>(
      [
        { value: 'root', label: 'Root / Sublinear Pacing (Smooth Expansion)' },
        { value: 'linear', label: 'Linear Pacing (Constant Rate)' },
        { value: 'step', label: 'Step-Wise Staged Curriculum' },
      ],
      'root',
      { label: 'pacing schedule' },
    ),
    mixtureTemp: slider(1.0, 5.0, 2.0, {
      step: 0.5,
      label: 'mixture temperature scaling T',
    }),
  })

  const { visualizerMode, pacingType, mixtureTemp } = state

  const results = useMemo(() => {
    if (visualizerMode === 'pacing_curriculum') {
      // Training progress 0% to 100%
      const steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      // Initial easy fraction lambda_0 = 20%
      const lambda0 = 20.0

      // Curves
      const activeData = steps.map((s) => {
        if (pacingType === 'root') {
          // sqrt pacing
          return Math.min(100.0, lambda0 + (100.0 - lambda0) * Math.sqrt(s / 100))
        } else if (pacingType === 'linear') {
          return Math.min(100.0, lambda0 + (100.0 - lambda0) * (s / 100))
        } else {
          // Step wise (3 stages: 20%, 60%, 100%)
          if (s < 33) return 20.0
          if (s < 66) return 60.0
          return 100.0
        }
      })

      // Constant baseline (100% full dataset from step 0)
      const fullDatasetBaseline = steps.map(() => 100.0)

      return {
        x: steps,
        y: activeData,
        baseline: fullDatasetBaseline,
        stat1: `Schedule: ${pacingType.toUpperCase()} pacing`,
        stat2: `At 50% training: ${activeData[5].toFixed(1)}% data active`,
        stat3: 'Initial seed: 20% easiest instances',
        warning:
          pacingType === 'step'
            ? 'Discrete step shifts can induce gradient shocks; continuous root pacing is smoother'
            : 'Early easy concentration guides parameters into stable optimization basins',
        axisXLabel: 'Training Step Progress (% of Total Budget)',
        axisYLabel: 'Active Available Dataset Fraction (%)',
        xRange: [0, 100] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'domain_mixture') {
      // 4 domains: 1: Web Crawl, 2: Code/GitHub, 3: High-Quality Papers/Books, 4: STEM/Math
      const domains = [1, 2, 3, 4]
      // Raw token proportions in natural crawl: Web=80%, Code=12%, Books=6%, Math=2%
      const rawWeights = [0.8, 0.12, 0.06, 0.02]
      const T = mixtureTemp

      // Temperature scaling: alpha_k proportional to w_k^(1/T)
      const tempWeights = rawWeights.map((w) => Math.pow(w, 1 / T))
      const sumTemp = tempWeights.reduce((a, b) => a + b, 0)
      const scaledPercents = tempWeights.map((tw) => (tw / sumTemp) * 100)
      const rawPercents = rawWeights.map((w) => w * 100)

      return {
        x: domains,
        y: scaledPercents,
        baseline: rawPercents,
        stat1: `Temperature T = ${T.toFixed(1)}`,
        stat2: `Math upweighting: ${rawPercents[3].toFixed(1)}% -> ${scaledPercents[3].toFixed(1)}%`,
        stat3: `Web suppression: ${rawPercents[0].toFixed(1)}% -> ${scaledPercents[0].toFixed(1)}%`,
        warning:
          T >= 4.0
            ? 'Extreme temperature flattening: overweights low-volume domains, causing rapid repetition'
            : 'Balanced mixture temperature: enriches high-utility reasoning and code tokens',
        axisXLabel: 'Domain (1: Web Crawl, 2: Code, 3: Papers & Books, 4: STEM / Math)',
        axisYLabel: 'Sampling Allocation Probability (%)',
        xRange: [0.5, 4.5] as [number, number],
        yRange: [0, 85] as [number, number],
      }
    } else {
      // Epochs / Repetitions: 1 to 10
      const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Effective token value relative to fresh unique tokens (Muennighoff 2023)
      // Drops slowly for epochs 1-4, then falls off a cliff
      const efficiency = epochs.map((ep) => {
        if (ep <= 4) return 100.0 - (ep - 1) * 3.5
        return Math.max(10.0, 89.5 - Math.pow(ep - 4, 1.8) * 12.0)
      })
      // Unique fresh token baseline (100% across all epochs)
      const uniqueBaseline = epochs.map(() => 100.0)

      return {
        x: epochs,
        y: efficiency,
        baseline: uniqueBaseline,
        stat1: 'Epoch 1-4: >89% fresh token efficiency',
        stat2: 'Epoch 8+: <30% marginal learning utility',
        stat3: 'Safe repetition threshold: ~4 epochs',
        warning: 'Exceeding 5 repetitions leads to severe memorization and capacity saturation',
        axisXLabel: 'Number of Dataset Repetitions / Training Epochs',
        axisYLabel: 'Marginal Token Learning Efficiency (%)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, pacingType, mixtureTemp])

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
      title="Curriculum Pacing, Domain Mixtures & Repetition Dynamics"
      state={state}
      hoverReadout={false}
      caption="Interactive exploration of training curricula: Continuous pacing functions from easy to hard, temperature-scaled heterogeneous data mixtures, and multi-epoch repetition decay."
      readouts={
        <>
          <Readout label="Curriculum State" value={results.stat1} />
          <Readout label="Target Allocation" value={results.stat2} />
          <Readout label="Comparative Bound" value={results.stat3} />
          <Readout
            label="Optimization Guidance"
            value={results.warning}
            color={
              (visualizerMode === 'domain_mixture' && mixtureTemp >= 4.0) || visualizerMode === 'repetition_epochs'
                ? 'var(--blue-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Data Curriculum Dynamics">
        <Curve
          name={
            visualizerMode === 'pacing_curriculum'
              ? 'Unpaced Full Dataset Baseline'
              : visualizerMode === 'domain_mixture'
                ? 'Raw Unscaled Natural Crawl Proportions'
                : 'Ideal Unique Token Baseline (100%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'pacing_curriculum'
              ? 'Active Curriculum Data Slice'
              : visualizerMode === 'domain_mixture'
                ? 'Temperature-Scaled Mixture Allocation'
                : 'Effective Marginal Token Efficiency'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Schedule Coordinates" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
