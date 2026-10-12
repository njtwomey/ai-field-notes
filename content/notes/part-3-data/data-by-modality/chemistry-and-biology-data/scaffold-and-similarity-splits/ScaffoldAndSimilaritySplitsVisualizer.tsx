import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ScaffoldAndSimilaritySplitsVisualizer() {
  const state = useFigureState({
    mode: choice<'split_generalization' | 'cluster_distribution' | 'distance_separation'>(
      [
        {
          value: 'split_generalization',
          label: 'Generalization Gap: Random vs Bemis-Murcko Scaffold vs Butina (Wu 2018)',
        },
        {
          value: 'cluster_distribution',
          label: 'Scaffold Heavy Tail: Cluster Size Distribution & Greedy Allocation (Bemis 1996)',
        },
        {
          value: 'distance_separation',
          label: 'Chemical Distance: Max Train-Test Tanimoto Similarity Distribution',
        },
      ],
      'split_generalization',
      { label: 'split evaluation' },
    ),
    param1: slider(20, 80, 50, {
      step: 10,
      label: 'similarity cutoff threshold tau (%) / super-scaffold proportion (%)',
    }),
    param2: slider(1, 5, 3, {
      step: 1,
      label: 'library scaffold diversity index / test holdout budget (%)',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'split_generalization') {
      const tau = param1 // 20% to 80%
      // Compare 3 split paradigms across increasing model complexity or chemical distance
      const complexitySteps = [1, 2, 3, 4, 5, 6, 7, 8]
      const randomTestAUC: number[] = []
      const scaffoldTestAUC: number[] = []

      for (const step of complexitySteps) {
        // Random split test performance remains high (memorizes congeneric series)
        randomTestAUC.push(93.5 + 0.6 * step)
        // Scaffold split performance reflects genuine extrapolation, dropping with complexity/shift
        const baseDrop = 22.0 * (tau / 50)
        scaffoldTestAUC.push(Math.max(55, 88.0 - baseDrop * 0.4 + 1.2 * step - 0.25 * step * step))
      }

      const activeRandom = 93.5 + 0.6 * 5
      const baseDrop = 22.0 * (tau / 50)
      const activeScaffold = Math.max(55, 88.0 - baseDrop * 0.4 + 1.2 * 5 - 0.25 * 25)

      return {
        x: complexitySteps,
        y1: randomTestAUC,
        y2: scaffoldTestAUC,
        stat1: `Cutoff τ = ${tau}% similarity`,
        stat2: `Random Test AUC: ${activeRandom.toFixed(1)}% (inflated)`,
        stat3: `Scaffold Test AUC: ${activeScaffold.toFixed(1)}% (honest)`,
        insight: `Generalization deficit: ${(activeRandom - activeScaffold).toFixed(1)}% drop on novel chemotypes`,
        axisXLabel: 'Model Capacity / Scaffold Divergence Step',
        axisYLabel: 'Test ROC-AUC Score (%)',
        xRange: [0.5, 8.5] as [number, number],
        yRange: [50, 100] as [number, number],
      }
    } else if (mode === 'cluster_distribution') {
      // Power-law distribution of scaffold sizes in a typical 5,000 compound screening library
      // Rank 1 to 20 dominant scaffolds
      const ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
      const rawClusterSizes: number[] = []
      const cumulativeFractions: number[] = []

      let cumulative = 0
      const totalMols = 5000
      const superScaffoldFactor = param1 / 50 // 0.4 to 1.6

      for (const r of ranks) {
        // Zipfian distribution: size ~ C / r^gamma
        const size = Math.round((950 * superScaffoldFactor) / Math.pow(r, 0.85))
        cumulative += size
        rawClusterSizes.push(size)
        cumulativeFractions.push(Math.min(100, (cumulative / totalMols) * 100))
      }

      return {
        x: ranks,
        y1: rawClusterSizes,
        y2: cumulativeFractions,
        stat1: `Top scaffold: ${rawClusterSizes[0]} molecules`,
        stat2: `Top 10 hold ${cumulativeFractions[9]?.toFixed(0) ?? 40}% of library`,
        stat3: 'Greedy split: 80% train / 10% val / 10% test',
        insight: 'Super-scaffolds disrupt split ratios if cluster size exceeds test partition budget',
        axisXLabel: 'Scaffold Cluster Rank (1 = largest)',
        axisYLabel: 'Molecules per Scaffold & Cumulative Library (%)',
        xRange: [0.5, 21] as [number, number],
        yRange: [0, 1000] as [number, number],
      }
    } else {
      // Maximum pairwise Tanimoto similarity between test set and nearest training molecule
      // Histogram percentiles from 10% to 100%
      const similarityBins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
      const randomNearSim: number[] = []
      const scaffoldNearSim: number[] = []

      for (const sim of similarityBins) {
        // Random split test molecules almost always have a near-identical analog (sim >= 0.85)
        const randProb = 100 / (1 + Math.exp(-0.15 * (sim - 75)))
        // Scaffold split enforces severe separation (sim rarely exceeds 0.50)
        const scafProb = 100 / (1 + Math.exp(-0.12 * (sim - 42)))
        randomNearSim.push(randProb)
        scaffoldNearSim.push(scafProb)
      }

      return {
        x: similarityBins,
        y1: randomNearSim,
        y2: scaffoldNearSim,
        stat1: 'Tanimoto: T(a, b) = |a ∩ b| / |a ∪ b|',
        stat2: 'Random max Sim: ~0.92 (near twins)',
        stat3: 'Scaffold max Sim: ~0.45 (distant)',
        insight: 'Scaffold splits eliminate chemical memorization shortcuts',
        axisXLabel: 'Nearest-Neighbor Tanimoto Similarity (%)',
        axisYLabel: 'Cumulative Compounds (%)',
        xRange: [15, 102] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Scaffold and Similarity Splits: Performance Leakage, Power-Law Scaffolds, and Chemical Distances"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Split Criterion" value={results.stat1} />
          <Readout label="Random Baseline" value={results.stat2} />
          <Readout label="Rigorous Split Metric" value={results.stat3} />
          <Readout label="Generalization Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Evaluation Fidelity and Chemical Cluster Partitioning Dynamics">
        <Curve
          name={
            mode === 'split_generalization'
              ? 'Random Train-Test Split (Inflated Test ROC-AUC %)'
              : mode === 'cluster_distribution'
                ? 'Cluster Size (Count of Molecules)'
                : 'Random Split Nearest-Neighbor Cumulative CDF (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'split_generalization'
              ? 'Bemis-Murcko Scaffold Split (Honest Generalization %)'
              : mode === 'cluster_distribution'
                ? 'Cumulative Library Fraction (%)'
                : 'Scaffold Split Nearest-Neighbor Cumulative CDF (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Regimes" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
