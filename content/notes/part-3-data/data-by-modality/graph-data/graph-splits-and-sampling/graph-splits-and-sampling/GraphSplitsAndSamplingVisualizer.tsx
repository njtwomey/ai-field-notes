import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function GraphSplitsAndSamplingVisualizer() {
  const state = useFigureState({
    splitMode: choice<'neighborhood_explosion' | 'link_negatives' | 'inductive_gap'>(
      [
        {
          value: 'neighborhood_explosion',
          label: 'Neighbor Sampling: Full Expansion vs GraphSAGE Bounded Fan-out (Hamilton 2017)',
        },
        {
          value: 'link_negatives',
          label: 'Negative Sampling: Uniform Random vs Hard 2-Hop Negative Mining (Grover 2016)',
        },
        {
          value: 'inductive_gap',
          label: 'Split Protocol: Transductive vs Inductive Generalization Gap (Chiang 2019)',
        },
      ],
      'neighborhood_explosion',
      { label: 'graph sampling perspective' },
    ),
    param1: slider(1, 5, 3, {
      step: 1,
      label: 'GNN layer depth L / hard negative share (%) / test node ratio',
    }),
    param2: slider(5, 30, 15, {
      step: 5,
      label: 'average degree d / GraphSAGE budget S / train density',
    }),
  })

  const { splitMode, param1, param2 } = state

  const results = useMemo(() => {
    if (splitMode === 'neighborhood_explosion') {
      // GNN layers from 1 to 5
      const layerDepths = [1, 2, 3, 4, 5]
      const avgDegree = param2 // 5 to 30
      // Full batch exponential node count: d^L (capped to avoid JS numeric overflow)
      const fullBatchNodes = layerDepths.map((l) => Math.min(1e6, Math.pow(avgDegree, l)))
      // GraphSAGE bounded sampling: S_1=15, S_2=10, S_3=5...
      const sageBudgets = [15, 10, 5, 5, 5]
      const sageNodes = layerDepths.map((l) => {
        let prod = 1
        for (let i = 0; i < l; i++) prod *= sageBudgets[i]
        return prod
      })

      const currL = param1
      const currFull = Math.min(1e6, Math.pow(avgDegree, currL))
      let currSage = 1
      for (let i = 0; i < currL; i++) currSage *= sageBudgets[i]

      return {
        x: layerDepths,
        y: fullBatchNodes.map((n) => Math.log10(Math.max(1, n))), // log10 scale
        baseline: sageNodes.map((n) => Math.log10(Math.max(1, n))),
        stat1: `Layer Depth: L = ${currL} hops`,
        stat2: `Full Expansion: ${currFull.toLocaleString()} nodes (log10: ${Math.log10(currFull).toFixed(1)})`,
        stat3: `GraphSAGE Sampled: ${currSage.toLocaleString()} nodes (Bounded)`,
        warning:
          currL >= 3 && avgDegree >= 15
            ? 'Catastrophic neighborhood explosion; full-batch message expansion aggregates >3,000 nodes per seed, exhausting GPU memory; fixed-budget sampling is mandatory'
            : 'Bounded neighborhood fan-out decouples batch memory footprint from global graph density and degree distribution',
        axisXLabel: 'GNN Message Passing Layer Depth (Hops L)',
        axisYLabel: 'Aggregated Receptive Field Size (Log10 Node Count)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [0, 6.5] as [number, number],
      }
    } else if (splitMode === 'link_negatives') {
      // Negative candidate rank positions: 1 to 20
      const ranks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]
      // Hard negative share % = param1 * 20 (20% to 100%)
      const hardShare = param1 * 20
      // Model Link Prediction AUC (%) under uniform random negatives (easy to separate)
      const uniformAuc = ranks.map(() => 94.5)
      // Model Link Prediction AUC (%) under hard 2-hop negatives (sharing common neighbors)
      const hardAuc = ranks.map((r) => Math.max(55, 94.5 - Math.pow(r / 10, 1.3) * (hardShare * 0.28)))

      const currHardShare = param1 * 20
      const currScore = Math.max(55, 94.5 - currHardShare * 0.32)

      return {
        x: ranks,
        y: hardAuc,
        baseline: uniformAuc,
        stat1: `Hard 2-Hop Negatives: ${currHardShare}%`,
        stat2: `Uniform Negative AUC: 94.5%`,
        stat3: `Hard Negative AUC: ${currScore.toFixed(1)}%`,
        warning:
          currHardShare > 40
            ? 'Hard 2-hop non-edges force the link predictor to learn subtle relational motifs rather than relying on trivial degree/community shortcuts'
            : 'Uniform random non-edges are easily separated by trivial graph distance, yielding artificially inflated benchmark scores (>95%)',
        axisXLabel: 'Candidate Negative Edge Difficulty Rank',
        axisYLabel: 'Link Prediction Discriminative AUC (%)',
        xRange: [0.5, 20.5] as [number, number],
        yRange: [50, 100] as [number, number],
      }
    } else {
      // Training graph node density ratio: 10% to 90%
      const densityRatios = [10, 20, 30, 40, 50, 60, 70, 80, 90]
      // Transductive node classification accuracy (test nodes present in graph during training)
      const transductiveAcc = densityRatios.map((d) => 65 + Math.pow(d / 90, 0.7) * 26)
      // Inductive node classification accuracy (completely unseen test graphs/components)
      const inductiveAcc = densityRatios.map((d) => Math.max(45, 65 + Math.pow(d / 90, 0.7) * 26 - 12.5 + param1 * 1.2))

      const currDensity = param2 * 3 // 15 to 90
      const currTrans = 65 + Math.pow(currDensity / 90, 0.7) * 26
      const currInduct = Math.max(45, currTrans - 12.5 + param1 * 1.2)
      const gap = currTrans - currInduct

      return {
        x: densityRatios,
        y: transductiveAcc,
        baseline: inductiveAcc,
        stat1: `Graph Density: ${currDensity}%`,
        stat2: `Transductive Acc: ${currTrans.toFixed(1)}%`,
        stat3: `Inductive Acc: ${currInduct.toFixed(1)}% (Generalization Gap: ${gap.toFixed(1)}%)`,
        warning:
          'Transductive splits exploit global topological connectivity between train and test nodes; inductive splits force models to generalize purely through learned local neighborhood aggregation functions',
        axisXLabel: 'Graph Training Edge Connectivity Density (%)',
        axisYLabel: 'Node Classification Accuracy (%)',
        xRange: [5, 95] as [number, number],
        yRange: [40, 98] as [number, number],
      }
    }
  }, [splitMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Splitting and Sampling Graphs: Neighborhood Expansion, Negative Mining, and Inductive Gaps"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Sampling Configuration" value={results.stat1} />
          <Readout label="Full / Baseline Measure" value={results.stat2} />
          <Readout label="Sampled / Inductive Measure" value={results.stat3} />
          <Readout label="Algorithmic Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Graph Scalability and Generalization Curves">
        <Curve
          name={
            splitMode === 'neighborhood_explosion'
              ? 'GraphSAGE Bounded Receptive Field (Log10)'
              : splitMode === 'link_negatives'
                ? 'Trivial Uniform Random Non-Edges Baseline (%)'
                : 'Inductive Generalization Performance (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            splitMode === 'neighborhood_explosion'
              ? 'Full-Batch Exponential Expansion (Log10)'
              : splitMode === 'link_negatives'
                ? 'Hard 2-Hop Non-Edge Discrimination AUC (%)'
                : 'Transductive In-Graph Evaluation Accuracy (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Evaluated Regimes" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
