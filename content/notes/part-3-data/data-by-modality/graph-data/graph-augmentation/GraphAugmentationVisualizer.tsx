import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function GraphAugmentationVisualizer() {
  const state = useFigureState({
    augMode: choice<'dropedge_smoothing' | 'feature_masking' | 'contrastive_views'>(
      [
        {
          value: 'dropedge_smoothing',
          label: 'DropEdge: Mitigating Deep GCN Oversmoothing & Dirichlet Energy Decay (Rong 2020)',
        },
        {
          value: 'feature_masking',
          label: 'GraphMAE Masking: Attribute Corruption Ratio vs Downstream Accuracy (Hou 2022)',
        },
        {
          value: 'contrastive_views',
          label: 'GraphCL Views: Perturbation Severity vs Semantic Information Preservation (You 2020)',
        },
      ],
      'dropedge_smoothing',
      { label: 'graph augmentation mechanism' },
    ),
    param1: slider(1, 16, 8, {
      step: 1,
      label: 'GNN layer depth L / masking ratio (%) / perturbation severity (%)',
    }),
    param2: slider(1, 5, 3, {
      step: 1,
      label: 'drop probability p_edge / decoder depth / InfoNCE temperature',
    }),
  })

  const { augMode, param1, param2 } = state

  const results = useMemo(() => {
    if (augMode === 'dropedge_smoothing') {
      // GNN layer depth L from 1 to 16
      const depths = [1, 2, 4, 6, 8, 10, 12, 14, 16]
      // Standard GCN node representation Dirichlet Energy / Variance collapses exponentially
      const baselineEnergy = depths.map((d) => 100 * Math.exp(-0.48 * (d - 1)))
      // DropEdge (p_edge = param2 * 0.1, e.g. 0.3) preserves non-zero variance
      const pEdge = param2 * 0.1
      const dropEdgeEnergy = depths.map((d) => Math.max(12, 100 * Math.exp(-0.22 * (1 - pEdge) * (d - 1))))

      const currDepth = param1
      const currBase = 100 * Math.exp(-0.48 * (currDepth - 1))
      const currDrop = Math.max(12, 100 * Math.exp(-0.22 * (1 - pEdge) * (currDepth - 1)))
      const isOversmoothed = currBase < 5

      return {
        x: depths,
        y: dropEdgeEnergy,
        baseline: baselineEnergy,
        stat1: `Layer Depth: L = ${currDepth} layers`,
        stat2: `DropEdge Energy: ${currDrop.toFixed(1)}% (p=${pEdge.toFixed(1)})`,
        stat3: `Standard GCN: ${currBase.toFixed(1)}% (${isOversmoothed ? 'OVERSMOOTHED' : 'HEALTHY'})`,
        warning:
          currDepth >= 8 && isOversmoothed
            ? 'Standard GCN representations collapse to a single uniform point (oversmoothing); DropEdge acts as message-passing dropout, preserving node distinctiveness in deep networks'
            : 'Moderate GNN depth with DropEdge prevents over-smoothing while enabling wide topological receptive fields',
        axisXLabel: 'GNN Convolutional Layer Depth (L Layers)',
        axisYLabel: 'Node Representation Dirichlet Energy / Variance (%)',
        xRange: [0.5, 16.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (augMode === 'feature_masking') {
      // Masking ratio p_mask (%) from 10% to 80%
      const maskRatios = [10, 20, 30, 40, 50, 60, 70, 80]
      // GraphMAE downstream linear probing classification accuracy (%) on OGB
      const downstreamAcc = maskRatios.map((r) => {
        // Peaks around 40-50% masking
        if (r <= 50) {
          return 72.0 + Math.pow(r / 50, 1.2) * 8.5
        } else {
          return 80.5 - Math.pow((r - 50) / 30, 1.8) * 11.0
        }
      })
      // Reconstruction cosine loss error (%)
      const reconLoss = maskRatios.map((r) => 15 + Math.pow(r / 80, 1.5) * 55)

      const currRatio = Math.min(80, Math.max(10, param1 * 5))
      const currAcc =
        currRatio <= 50
          ? 72.0 + Math.pow(currRatio / 50, 1.2) * 8.5
          : 80.5 - Math.pow((currRatio - 50) / 30, 1.8) * 11.0
      const currLoss = 15 + Math.pow(currRatio / 80, 1.5) * 55

      return {
        x: maskRatios,
        y: downstreamAcc,
        baseline: reconLoss,
        stat1: `Feature Masking: p = ${currRatio}%`,
        stat2: `Downstream Accuracy: ${currAcc.toFixed(1)}%`,
        stat3: `Reconstruction Loss: ${currLoss.toFixed(1)}%`,
        warning:
          currRatio > 65
            ? 'Excessive masking destroys attribute semantics; the model lacks sufficient context to reconstruct molecular node properties'
            : currRatio < 25
              ? 'Insufficient masking allows trivial local identity copying; the model fails to learn structural topological dependencies'
              : 'Optimal 50% masking forces GraphMAE to infer node identities from multi-hop neighborhood graphs',
        axisXLabel: 'Node Feature Masking Ratio p_mask (%)',
        axisYLabel: 'Downstream Evaluation Accuracy (%) / Reconstruction Loss (%)',
        xRange: [5, 85] as [number, number],
        yRange: [10, 90] as [number, number],
      }
    } else {
      // Perturbation severity alpha (%) from 5% to 60%
      const severities = [5, 10, 15, 20, 25, 30, 40, 50, 60]
      // GraphCL Contrastive Mutual Information I(G1; G2) (%)
      const mutualInfo = severities.map((s) => Math.max(10, 95 * Math.exp(-0.045 * s)))
      // Downstream Contrastive Representation Quality (%)
      const downstreamTransfer = severities.map((s) => {
        if (s <= 25) {
          return 75 + Math.pow(s / 25, 1.1) * 8.2
        } else {
          return 83.2 - Math.pow((s - 25) / 35, 1.6) * 22.0
        }
      })

      const currSev = Math.min(60, Math.max(5, param1 * 4))
      const currMi = Math.max(10, 95 * Math.exp(-0.045 * currSev))
      const currTrans =
        currSev <= 25 ? 75 + Math.pow(currSev / 25, 1.1) * 8.2 : 83.2 - Math.pow((currSev - 25) / 35, 1.6) * 22.0

      return {
        x: severities,
        y: downstreamTransfer,
        baseline: mutualInfo,
        stat1: `Perturbation Scale: alpha = ${currSev}%`,
        stat2: `Contrastive Transfer: ${currTrans.toFixed(1)}%`,
        stat3: `Mutual Information: ${currMi.toFixed(1)}%`,
        warning:
          currSev > 35
            ? 'Severe graph augmentations (dropping >35% edges/nodes) trigger semantic drift; the augmented view ceases to represent the original class'
            : 'Mild stochastic perturbations generate challenging positive pairs for InfoNCE contrastive learning while preserving core graph semantics',
        axisXLabel: 'Stochastic View Augmentation Severity (Drop/Mask Share %)',
        axisYLabel: 'Transfer Performance (%) / View Mutual Information (%)',
        xRange: [2, 63] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [augMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Graph Augmentation: DropEdge Regularization, Feature Masking, and Contrastive Views"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Augmentation Setting" value={results.stat1} />
          <Readout label="Primary Representation Signal" value={results.stat2} />
          <Readout label="Baseline / Transfer Metric" value={results.stat3} />
          <Readout label="Topological Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Graph Representation Invariance and Regularization Response">
        <Curve
          name={
            augMode === 'dropedge_smoothing'
              ? 'Unaugmented GCN Dirichlet Energy Collapse'
              : augMode === 'feature_masking'
                ? 'Feature Reconstruction Cosine Loss Error (%)'
                : 'Inter-View Mutual Information I(G1; G2) (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            augMode === 'dropedge_smoothing'
              ? 'DropEdge Preserved Representation Energy (%)'
              : augMode === 'feature_masking'
                ? 'GraphMAE Downstream Classification Accuracy (%)'
                : 'GraphCL Downstream Contrastive Transfer Score (%)'
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
