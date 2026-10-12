import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function HardExampleMiningVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'triplet_zones' | 'ohem_selection' | 'false_negative_hazard'>(
      [
        { value: 'triplet_zones', label: 'Triplet Mining Geometry & Semi-Hard Zones (Schroff 2015)' },
        { value: 'ohem_selection', label: 'OHEM Batch Pruning & Gradient Concentration (Shrivastava 2016)' },
        { value: 'false_negative_hazard', label: 'Hard Negative Similarity vs False-Negative Contamination' },
      ],
      'triplet_zones',
      { label: 'mining architecture' },
    ),
    margin: slider(0.2, 0.8, 0.4, {
      step: 0.05,
      label: 'triplet margin alpha',
    }),
    miningTopK: slider(8, 64, 32, {
      step: 4,
      label: 'OHEM selected hard samples K (from 128 proposals)',
    }),
  })

  const { visualizerMode, margin, miningTopK } = state

  const results = useMemo(() => {
    if (visualizerMode === 'triplet_zones') {
      // Negative distances from 0.1 to 1.8. Fixed positive distance d(a,p) = 0.60
      const dists = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
      const posDist = 0.6
      const alpha = margin

      // Triplet loss: max(0, posDist^2 - d^2 + alpha)
      const losses = dists.map((d) => Math.max(0, Math.pow(posDist, 2) - Math.pow(d, 2) + alpha))
      const lossZero = dists.map(() => 0)

      // Evaluation at d_neg = 0.75 (semi-hard example)
      const testD = 0.75
      const testLoss = Math.max(0, Math.pow(posDist, 2) - Math.pow(testD, 2) + alpha)

      return {
        x: dists,
        y: losses,
        baseline: lossZero,
        stat1: `Positive anchor distance: d(a,p) = ${posDist.toFixed(2)}`,
        stat2: `Semi-hard band: [${posDist.toFixed(2)}, ${Math.sqrt(Math.pow(posDist, 2) + alpha).toFixed(2)}]`,
        stat3: `Active loss at d=0.75: ${testLoss.toFixed(3)}`,
        warning:
          alpha > 0.6
            ? 'Large margin: admits too many hard negatives, risking training instability and collapse'
            : 'Balanced semi-hard regime: non-zero gradients without local minima traps',
        axisXLabel: 'Negative Distance d(a, n)',
        axisYLabel: 'Triplet Margin Loss Value',
        xRange: [0.05, 1.85] as [number, number],
        yRange: [0, 1.2] as [number, number],
      }
    } else if (visualizerMode === 'ohem_selection') {
      // 128 proposal candidates sorted by loss from highest (1) to lowest (128)
      const ranks = [1, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128]
      // Exponential loss decay: easy negatives have near-zero loss
      const losses = ranks.map((r) => 2.5 * Math.exp(-r / 22))

      // Gradient contribution: selected K samples provide 100% of backward pass
      const selectedLossSum = ranks.filter((r) => r <= miningTopK).reduce((sum, r) => sum + 2.5 * Math.exp(-r / 22), 0)
      const totalLossSum = ranks.reduce((sum, r) => sum + 2.5 * Math.exp(-r / 22), 0)
      const gradFrac = Math.min(100, (selectedLossSum / totalLossSum) * 100)

      const cutoffLine = ranks.map((r) => (r === miningTopK ? 2.5 : 0))

      return {
        x: ranks,
        y: losses,
        baseline: cutoffLine,
        stat1: `Retained: ${miningTopK} / 128 (${((miningTopK / 128) * 100).toFixed(0)}% batch)`,
        stat2: `Gradient energy captured: ${gradFrac.toFixed(1)}%`,
        stat3: `Backward pass speedup: ${(128 / miningTopK).toFixed(1)}x`,
        warning:
          miningTopK < 16
            ? 'Extreme pruning: risk of memorizing noisy outliers and gradient variance spikes'
            : 'Effective OHEM: suppresses flood of uninformative background proposals',
        axisXLabel: 'Candidate Proposal Rank (Sorted by Descending Loss)',
        axisYLabel: 'Per-Candidate Loss Magnitude',
        xRange: [0, 130] as [number, number],
        yRange: [0, 2.7] as [number, number],
      }
    } else {
      // Cosine similarity between anchor query and candidate negative: 0.1 to 0.95
      const similarities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
      // False negative contamination rate (% of unlabelled candidates that are actually relevant)
      const fnRate = similarities.map((s) => {
        if (s < 0.5) return 0.5
        return 0.5 + 48.0 * Math.pow((s - 0.5) / 0.45, 2.5)
      })
      const baselineSafe = similarities.map(() => 5.0)

      return {
        x: similarities,
        y: fnRate,
        baseline: baselineSafe,
        stat1: 'Safe threshold: similarity < 0.65',
        stat2: `At sim=0.90: ${fnRate[9].toFixed(1)}% false negatives`,
        stat3: 'True negative purity degrades sharply',
        warning:
          'Top-1 hard negatives frequently retrieve unlabelled true positives; requires cross-encoder verification',
        axisXLabel: 'Embedding Cosine Similarity with Anchor Query',
        axisYLabel: 'False Negative Contamination Rate (%)',
        xRange: [0.05, 1.0] as [number, number],
        yRange: [0, 55] as [number, number],
      }
    }
  }, [visualizerMode, margin, miningTopK])

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
      title="Hard-Example Mining & Negative Sampling Dynamics"
      state={state}
      hoverReadout={false}
      caption="Interactive exploration of metric learning margin geometries, online hard example mining (OHEM) loss pruning, and the false negative contamination hazard in aggressive negative mining."
      readouts={
        <>
          <Readout label="Operating Metric" value={results.stat1} />
          <Readout label="Mining Profile" value={results.stat2} />
          <Readout label="Gradient / Hazard" value={results.stat3} />
          <Readout
            label="Optimization Notice"
            value={results.warning}
            color={
              (visualizerMode === 'ohem_selection' && miningTopK < 16) || visualizerMode === 'false_negative_hazard'
                ? 'var(--red-600)'
                : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Mining Characteristics">
        <Curve
          name={
            visualizerMode === 'triplet_zones'
              ? 'Zero Loss Boundary'
              : visualizerMode === 'ohem_selection'
                ? 'Selection Cutoff Rank'
                : 'Acceptable Contamination Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'triplet_zones'
              ? 'Triplet Margin Loss'
              : visualizerMode === 'ohem_selection'
                ? 'Proposal Loss Curve'
                : 'False Negative Contamination'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Samples" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
