import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TabularDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'tree_vs_neural' | 'numerical_embeddings' | 'column_interactions'>(
      [
        { value: 'tree_vs_neural', label: 'Decision Boundaries: GBDT Step vs Neural Smooth (Grinsztajn 2022)' },
        { value: 'numerical_embeddings', label: 'Numerical Feature Tokenization: Linear vs PLE (Gorishniy 2022)' },
        { value: 'column_interactions', label: 'FT-Transformer Column Attention Strengths (Gorishniy 2021)' },
      ],
      'tree_vs_neural',
      { label: 'tabular architecture' },
    ),
    rotationAngle: slider(0, 45, 0, {
      step: 5,
      label: 'coordinate axis rotation theta (degrees)',
    }),
    embeddingBins: slider(2, 10, 5, {
      step: 1,
      label: 'PLE numerical embedding frequency bins',
    }),
  })

  const { visualizerMode, rotationAngle, embeddingBins } = state

  const results = useMemo(() => {
    if (visualizerMode === 'tree_vs_neural') {
      // Feature x1 from 0 to 10
      const x1 = [0, 1, 2, 3, 3.01, 5, 6, 6.01, 8, 9, 10]
      // GBDT step-function boundary on x2: axis-aligned thresholds at x1 = 3 and x1 = 6
      const treeBoundary = [2.0, 2.0, 2.0, 2.0, 6.5, 6.5, 6.5, 4.0, 4.0, 4.0, 4.0]
      // Neural network smooth boundary: sigmoid transition
      const neuralBoundary = x1.map((x) => 2.0 + 4.5 / (1 + Math.exp(-1.8 * (x - 4.5))))

      // When coordinate system is rotated by theta, tree efficiency degrades sharply
      const treeAccuracy = Math.max(55, 92 - rotationAngle * 0.75)
      const neuralAccuracy = 84.5 // rotation invariant

      return {
        x: x1,
        y: treeBoundary,
        baseline: neuralBoundary,
        stat1: `Axis rotation theta = ${rotationAngle}°`,
        stat2: `GBDT accuracy: ${treeAccuracy.toFixed(1)}%`,
        stat3: `Neural accuracy: ${neuralAccuracy.toFixed(1)}%`,
        warning:
          rotationAngle > 20
            ? 'Axis rotation destroys GBDT axis-aligned efficiency; neural networks remain rotation-invariant'
            : 'Natural unrotated tabular features strongly favor axis-aligned decision tree splits',
        axisXLabel: 'Continuous Feature x_1 (e.g. Normalised Age)',
        axisYLabel: 'Continuous Feature x_2 Decision Boundary (e.g. Income Threshold)',
        xRange: [0, 10] as [number, number],
        yRange: [0, 8] as [number, number],
      }
    } else if (visualizerMode === 'numerical_embeddings') {
      // Continuous input feature x from 0 to 1
      const xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      const K = embeddingBins

      // Linear embedding response: w * x
      const linearResp = xs.map((x) => x * 10)
      // Piecewise Linear / Periodic embedding response: capturing periodic or multi-modal jumps
      const pleResp = xs.map((x) => 5 * Math.sin(x * K * Math.PI) + 5 * x)

      return {
        x: xs.map((x) => x * 100),
        y: pleResp,
        baseline: linearResp,
        stat1: `PLE bins K = ${K}`,
        stat2: 'Non-linear continuous expressivity',
        stat3: 'Captures multi-modal target jumps',
        warning:
          'Scalar linear projection cannot represent non-monotonic relationships; PLE tokenizes continuous features',
        axisXLabel: 'Raw Continuous Feature Value x (%)',
        axisYLabel: 'Embedded Dimension Activation Response',
        xRange: [0, 100] as [number, number],
        yRange: [-6, 16] as [number, number],
      }
    } else {
      // 6 tabular columns: 1: Age, 2: Income, 3: Education, 4: CreditScore, 5: LoanAmount, 6: DebtRatio
      const cols = [1, 2, 3, 4, 5, 6]
      // Learned self-attention weights from Target Column "LoanAmount" (Col 5) to other columns
      const attentionWeights = [12.0, 38.5, 8.2, 26.4, 10.0, 4.9]
      const uniformFlat = cols.map(() => 100 / 6)

      return {
        x: cols,
        y: attentionWeights,
        baseline: uniformFlat,
        stat1: 'Top interaction: Income -> LoanAmount (38.5%)',
        stat2: 'Secondary: CreditScore (26.4%)',
        stat3: 'FT-Transformer pairwise self-attention',
        warning: 'FT-Transformer automatically discovers pairwise non-linear interactions across column tokens',
        axisXLabel: 'Column Token (1: Age, 2: Income, 3: Education, 4: Credit, 5: Loan, 6: Debt)',
        axisYLabel: 'Learned Multi-Head Attention Weight (%)',
        xRange: [0.5, 6.5] as [number, number],
        yRange: [0, 45] as [number, number],
      }
    }
  }, [visualizerMode, rotationAngle, embeddingBins])

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
      title="Tabular Representations, Embeddings & Decision Space Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating tabular data representations: GBDT axis-aligned steps versus smooth neural boundaries under coordinate rotation, numerical feature tokenization (PLE), and FT-Transformer column self-attention."
      readouts={
        <>
          <Readout label="Representation Mode" value={results.stat1} />
          <Readout label="Model Efficiency" value={results.stat2} />
          <Readout label="Comparative Metric" value={results.stat3} />
          <Readout
            label="Inductive Bias Directive"
            value={results.warning}
            color={visualizerMode === 'tree_vs_neural' && rotationAngle > 20 ? 'var(--red-600)' : 'var(--blue-600)'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Tabular Representation Dynamics">
        <Curve
          name={
            visualizerMode === 'tree_vs_neural'
              ? 'Smooth Neural Network Boundary'
              : visualizerMode === 'numerical_embeddings'
                ? 'Naive Scalar Linear Projection'
                : 'Uniform Random Attention Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'tree_vs_neural'
              ? 'GBDT Axis-Aligned Step Boundary'
              : visualizerMode === 'numerical_embeddings'
                ? 'Piecewise Linear / Periodic Embedding (PLE)'
                : 'Learned Column Attention Weight'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Evaluation Coordinates" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
