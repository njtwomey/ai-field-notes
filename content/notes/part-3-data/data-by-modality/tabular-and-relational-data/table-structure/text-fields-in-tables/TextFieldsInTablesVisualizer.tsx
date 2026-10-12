import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TextFieldsInTablesVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'dimensionality_dilution' | 'latency_frontier' | 'hashing_collision'>(
      [
        { value: 'dimensionality_dilution', label: 'Feature Dilution: Text Dimension vs Tabular Split Probability' },
        { value: 'latency_frontier', label: 'Inference Latency vs Semantic Capacity Trade-off (Reimers 2019)' },
        { value: 'hashing_collision', label: 'Hashed N-Grams: Hash Space Bits vs Collision Rate (Weinberger 2009)' },
      ],
      'dimensionality_dilution',
      { label: 'tabular-text integration mode' },
    ),
    tabularFeatures: slider(5, 50, 15, {
      step: 5,
      label: 'native tabular columns D_tab',
    }),
    textDimensions: slider(16, 768, 384, {
      step: 16,
      label: 'text embedding dimensions D_text',
    }),
  })

  const { visualizerMode, tabularFeatures, textDimensions } = state

  const results = useMemo(() => {
    if (visualizerMode === 'dimensionality_dilution') {
      // Vary text dimensionality from 16 to 768
      const dims = [16, 32, 64, 128, 256, 384, 512, 768]
      // Probability that a random tree split selects a native tabular feature: D_tab / (D_tab + D_text)
      const tabProbabilities = dims.map((d) => (tabularFeatures / (tabularFeatures + d)) * 100)
      const equalShareBaseline = dims.map(() => 50)

      const currProb = (tabularFeatures / (tabularFeatures + textDimensions)) * 100
      const currTotal = tabularFeatures + textDimensions

      return {
        x: dims,
        y: tabProbabilities,
        baseline: equalShareBaseline,
        stat1: `Native tabular features: ${tabularFeatures}`,
        stat2: `Total feature vector: ${currTotal} dims`,
        stat3: `Tabular split probability: ${currProb.toFixed(1)}%`,
        warning:
          currProb < 10
            ? 'Severe feature dilution: High-dimensional text embeddings drown out structured numerical signals in tree split search'
            : 'Balanced representation: Structured features retain sufficient split candidate exposure',
        axisXLabel: 'Text Feature Representation Dimensionality (D_text)',
        axisYLabel: 'Probability of Selecting Native Tabular Column (%)',
        xRange: [10, 800] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (visualizerMode === 'latency_frontier') {
      // Latency scale across 6 representations:
      // 0: One-hot/category, 1: Hashing trick, 2: TF-IDF, 3: TF-IDF+SVD, 4: SBERT-MiniLM, 5: Dense Transformer
      const reps = [1, 2, 3, 4, 5, 6]
      // Latency in milliseconds (log-like behavior): [0.05, 0.2, 0.8, 1.5, 6.0, 48.0]
      const latencies = [0.05, 0.2, 0.8, 1.5, 6.0, 48.0]
      const targetLatencyBaseline = reps.map(() => 5.0) // 5 ms SLA

      const isTransformer = textDimensions >= 256
      const estimatedLatency = isTransformer ? (textDimensions >= 512 ? 48.0 : 6.0) : 1.5

      return {
        x: reps,
        y: latencies,
        baseline: targetLatencyBaseline,
        stat1: `Current text dims: ${textDimensions}`,
        stat2: `Serving pipeline latency: ${estimatedLatency.toFixed(1)} ms`,
        stat3: isTransformer ? 'Transformer Encoder' : 'Linear / N-Gram Vectorizer',
        warning:
          estimatedLatency > 5.0
            ? 'Production serving bottleneck: End-to-end neural text encoding exceeds real-time tabular SLAs (< 5 ms)'
            : 'Real-time compatible: Sub-millisecond vectorization fits high-throughput tabular scoring loops',
        axisXLabel: 'Representation Pipeline (1=Hash, 3=TFIDF, 5=SBERT, 6=Full BERT)',
        axisYLabel: 'Inference Latency per Row (ms)',
        xRange: [0.8, 6.2] as [number, number],
        yRange: [0, 52] as [number, number],
      }
    } else {
      // Hashed n-grams collision rate as hash bit size B grows from 8 (256) to 18 (262k)
      const bits = [8, 10, 12, 14, 16, 18]
      // Assume vocabulary V = 20,000 unique n-grams
      const V = 20000
      // Expected collision rate approximation: 1 - exp(-V^2 / (2 * 2^B))
      const collisionRates = bits.map((b) => {
        const m = Math.pow(2, b)
        const expTerm = -(V * V) / (2 * m)
        return Math.min(100, Math.max(0, (1 - Math.exp(expTerm)) * 100))
      })
      const zeroCollisionBaseline = bits.map(() => 1.0)

      const currBits = Math.min(18, Math.max(8, Math.round(Math.log2(textDimensions * 50))))
      const currM = Math.pow(2, currBits)
      const currCollision = (1 - Math.exp(-(V * V) / (2 * currM))) * 100

      return {
        x: bits,
        y: collisionRates,
        baseline: zeroCollisionBaseline,
        stat1: `Vocabulary |V| = ${V.toLocaleString()} n-grams`,
        stat2: `Hash space: 2^${currBits} (${currM.toLocaleString()} bins)`,
        stat3: `Collision rate: ${currCollision.toFixed(1)}%`,
        warning:
          currCollision > 25
            ? 'High hash collision rate: Semantically distinct tokens map to identical feature bins, adding gradient noise'
            : 'Acceptable collision rate: Feature hashing preserves token differentiation within compact memory',
        axisXLabel: 'Hash Table Address Space (Bits B)',
        axisYLabel: 'Expected Feature Hash Collision Rate (%)',
        xRange: [7.5, 18.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, tabularFeatures, textDimensions])

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
      title="Text Fields in Tables: Feature Dilution, Serving Latency, and N-Gram Hashing Trade-offs"
      state={state}
      hoverReadout={false}
      caption="Evaluating free-text columns within tabular pipelines: Disproportionate feature dimensionality diluting tree splits, the latency frontier between linear n-grams and neural transformers, and hash space collision dynamics."
      readouts={
        <>
          <Readout label="Pipeline Scope" value={results.stat1} />
          <Readout label="Capacity Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'dimensionality_dilution'
                ? 'Split Exposure'
                : visualizerMode === 'latency_frontier'
                  ? 'Architecture'
                  : 'Collision'
            }
            value={results.stat3}
            color={visualizerMode === 'latency_frontier' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Design Guidance" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Tabular-Text Representation Profile">
        <Curve
          name={
            visualizerMode === 'dimensionality_dilution'
              ? '50% Equal Partition Reference'
              : visualizerMode === 'latency_frontier'
                ? '5 ms Real-Time SLA Limit'
                : '1% Negligible Collision Threshold'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'dimensionality_dilution'
              ? 'Tabular Column Selection Probability'
              : visualizerMode === 'latency_frontier'
                ? 'Per-Sample Inference Latency'
                : 'N-Gram Hash Collision Probability'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Coordinates" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
