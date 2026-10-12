import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function EmbeddingBasedExplorationVisualizer() {
  const state = useFigureState({
    retrievalMode: choice<'neighbors' | 'ood_outliers' | 'zero_shot'>(
      [
        { value: 'neighbors', label: 'Semantic Nearest-Neighbour Query' },
        { value: 'ood_outliers', label: 'k-NN Out-of-Distribution Outlier Detection' },
        { value: 'zero_shot', label: 'Zero-Shot Prototype Centroids & Partitioning' },
      ],
      'neighbors',
      { label: 'embedding exploration lens' },
    ),
    simThreshold: slider(0.4, 0.95, 0.75, {
      step: 0.05,
      label: 'cosine similarity retrieval cutoff',
    }),
    outlierThreshold: slider(0.4, 1.2, 0.75, {
      step: 0.05,
      label: 'mean k-NN distance OOD threshold',
    }),
  })

  const { retrievalMode, simThreshold, outlierThreshold } = state

  const results = useMemo(() => {
    // Generate synthetic points on 2D embedding plane (normalized)
    // Query point at [0, 0]
    // 2 dense semantic clusters + 3 isolated OOD points
    const pointsX: number[] = []
    const pointsY: number[] = []

    // Cluster A (Text Query neighborhood: center [1.5, 1.0], 15 points)
    for (let i = 0; i < 15; i++) {
      const a = (i / 15) * 2 * Math.PI
      const r = 0.5 + 0.3 * Math.sin(i * 4)
      pointsX.push(1.5 + r * Math.cos(a))
      pointsY.push(1.0 + r * Math.sin(a))
    }

    // Cluster B (Opposite semantic topic: center [-1.8, -1.2], 15 points)
    for (let i = 0; i < 15; i++) {
      const a = (i / 15) * 2 * Math.PI
      const r = 0.6 + 0.2 * Math.cos(i * 3)
      pointsX.push(-1.8 + r * Math.cos(a))
      pointsY.push(-1.2 + r * Math.sin(a))
    }

    // 4 OOD outlier points far from clusters
    const oodX = [3.5, -3.2, 0.2, -0.8]
    const oodY = [-2.8, 3.1, -3.4, 3.2]
    oodX.forEach((x, idx) => {
      pointsX.push(x)
      pointsY.push(oodY[idx])
    })

    // Query point
    const queryPt: [number, number] = [1.2, 0.8]

    // Compute distances from query point to all points
    const retrievedX: number[] = []
    const retrievedY: number[] = []
    pointsX.forEach((px, i) => {
      const py = pointsY[i]
      const dist = Math.sqrt(Math.pow(px - queryPt[0], 2) + Math.pow(py - queryPt[1], 2))
      // Approximate cosine sim ~ 1 - dist / 3
      const approxSim = Math.max(0, 1 - dist / 3.2)
      if (approxSim >= simThreshold) {
        retrievedX.push(px)
        retrievedY.push(py)
      }
    })

    // Decision boundary line for zero_shot: line perpendicular to line between cluster centroids
    const boundX = [-3.5, 3.5]
    const boundY = [2.5, -2.5]

    return {
      allX: pointsX,
      allY: pointsY,
      queryX: [queryPt[0]],
      queryY: [queryPt[1]],
      retrievedX,
      retrievedY,
      boundX,
      boundY,
      retrievedCount: retrievedX.length,
      oodCount: outlierThreshold < 0.8 ? 4 : 2,
    }
  }, [retrievalMode, simThreshold, outlierThreshold])

  const axisX = useAxis({
    label: 'Latent Embedding Coordinate Z1',
    range: [-4.5, 4.5],
  })

  const axisY = useAxis({
    label: 'Latent Embedding Coordinate Z2',
    range: [-4.5, 4.5],
  })

  return (
    <Figure
      title="Unstructured Data Embedding Explorer"
      state={state}
      hoverReadout={false}
      caption="Exploring unstructured images, text, and audio via foundation model embeddings. Vector representations enable semantic nearest-neighbor retrieval, isolation of multimodal outliers, and zero-shot partition audits."
      readouts={
        <>
          {retrievalMode === 'neighbors' && (
            <>
              <Readout label="Retrieved Neighbours" value={`${results.retrievedCount} instances`} />
              <Readout
                label="Retrieval Status"
                value={
                  results.retrievedCount > 0
                    ? `Matched with cosine sim ≥ ${simThreshold.toFixed(2)}`
                    : 'Threshold too strict: No matches'
                }
              />
            </>
          )}
          {retrievalMode === 'ood_outliers' && (
            <>
              <Readout label="OOD Anomalies Flagged" value={`${results.oodCount} isolated points`} />
              <Readout
                label="Auditing Priority"
                value="Flagged records include corrupted files, blank inputs, or out-of-domain noise"
              />
            </>
          )}
          {retrievalMode === 'zero_shot' && (
            <>
              <Readout label="Zero-Shot Boundary" value="Separates Class Prompt A vs Class Prompt B" />
              <Readout
                label="Semantic Separation"
                value="Clear cluster margin indicates strong zero-shot discriminability"
              />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Embedding Latent Space">
        <Points name="Background Unstructured Embeddings" x={results.allX} y={results.allY} slot={0} size={6} />
        {retrievalMode === 'neighbors' && (
          <>
            <Points name="Semantic Query Point" x={results.queryX} y={results.queryY} slot={2} size={11} />
            <Points
              name="Retrieved Nearest Neighbours"
              x={results.retrievedX}
              y={results.retrievedY}
              slot={1}
              size={8}
            />
          </>
        )}
        {retrievalMode === 'zero_shot' && (
          <Curve name="Zero-Shot Decision Boundary" x={results.boundX} y={results.boundY} slot={2} dashed width={2} />
        )}
      </Plot>
    </Figure>
  )
}
