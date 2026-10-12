import { useMemo } from 'react'
import { choice, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ProjectionForExplorationVisualizer() {
  const state = useFigureState({
    method: choice<'pca' | 'tsne' | 'umap'>(
      [
        { value: 'pca', label: 'PCA (Linear Variance Preservation)' },
        { value: 'tsne', label: 't-SNE (Local Neighborhoods & Island Clustering)' },
        { value: 'umap', label: 'UMAP (Local Topology + Global Distance Memory)' },
      ],
      'tsne',
      { label: 'projection algorithm' },
    ),
    perplexity: slider(5, 80, 30, {
      step: 5,
      label: 'perplexity / n_neighbors parameter',
    }),
    clusterSeparation: slider(0.5, 3.0, 1.8, {
      step: 0.25,
      label: 'ground-truth cluster separation distance',
    }),
  })

  const { method, perplexity, clusterSeparation } = state

  const results = useMemo(() => {
    // 3 ground-truth clusters in high-dimensional space:
    // Cluster 1 (Red/Slot 0): 10 points
    // Cluster 2 (Blue/Slot 1): 10 points
    // Cluster 3 (Green/Slot 2): 10 points
    const pointsX: number[] = []
    const pointsY: number[] = []
    const clusterSlots: number[] = []

    // Base synthetic points around centers (-sep, 0), (sep, sep), (0, -sep)
    const centers = [
      [-clusterSeparation * 2, -clusterSeparation],
      [clusterSeparation * 1.5, clusterSeparation * 1.2],
      [0.5, -clusterSeparation * 1.8],
    ]

    for (let c = 0; c < 3; c++) {
      const cx = centers[c][0]
      const cy = centers[c][1]

      for (let i = 0; i < 10; i++) {
        const angle = (i / 10) * 2 * Math.PI
        const r = 0.6 + 0.3 * Math.sin(i * 3)

        let x = cx + r * Math.cos(angle)
        let y = cy + r * Math.sin(angle)

        if (method === 'pca') {
          // PCA linearly projects; clusters with non-linear or lower variance overlap
          x = cx * 0.7 + r * Math.cos(angle) * 1.5
          y = cy * 0.4 + r * Math.sin(angle) * 1.1
        } else if (method === 'tsne') {
          // t-SNE expands clusters into circular islands
          // If perplexity is very low (<15), clusters fragment into sub-blobs
          const frag = perplexity < 15 ? 1.8 * Math.sin(i * 1.5) : 0
          // If perplexity is high (>60), clusters pull together
          const pull = perplexity > 50 ? 0.4 : 1.2
          x = cx * pull + frag + r * 0.9 * Math.cos(angle)
          y = cy * pull + r * 0.9 * Math.sin(angle)
        } else {
          // UMAP preserves tight local density with semi-global geometry
          const uPull = perplexity < 15 ? 1.3 : 1.0
          x = cx * uPull + r * 0.75 * Math.cos(angle)
          y = cy * uPull + r * 0.75 * Math.sin(angle)
        }

        pointsX.push(x)
        pointsY.push(y)
        clusterSlots.push(c)
      }
    }

    // Reference boundary / axis bounds
    return {
      x: pointsX,
      y: pointsY,
      slots: clusterSlots,
      diagnostic:
        method === 'pca'
          ? 'Linear projection: Preserves global distances, but clusters partially overlap'
          : method === 'tsne'
            ? perplexity < 15
              ? 'Low perplexity artifact: Clusters artificially fragmented into sub-clumps'
              : 'Optimal t-SNE: Clear cluster islands, but inter-cluster distances are warped'
            : 'UMAP projection: Balances tight local grouping with broad inter-cluster topology',
    }
  }, [method, perplexity, clusterSeparation])

  const axisX = useAxis({
    label: method === 'pca' ? 'Principal Component 1 (PC1)' : `${method.toUpperCase()} Dimension 1`,
    range: [-8, 8],
  })

  const axisY = useAxis({
    label: method === 'pca' ? 'Principal Component 2 (PC2)' : `${method.toUpperCase()} Dimension 2`,
    range: [-8, 8],
  })

  return (
    <Figure
      title="High-Dimensional Projection & Manifold Embedding Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating PCA, t-SNE, and UMAP projections on 3 high-dimensional clusters. Each method introduces distinct geometric distortions: PCA collapses non-linear bounds, while t-SNE distorts cluster scale and inter-cluster distances."
      readouts={
        <>
          <Readout label="Projection Mechanism" value={results.diagnostic} />
          {method === 'tsne' && (
            <Readout
              label="Perplexity Caveat"
              value={
                perplexity < 15
                  ? 'Perplexity too low: False sub-clusters generated'
                  : perplexity > 50
                    ? 'Perplexity high: Inter-cluster separation reduced'
                    : 'Perplexity well-calibrated (30-50)'
              }
            />
          )}
          {method === 'pca' && (
            <Readout
              label="Explained Variance"
              value="Preserves global metric distances; unable to unfold non-linear manifolds"
            />
          )}
          {method === 'umap' && (
            <Readout
              label="Topological Preservation"
              value="Faster scaling than t-SNE; retains broader continuum between clusters"
            />
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="2D Projection Space">
        <Points name="Projected High-Dimensional Observations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
