import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function GraphConstructionVisualizer() {
  const state = useFigureState({
    graphMode: choice<'knn_topology' | 'epsilon_rbf' | 'bipartite_projection'>(
      [
        {
          value: 'knn_topology',
          label: 'k-NN Topology: Giant Component vs Oversmoothing (Tenenbaum 2000, Kipf 2017)',
        },
        {
          value: 'epsilon_rbf',
          label: 'Similarity Thresholds: Epsilon Radius vs Gaussian RBF Bandwidth (von Luxburg 2007)',
        },
        {
          value: 'bipartite_projection',
          label: 'Interaction Graphs: Bipartite Projection Edge Explosion & Hub Cliques',
        },
      ],
      'knn_topology',
      { label: 'graph construction perspective' },
    ),
    param1: slider(1, 25, 6, {
      step: 1,
      label: 'nearest neighbors k / distance radius epsilon / hub degree',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'cluster separation / RBF bandwidth sigma / user count',
    }),
  })

  const { graphMode, param1, param2 } = state

  const results = useMemo(() => {
    if (graphMode === 'knn_topology') {
      // k from 1 to 25
      const kValues = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25]
      // Fraction of nodes in the giant connected component (%)
      const giantComponentPct = kValues.map((k) => Math.min(100, 15 + 85 * (1 - Math.exp(-0.45 * (k - 0.5)))))
      // Algebraic connectivity / Fiedler eigenvalue lambda_2 of normalized Laplacian (scaled x100)
      const fiedlerValue = kValues.map((k) => {
        if (k < 3) return 0.5
        return Math.min(65, Math.pow((k - 2) / 8, 1.3) * 22)
      })

      const currK = param1
      const currGiant = Math.min(100, 15 + 85 * (1 - Math.exp(-0.45 * (currK - 0.5))))
      const currFiedler = currK < 3 ? 0.5 : Math.min(65, Math.pow((currK - 2) / 8, 1.3) * 22)

      return {
        x: kValues,
        y: giantComponentPct,
        baseline: fiedlerValue,
        stat1: `Neighbor Degree: k = ${currK}`,
        stat2: `Giant Component: ${currGiant.toFixed(1)}% of nodes`,
        stat3: `Graph Connectivity λ2: ${(currFiedler / 100).toFixed(3)}`,
        warning:
          currK < 4
            ? 'Graph is severely disconnected into isolated multi-component islands; message passing cannot propagate across disconnected clusters'
            : currK > 15
              ? 'High k connects distant manifold clusters; shortcuts induce over-squashing and rapid over-smoothing in deep GNN layers'
              : 'Balanced k-NN graph preserves local manifold geodesic distances while forming a single connected giant component',
        axisXLabel: 'Number of Nearest Neighbors (k in k-NN Construction)',
        axisYLabel: 'Giant Connected Component (%) / Algebraic Connectivity (x100)',
        xRange: [0.5, 26] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (graphMode === 'epsilon_rbf') {
      // Distance metric d from 0 to 5.0
      const distances = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
      // Gaussian RBF kernel weight: W = exp(-d^2 / (2 * sigma^2))
      const sigma = 0.5 + param2 * 0.25 // 0.75 to 3.0
      const rbfWeights = distances.map((d) => 100 * Math.exp(-(d * d) / (2 * sigma * sigma)))

      // Epsilon hard cutoff at distance = param1 * 0.2
      const epsilonDist = param1 * 0.2
      const epsilonHardCut = distances.map((d) => (d <= epsilonDist ? 100 : 0))

      const currD = 1.5
      const currRbf = 100 * Math.exp(-(currD * currD) / (2 * sigma * sigma))
      const currEps = currD <= epsilonDist ? 100 : 0

      return {
        x: distances,
        y: rbfWeights,
        baseline: epsilonHardCut,
        stat1: `Epsilon Cutoff: eps = ${epsilonDist.toFixed(1)}`,
        stat2: `RBF Bandwidth: sigma = ${sigma.toFixed(2)}`,
        stat3: `Weight at d=1.5: ${currRbf.toFixed(1)}% (Eps: ${currEps}%)`,
        warning:
          'Epsilon graphs produce non-uniform degree distributions in variable-density data (dense regions explode in edges while sparse regions isolate); continuous RBF kernels soften boundary discontinuities',
        axisXLabel: 'Metric Feature Distance Between Node Pair ||x_i - x_j||',
        axisYLabel: 'Constructed Edge Adjacency Weight (%)',
        xRange: [0, 5.2] as [number, number],
        yRange: [-5, 105] as [number, number],
      }
    } else {
      // Hub item degree d_item from 2 to 50
      const hubDegrees = [2, 5, 10, 15, 20, 25, 30, 35, 40, 50]
      // Projected user-user edges created by single hub item: d * (d - 1) / 2
      const projectedEdges = hubDegrees.map((d) => (d * (d - 1)) / 2)
      // Linear baseline (preserving original bipartite edge count d)
      const bipartiteEdges = hubDegrees.map((d) => d)

      const currHub = param1 * 2 // 2 to 50
      const currProj = (currHub * (currHub - 1)) / 2

      return {
        x: hubDegrees,
        y: projectedEdges,
        baseline: bipartiteEdges.map((b) => b * 10), // scaled for visualization
        stat1: `Hub Item Degree: ${currHub} users`,
        stat2: `Projected Clique: ${currProj} edges`,
        stat3: `Density Explosion: ${(currProj / currHub).toFixed(1)}x expansion`,
        warning:
          currHub > 20
            ? 'Projecting bipartite graphs via BB^T creates dense O(d^2) cliques around popular items; GNN message passing becomes overwhelmed by dense hub shortcuts'
            : 'Low-degree interactions retain heterogeneous bipartite bipartite structure without quadratic edge inflation',
        axisXLabel: 'Item Node Degree in Bipartite Interaction Graph',
        axisYLabel: 'Induced Edges in One-Mode Homogeneous Projection (d(d-1)/2)',
        xRange: [1, 52] as [number, number],
        yRange: [0, Math.max(...projectedEdges) * 1.15] as [number, number],
      }
    }
  }, [graphMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Graph Construction: k-NN Manifold Topology, Similarity Kernels, and Interaction Projections"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Construction Parameter" value={results.stat1} />
          <Readout label="Connectivity Metric" value={results.stat2} />
          <Readout label="Spectral / Topological Property" value={results.stat3} />
          <Readout label="Inductive Bias Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Graph Adjacency Structure and Topological Response">
        <Curve
          name={
            graphMode === 'knn_topology'
              ? 'Algebraic Connectivity / Fiedler Eigenvalue (x100)'
              : graphMode === 'epsilon_rbf'
                ? 'Hard Epsilon Neighborhood Step Function (%)'
                : 'Original Bipartite Edge Count (x10)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            graphMode === 'knn_topology'
              ? 'Giant Connected Component Node Fraction (%)'
              : graphMode === 'epsilon_rbf'
                ? 'Smooth Gaussian RBF Kernel Weight (%)'
                : 'Projected One-Mode Clique Edges (Quadratic)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Evaluated Graph Topologies" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
