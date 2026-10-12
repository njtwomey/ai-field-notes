import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function GraphFeaturesVisualizer() {
  const state = useFigureState({
    featureMode: choice<'centrality_profiles' | 'laplacian_pe' | 'rwpe_return'>(
      [
        {
          value: 'centrality_profiles',
          label: 'Centrality Metrics: Degree vs PageRank vs Betweenness Bridge Peaks (Page 1999)',
        },
        {
          value: 'laplacian_pe',
          label: 'Laplacian PE: Eigenvector Spatial Frequency Harmonics (Dwivedi 2020, Lim 2022)',
        },
        {
          value: 'rwpe_return',
          label: 'Random Walk PE: Step-k Return Probabilities P^k_ii & Subgraph Motifs',
        },
      ],
      'centrality_profiles',
      { label: 'graph feature modality' },
    ),
    param1: slider(1, 10, 2, {
      step: 1,
      label: 'harmonic index k / probe node ID / walk steps',
    }),
    param2: slider(1, 10, 5, {
      step: 1,
      label: 'damping alpha / sign flip toggle / cycle density',
    }),
  })

  const { featureMode, param1, param2 } = state

  const results = useMemo(() => {
    if (featureMode === 'centrality_profiles') {
      // 10 structural positions: 1=Periphery leaf, 2-4=Chain nodes, 5=Bridge cut-vertex, 6-8=Dense cluster, 9-10=Central star hubs
      const nodeIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Normalized Degree Centrality (%): high on hub (9,10)
      const degreeCentrality = [10, 20, 20, 30, 40, 60, 60, 70, 95, 100]
      // Betweenness Centrality (%): massive peak on Bridge cut-vertex (node 5)
      const betweenness = [0, 8, 14, 25, 98, 35, 40, 42, 75, 80]
      // PageRank Centrality (%) with damping alpha = 0.85
      const pageRank = [12, 18, 22, 28, 55, 62, 65, 72, 90, 96]

      const currNode = Math.min(10, Math.max(1, param1))
      const currDeg = degreeCentrality[currNode - 1]
      const currBet = betweenness[currNode - 1]
      const currPr = pageRank[currNode - 1]
      const isBridge = currNode === 5

      return {
        x: nodeIds,
        y: betweenness,
        baseline: degreeCentrality,
        stat1: `Node #${currNode}: ${isBridge ? 'Bottleneck Bridge' : currNode >= 9 ? 'Hub Core' : 'Cluster Member'}`,
        stat2: `Betweenness: ${currBet}% | Degree: ${currDeg}%`,
        stat3: `PageRank Score: ${currPr}%`,
        warning: isBridge
          ? 'Node 5 is a critical articulation point (cut-vertex); betweenness peaks near 100% while local degree is moderate, identifying a bottleneck bridging two large graph communities'
          : 'Hub nodes exhibit dominant degree and PageRank centrality; peripheral leaves receive low message-passing influence',
        axisXLabel: 'Graph Structural Node Topology Position (1=Leaf, 5=Bridge, 10=Hub)',
        axisYLabel: 'Relative Centrality Metric Value (%)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (featureMode === 'laplacian_pe') {
      // 1D Path graph of 20 nodes: 1 to 20
      const nodeIndices = Array.from({ length: 20 }, (_, i) => i + 1)
      const harmonicK = Math.min(6, Math.max(1, param1))
      const signFlip = param2 > 5 ? -1 : 1

      // Eigenvector phi_k(i) = sqrt(2/N) * cos( (i - 0.5) * pi * k / N )
      const eigenValues = nodeIndices.map((i) => {
        const phi = Math.sqrt(2 / 20) * Math.cos(((i - 0.5) * Math.PI * harmonicK) / 20)
        return phi * signFlip * 100 // scaled
      })

      // Fundamental dipole wave baseline (k=1)
      const dipoleBaseline = nodeIndices.map((i) => Math.sqrt(2 / 20) * Math.cos(((i - 0.5) * Math.PI * 1) / 20) * 100)

      return {
        x: nodeIndices,
        y: eigenValues,
        baseline: dipoleBaseline,
        stat1: `LapPE Harmonic: k = ${harmonicK}`,
        stat2: `Sign Orientation: ${signFlip > 0 ? '+1 (Standard)' : '-1 (Flipped Sign)'}`,
        stat3: `Eigenvalue λ_${harmonicK}: ${(0.05 * harmonicK * harmonicK).toFixed(3)}`,
        warning:
          'Laplacian eigenvectors provide smooth global coordinate frames, but exhibit sign ambiguity (phi_k = -phi_k); models require SignNet or basis-invariant layers to prevent arbitrary coordinate flips',
        axisXLabel: 'Node Position Along Graph Manifold (1 to 20)',
        axisYLabel: 'Laplacian Eigenvector Amplitude phi_k(v) (Scaled)',
        xRange: [0.5, 20.5] as [number, number],
        yRange: [-40, 40] as [number, number],
      }
    } else {
      // Walk steps k from 1 to 16
      const walkSteps = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
      // Return probability P^k_ii for Triangle Node (has 3-cycle return, non-bipartite)
      const triangleReturn = walkSteps.map((k) => {
        if (k === 1) return 0
        if (k === 2) return 50
        if (k === 3) return 25 // 3-cycle motif signature!
        return Math.max(12, 33 + Math.sin(k * 1.5) * 12 * Math.exp(-k * 0.15))
      })

      // Bipartite Grid Node (cannot return on odd steps!)
      const bipartiteReturn = walkSteps.map((k) => (k % 2 === 0 ? Math.max(10, 50 * Math.pow(0.8, k / 2)) : 0))

      const currStep = Math.min(16, Math.max(1, param1 * 2))
      const isOdd = currStep % 2 !== 0

      return {
        x: walkSteps,
        y: triangleReturn,
        baseline: bipartiteReturn,
        stat1: `Walk Step: k = ${currStep}`,
        stat2: `Triangle Node P^k: ${triangleReturn[Math.min(triangleReturn.length - 1, Math.floor(currStep / 2))].toFixed(1)}%`,
        stat3: `Bipartite Node P^k: ${isOdd ? '0.0% (Bipartite Odd Zero)' : 'Non-zero'}`,
        warning: isOdd
          ? 'Odd walk steps k=3,5 yield strictly zero return on bipartite graphs; non-zero odd returns directly measure local triangle and clique motif densities'
          : 'Even walk steps capture local back-and-forth diffusion rates, encoding local neighborhood volume and node degree',
        axisXLabel: 'Random Walk Horizon Steps (k in P^k_ii)',
        axisYLabel: 'Random Walk Return Probability P^k_ii (%)',
        xRange: [0.5, 16.5] as [number, number],
        yRange: [-2, 60] as [number, number],
      }
    }
  }, [featureMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Graph Features: Centrality Metrics, Laplacian Spectral Harmonics, and Random Walk Motifs"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Feature Focus" value={results.stat1} />
          <Readout label="Primary Structural Signal" value={results.stat2} />
          <Readout label="Spectral / Topological Value" value={results.stat3} />
          <Readout label="Representational Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Graph Structural Invariants and Positional Encodings">
        <Curve
          name={
            featureMode === 'centrality_profiles'
              ? 'Degree Centrality Distribution (%)'
              : featureMode === 'laplacian_pe'
                ? 'Fundamental Dipole Harmonic (k=1)'
                : 'Bipartite Grid Node Return Probability (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            featureMode === 'centrality_profiles'
              ? 'Betweenness Centrality Bottleneck Peak (%)'
              : featureMode === 'laplacian_pe'
                ? 'Selected Eigenvector Wave Harmonic phi_k'
                : 'Triangle Clique Node Return Probability (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Node / Step Evaluations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
