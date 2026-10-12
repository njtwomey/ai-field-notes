import { useMemo } from 'react'
import { choice, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ClusteringForExplorationVisualizer() {
  const state = useFigureState({
    samplingStrategy: choice<'random' | 'cluster_stratified' | 'hdbscan_noise'>(
      [
        { value: 'random', label: 'Uniform Random Sampling (Misses Rare Clusters)' },
        { value: 'cluster_stratified', label: 'Cluster-Then-Sample (Guaranteed Subgroup Coverage)' },
        { value: 'hdbscan_noise', label: 'Density-Based Noise Isolation (HDBSCAN)' },
      ],
      'cluster_stratified',
      { label: 'exploratory sampling workflow' },
    ),
    inspectionBudget: slider(1, 8, 4, {
      step: 1,
      label: 'inspected samples per cluster',
    }),
    anomalyPrevalence: slider(0.1, 4.0, 0.8, {
      step: 0.1,
      label: 'rare anomaly cluster prevalence (%)',
    }),
  })

  const { samplingStrategy, inspectionBudget, anomalyPrevalence } = state

  const results = useMemo(() => {
    // 3 main clusters + 1 rare anomaly cluster + noise points
    // C1: Main user traffic (Center [-3, 2], radius ~1.2) - 40 points
    // C2: Mobile checkout (Center [3, 2], radius ~1.0) - 35 points
    // C3: Bot templated loop (Center [0, -3], radius ~0.2 - ultra dense) - 30 points
    // C4: Rare security anomaly (Center [6, -4], radius ~0.4) - 5 points (prevalence ~1%)
    const allX: number[] = []
    const allY: number[] = []
    const sampleX: number[] = []
    const sampleY: number[] = []

    // Synthetic generation
    const clusterDefs = [
      { cx: -3.0, cy: 2.0, r: 1.2, n: 40, id: 0 },
      { cx: 3.0, cy: 2.0, r: 1.0, n: 35, id: 1 },
      { cx: 0.0, cy: -3.0, r: 0.25, n: 30, id: 2 }, // Template
      { cx: 5.5, cy: -3.5, r: 0.4, n: Math.max(3, Math.round(anomalyPrevalence * 5)), id: 3 }, // Rare
    ]

    let foundAnomaly = false

    clusterDefs.forEach((c, cIdx) => {
      for (let i = 0; i < c.n; i++) {
        const angle = (i / c.n) * 2 * Math.PI
        const dist = c.r * (0.3 + 0.7 * Math.sin(i * 5))
        const x = c.cx + dist * Math.cos(angle)
        const y = c.cy + dist * Math.sin(angle)
        allX.push(x)
        allY.push(y)

        if (samplingStrategy === 'cluster_stratified') {
          // Select first 'inspectionBudget' points from each cluster
          if (i < inspectionBudget && i < c.n) {
            sampleX.push(x)
            sampleY.push(y)
            if (cIdx === 3) foundAnomaly = true
          }
        }
      }
    })

    if (samplingStrategy === 'random') {
      // Pick 4 * inspectionBudget random points from all points
      // Total points = 110. Anomaly points = ~4. Probability of picking anomaly in sample size 16:
      const totalBudget = 4 * inspectionBudget
      for (let i = 0; i < totalBudget; i++) {
        // Bias heavily toward majority clusters (indices 0 to 105)
        const idx = Math.min(allX.length - 1, Math.floor((i / totalBudget) * 105))
        sampleX.push(allX[idx])
        sampleY.push(allY[idx])
        if (idx >= 105) foundAnomaly = true
      }
    } else if (samplingStrategy === 'hdbscan_noise') {
      // Show noise points outside clusters
      const noiseX = [-5, 6, -1, 4, -4]
      const noiseY = [-4, 4, 5, -1, -1]
      noiseX.forEach((nx, i) => {
        allX.push(nx)
        allY.push(noiseY[i])
        sampleX.push(nx)
        sampleY.push(noiseY[i])
      })
    }

    // Miss probability calculation for random sampling: (1 - p)^N
    const pAnom = anomalyPrevalence / 100
    const sampleN = 4 * inspectionBudget
    const missProbRandom = Math.pow(1 - pAnom, sampleN) * 100

    return {
      allX,
      allY,
      sampleX,
      sampleY,
      foundAnomaly,
      missProbRandom,
    }
  }, [samplingStrategy, inspectionBudget, anomalyPrevalence])

  const axisX = useAxis({
    label: 'Latent Feature Dimension 1',
    range: [-7, 8],
  })

  const axisY = useAxis({
    label: 'Latent Feature Dimension 2',
    range: [-6, 6],
  })

  return (
    <Figure
      title="Cluster-Then-Sample: Exploratory Subgroup Discovery Explorer"
      state={state}
      hoverReadout={false}
      caption="Comparing uniform random sampling against cluster-stratified sampling. Grouping data into clusters guarantees that rare pathological sub-populations and ultra-dense templated bot traffic are surfaced during human audits."
      readouts={
        <>
          <Readout
            label="Rare Anomaly Discovery"
            value={
              samplingStrategy === 'cluster_stratified'
                ? 'GUARANTEED (Covered in Cluster 4)'
                : samplingStrategy === 'random'
                  ? results.foundAnomaly
                    ? 'Found by random chance'
                    : 'MISSED: Anomaly omitted from review'
                  : 'Isolated as Unclustered Noise'
            }
            color={
              results.foundAnomaly || samplingStrategy === 'cluster_stratified' ? 'var(--blue-600)' : 'var(--red-600)'
            }
          />
          <Readout label="Random Sample Miss Probability" value={`${results.missProbRandom.toFixed(1)}%`} />
          <Readout label="Bot Template Cluster (C3)" value="Ultra-dense radius (r=0.25): Flags automated script" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Exploratory Cluster Space">
        <Points name="Full Dataset Background" x={results.allX} y={results.allY} slot={0} size={5} />
        <Points name="Audited Human Sample" x={results.sampleX} y={results.sampleY} slot={1} size={9} />
      </Plot>
    </Figure>
  )
}
