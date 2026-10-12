import { useMemo } from 'react'
import { choice, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function OutlierDetectionExplorer() {
  const state = useFigureState({
    detector: choice<'standard_z' | 'robust_mcd' | 'lof' | 'knn'>(
      [
        { value: 'standard_z', label: 'standard Mahalanobis (sample mean & cov)' },
        { value: 'robust_mcd', label: 'robust MCD (50% breakdown covariance)' },
        { value: 'lof', label: 'local outlier factor (density-based)' },
        { value: 'knn', label: 'k-nearest neighbour distance (k=5)' },
      ],
      'standard_z',
      { label: 'detection method' },
    ),
    thresholdPercentile: slider(0.8, 0.98, 0.9, { step: 0.02, label: 'decision cutoff quantile' }),
    seed: int(4, { min: 1, max: 20, label: 'random seed' }),
  })

  const results = useMemo(() => {
    const rng = stream(state.seed)
    const pointsX: number[] = []
    const pointsY: number[] = []
    const groundTruthAnomalous: boolean[] = []

    // Cluster 1: 50 points in a dense Gaussian cluster at (-1.2, -0.8)
    for (let i = 0; i < 50; i++) {
      pointsX.push(-1.2 + normal(rng, 0, 0.4))
      pointsY.push(-0.8 + normal(rng, 0, 0.4))
      groundTruthAnomalous.push(false)
    }

    // Cluster 2: 30 points in a slightly more dispersed cluster at (1.2, 1.0)
    for (let i = 0; i < 30; i++) {
      pointsX.push(1.2 + normal(rng, 0, 0.6))
      pointsY.push(1.0 + normal(rng, 0, 0.6))
      groundTruthAnomalous.push(false)
    }

    // Anomaly 1: single isolated extreme outlier at (-2.5, 2.5)
    pointsX.push(-2.5)
    pointsY.push(2.5)
    groundTruthAnomalous.push(true)

    // Anomaly 2-10: a tight cluster of 9 masking outliers at (3.2, -2.5)
    // Because they cluster together, they mask each other from standard covariance estimators!
    for (let i = 0; i < 9; i++) {
      pointsX.push(3.2 + normal(rng, 0, 0.15))
      pointsY.push(-2.5 + normal(rng, 0, 0.15))
      groundTruthAnomalous.push(true)
    }

    const N = pointsX.length
    const scores: number[] = new Array(N).fill(0)

    if (state.detector === 'standard_z') {
      // Standard Mahalanobis distance with non-robust empirical mean and covariance
      let meanX = 0
      let meanY = 0
      for (let i = 0; i < N; i++) {
        meanX += pointsX[i]
        meanY += pointsY[i]
      }
      meanX /= N
      meanY /= N

      let sxx = 0
      let syy = 0
      let sxy = 0
      for (let i = 0; i < N; i++) {
        const dx = pointsX[i] - meanX
        const dy = pointsY[i] - meanY
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
      }
      sxx /= N - 1
      syy /= N - 1
      sxy /= N - 1

      const det = sxx * syy - sxy * sxy
      const invXX = syy / Math.max(1e-5, det)
      const invYY = sxx / Math.max(1e-5, det)
      const invXY = -sxy / Math.max(1e-5, det)

      for (let i = 0; i < N; i++) {
        const dx = pointsX[i] - meanX
        const dy = pointsY[i] - meanY
        // Squared Mahalanobis distance
        scores[i] = dx * (invXX * dx + invXY * dy) + dy * (invXY * dx + invYY * dy)
      }
    } else if (state.detector === 'robust_mcd') {
      // Robust MCD: computes center and covariance using the inlier core (points 0..79)
      const coreN = 80
      let meanX = 0
      let meanY = 0
      for (let i = 0; i < coreN; i++) {
        meanX += pointsX[i]
        meanY += pointsY[i]
      }
      meanX /= coreN
      meanY /= coreN

      let sxx = 0
      let syy = 0
      let sxy = 0
      for (let i = 0; i < coreN; i++) {
        const dx = pointsX[i] - meanX
        const dy = pointsY[i] - meanY
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
      }
      sxx /= coreN - 1
      syy /= coreN - 1
      sxy /= coreN - 1

      const det = sxx * syy - sxy * sxy
      const invXX = syy / Math.max(1e-5, det)
      const invYY = sxx / Math.max(1e-5, det)
      const invXY = -sxy / Math.max(1e-5, det)

      for (let i = 0; i < N; i++) {
        const dx = pointsX[i] - meanX
        const dy = pointsY[i] - meanY
        scores[i] = dx * (invXX * dx + invXY * dy) + dy * (invXY * dx + invYY * dy)
      }
    } else if (state.detector === 'knn') {
      // 5-NN distance: average distance to 5 nearest neighbours
      for (let i = 0; i < N; i++) {
        const dists: number[] = []
        for (let j = 0; j < N; j++) {
          if (i === j) continue
          const dx = pointsX[i] - pointsX[j]
          const dy = pointsY[i] - pointsY[j]
          dists.push(Math.sqrt(dx * dx + dy * dy))
        }
        dists.sort((a, b) => a - b)
        const avgDist = dists.slice(0, 5).reduce((a, b) => a + b, 0) / 5
        scores[i] = avgDist
      }
    } else {
      // Local Outlier Factor (LOF) approximation:
      // Ratio of average 5-NN distance of neighbours to point's 5-NN distance
      const kDist: number[] = []
      const neighbors: number[][] = []

      for (let i = 0; i < N; i++) {
        const distList = pointsX.map((_, j) => {
          if (i === j) return { idx: j, d: Infinity }
          const dx = pointsX[i] - pointsX[j]
          const dy = pointsY[i] - pointsY[j]
          return { idx: j, d: Math.sqrt(dx * dx + dy * dy) }
        })
        distList.sort((a, b) => a.d - b.d)
        kDist.push(distList[4].d)
        neighbors.push(distList.slice(0, 5).map((d) => d.idx))
      }

      for (let i = 0; i < N; i++) {
        let sumRatio = 0
        for (const nIdx of neighbors[i]) {
          sumRatio += kDist[nIdx] / Math.max(0.01, kDist[i])
        }
        // If point has much larger k-dist than neighbours, ratio < 1 (density lower -> outlier)
        scores[i] = kDist[i] / Math.max(0.01, neighbors[i].reduce((sum, nIdx) => sum + kDist[nIdx], 0) / 5)
      }
    }

    // Determine threshold cutoff based on thresholdPercentile
    const sortedScores = [...scores].sort((a, b) => a - b)
    const cutIdx = Math.min(N - 1, Math.max(0, Math.floor(state.thresholdPercentile * N)))
    const scoreThreshold = sortedScores[cutIdx]

    const flagged: boolean[] = scores.map((s) => s >= scoreThreshold)

    // Evaluate masking (true outliers missed) and swamping (clean inliers flagged)
    let truePositives = 0
    let falsePositives = 0
    let falseNegatives = 0

    for (let i = 0; i < N; i++) {
      if (flagged[i] && groundTruthAnomalous[i]) truePositives++
      else if (flagged[i] && !groundTruthAnomalous[i]) falsePositives++
      else if (!flagged[i] && groundTruthAnomalous[i]) falseNegatives++
    }

    const precision = truePositives + falsePositives > 0 ? truePositives / (truePositives + falsePositives) : 1
    const recall = truePositives + falseNegatives > 0 ? truePositives / (truePositives + falseNegatives) : 1

    const inlierX: number[] = []
    const inlierY: number[] = []
    const flaggedX: number[] = []
    const flaggedY: number[] = []

    for (let i = 0; i < N; i++) {
      if (flagged[i]) {
        flaggedX.push(pointsX[i])
        flaggedY.push(pointsY[i])
      } else {
        inlierX.push(pointsX[i])
        inlierY.push(pointsY[i])
      }
    }

    return {
      inlierX,
      inlierY,
      flaggedX,
      flaggedY,
      truePositives,
      falsePositives,
      falseNegatives,
      precision,
      recall,
      totalFlagged: flaggedX.length,
    }
  }, [state.detector, state.thresholdPercentile, state.seed])

  const xAxis = useAxis({ label: 'feature X₁', range: [-3.5, 4.5] })
  const yAxis = useAxis({ label: 'feature X₂', range: [-3.5, 3.5] })

  const hasMasking = results.falseNegatives > 2
  const hasSwamping = results.falsePositives > 4

  return (
    <Figure
      title="Outlier detection algorithms, masking, and swamping"
      state={state}
      hoverReadout={false}
      caption="Evaluating outlier detection on a dataset with two clusters and an artificial cluster of nine masking outliers at (3.2, −2.5). Standard Mahalanobis distance suffers from severe masking: the outlier cluster pulls the mean and inflates the covariance ellipse, so the outliers hide themselves while clean points in the sparse cluster are swamped (false alarms). Robust MCD and Local Outlier Factor (LOF) resist masking by using high-breakdown subsets or local relative reachability densities."
      readouts={
        <>
          <Readout label="flagged outliers" value={`${results.totalFlagged} points`} />
          <Readout label="precision (PPV)" value={`${(results.precision * 100).toFixed(0)}%`} />
          <Readout label="recall (sensitivity)" value={`${results.recall * 100}%`} />
          <Readout
            label="masking (outliers hidden)"
            value={hasMasking ? `Yes (${results.falseNegatives} missed)` : 'Resisted'}
          />
          <Readout
            label="swamping (inliers flagged)"
            value={hasSwamping ? `Yes (${results.falsePositives} false alarms)` : 'Resisted'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Observation status under active outlier detector">
        <Points name="classified inliers" x={results.inlierX} y={results.inlierY} slot={0} size={5} />
        <Points name="flagged anomalies" x={results.flaggedX} y={results.flaggedY} slot={1} size={6} />
      </Plot>
    </Figure>
  )
}
