import { useMemo } from 'react'
import { choice, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function PretrainedEmbeddingsVisualizer() {
  const state = useFigureState({
    dimensionMode: choice<'pca_reduced' | 'full_dense' | 'version_drift'>(
      [
        { value: 'pca_reduced', label: 'PCA / bottleneck reduction (32 dims -> 2D view)' },
        { value: 'full_dense', label: 'unreduced 768-d embedding (high latency for GBDT)' },
        { value: 'version_drift', label: 'silent upstream encoder update (v1.0 -> v2.0 drift)' },
      ],
      'pca_reduced',
      { label: 'pipeline state' },
    ),
    downstreamModel: choice<'ridge' | 'gbdt'>(
      [
        { value: 'ridge', label: 'Linear Ridge Classifier' },
        { value: 'gbdt', label: 'Gradient Boosted Trees (GBDT)' },
      ],
      'gbdt',
      { label: 'downstream model' },
    ),
    latentNoise: slider(0.05, 0.4, 0.15, { step: 0.05, label: 'embedding variance / noise' }),
  })

  const results = useMemo(() => {
    const rng = stream(42)
    const c0X: number[] = []
    const c0Y: number[] = []
    const c1X: number[] = []
    const c1Y: number[] = []
    const c2X: number[] = []
    const c2Y: number[] = []

    const centers: [number, number][] = [
      [-1.5, -0.8],
      [1.4, -0.6],
      [0.1, 1.6],
    ]

    const N_PER_CLASS = 25

    for (let c = 0; c < 3; c++) {
      const [cx, cy] = centers[c]
      for (let i = 0; i < N_PER_CLASS; i++) {
        let x = cx + normal(rng, 0, state.latentNoise)
        let y = cy + normal(rng, 0, state.latentNoise)

        if (state.dimensionMode === 'version_drift') {
          const angle = 0.85
          const rx = x * Math.cos(angle) - y * Math.sin(angle) + 0.4
          const ry = x * Math.sin(angle) + y * Math.cos(angle) - 0.3
          x = rx
          y = ry
        }

        if (c === 0) {
          c0X.push(x)
          c0Y.push(y)
        } else if (c === 1) {
          c1X.push(x)
          c1Y.push(y)
        } else {
          c2X.push(x)
          c2Y.push(y)
        }
      }
    }

    let accuracy = 0
    let latencyMs = 0
    let featureDim = 32

    if (state.dimensionMode === 'full_dense') {
      featureDim = 768
      if (state.downstreamModel === 'ridge') {
        accuracy = 94.2 - state.latentNoise * 20
        latencyMs = 18.5
      } else {
        accuracy = 91.0 - state.latentNoise * 25
        latencyMs = 142.0
      }
    } else if (state.dimensionMode === 'pca_reduced') {
      featureDim = 32
      if (state.downstreamModel === 'ridge') {
        accuracy = 93.8 - state.latentNoise * 18
        latencyMs = 4.2
      } else {
        accuracy = 95.1 - state.latentNoise * 16
        latencyMs = 12.0
      }
    } else {
      featureDim = 32
      accuracy = 38.4
      latencyMs = state.downstreamModel === 'ridge' ? 4.2 : 12.0
    }

    return {
      c0X,
      c0Y,
      c1X,
      c1Y,
      c2X,
      c2Y,
      accuracy: Math.max(0, Math.min(100, accuracy)),
      latencyMs,
      featureDim,
    }
  }, [state.dimensionMode, state.downstreamModel, state.latentNoise])

  const xAxis = useAxis({
    label: 'latent representation dimension 1',
    range: [-3.5, 3.5],
  })
  const yAxis = useAxis({
    label: 'latent representation dimension 2',
    range: [-3.0, 3.5],
  })

  return (
    <Figure
      title="Frozen pretrained embeddings: dimension compression and upstream version drift"
      state={state}
      hoverReadout={false}
      caption="Interactive exploration of frozen pretrained embeddings. Reducing 768-d embeddings to 32 principal components speeds up GBDT split evaluation by ~12x with negligible loss in accuracy. However, upstream encoder version changes without downstream re-training cause immediate coordinate misalignment and catastrophic accuracy collapse."
      readouts={
        <>
          <Readout label="feature dimension" value={`${results.featureDim} dims`} />
          <Readout label="downstream accuracy" value={`${results.accuracy.toFixed(1)}%`} />
          <Readout label="training latency" value={`${results.latencyMs.toFixed(1)} ms`} />
          <Readout
            label="pipeline status"
            value={state.dimensionMode === 'version_drift' ? 'Catastrophic Skew (Drift)' : 'Aligned Representation'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Semantic embedding projections across categories">
        <Points name="semantic cluster A (technology)" x={results.c0X} y={results.c0Y} slot={0} size={5} />
        <Points name="semantic cluster B (finance)" x={results.c1X} y={results.c1Y} slot={1} size={5} />
        <Points name="semantic cluster C (healthcare)" x={results.c2X} y={results.c2Y} slot={2} size={5} />
      </Plot>
    </Figure>
  )
}
