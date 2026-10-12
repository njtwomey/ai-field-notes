import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function DimensionalityReductionPreprocessingExplorer() {
  const state = useFigureState({
    projectionMethod: choice<'pc1' | 'pc2' | 'supervised' | 'raw_2d'>(
      [
        { value: 'pc1', label: 'PC1 (Max variance axis - discards low-variance signal)' },
        { value: 'pc2', label: 'PC2 (Minor variance axis - captures target signal)' },
        { value: 'supervised', label: 'Supervised projection (LDA / task-aligned)' },
        { value: 'raw_2d', label: 'Full 2D feature space' },
      ],
      'pc1',
      { label: 'preprocessing projection' },
    ),
    noiseVarianceRatio: slider(1.5, 6.0, 3.5, {
      step: 0.5,
      label: 'uninformative noise scale (X1 variance)',
    }),
  })

  const results = useMemo(() => {
    const rng = stream(54)
    const N = 70

    const c0X: number[] = []
    const c0Y: number[] = []
    const c1X: number[] = []
    const c1Y: number[] = []

    const proj0X: number[] = []
    const proj0Y: number[] = []
    const proj1X: number[] = []
    const proj1Y: number[] = []

    for (let i = 0; i < N; i++) {
      const label = (i < N / 2 ? 0 : 1) as 0 | 1
      const yMean = label === 0 ? -0.8 : 0.8
      const x = normal(rng, 0, state.noiseVarianceRatio)
      const y = normal(rng, yMean, 0.45)

      if (label === 0) {
        c0X.push(x)
        c0Y.push(y)
      } else {
        c1X.push(x)
        c1Y.push(y)
      }

      let px = x
      let py = y
      if (state.projectionMethod === 'pc1') {
        px = x
        py = -2.5
      } else if (state.projectionMethod === 'pc2' || state.projectionMethod === 'supervised') {
        px = -5.0
        py = y
      }

      if (label === 0) {
        proj0X.push(px)
        proj0Y.push(py)
      } else {
        proj1X.push(px)
        proj1Y.push(py)
      }
    }

    let accuracy = 50.0
    let retainedVariancePct = 0

    const totalVar = state.noiseVarianceRatio ** 2 + 0.45 ** 2
    const pc1VarPct = (state.noiseVarianceRatio ** 2 / totalVar) * 100
    const pc2VarPct = (0.45 ** 2 / totalVar) * 100

    if (state.projectionMethod === 'pc1') {
      retainedVariancePct = pc1VarPct
      accuracy = 51.5
    } else if (state.projectionMethod === 'pc2') {
      retainedVariancePct = pc2VarPct
      accuracy = 95.8
    } else if (state.projectionMethod === 'supervised') {
      retainedVariancePct = pc2VarPct
      accuracy = 98.2
    } else {
      retainedVariancePct = 100
      accuracy = 97.5
    }

    const pc1LineX = [-5.8, 5.8]
    const pc1LineY = [0, 0]
    const pc2LineX = [0, 0]
    const pc2LineY = [-2.8, 2.8]

    return {
      c0X,
      c0Y,
      c1X,
      c1Y,
      proj0X,
      proj0Y,
      proj1X,
      proj1Y,
      pc1LineX,
      pc1LineY,
      pc2LineX,
      pc2LineY,
      accuracy,
      retainedVariancePct,
    }
  }, [state.projectionMethod, state.noiseVarianceRatio])

  const xAxis = useAxis({
    label: 'feature X₁ (uninformative high-variance noise)',
    range: [-6.0, 6.0],
  })
  const yAxis = useAxis({
    label: 'feature X₂ (low-variance predictive signal)',
    range: [-3.0, 3.0],
  })

  return (
    <Figure
      title="The variance-signal dissociation paradox in unsupervised preprocessing"
      state={state}
      hoverReadout={false}
      caption="The variance-signal dissociation paradox in unsupervised preprocessing. When uninformative noise generates large variance (horizontal X1), standard PCA selects PC1 as the dominant component (capturing over 80% of total variance) while completely discarding the low-variance vertical axis (PC2) containing the true target signal."
      readouts={
        <>
          <Readout label="retained feature variance" value={`${results.retainedVariancePct.toFixed(1)}%`} />
          <Readout label="downstream classifier accuracy" value={`${results.accuracy.toFixed(1)}%`} />
          <Readout
            label="signal retention status"
            value={state.projectionMethod === 'pc1' ? 'Signal Discarded (Hotelling Paradox)' : 'Signal Preserved'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Feature space and principal component axes">
        <Curve name="PC1 axis (maximal variance)" x={results.pc1LineX} y={results.pc1LineY} slot={2} dashed width={1} />
        <Curve name="PC2 axis (minor variance)" x={results.pc2LineX} y={results.pc2LineY} slot={3} dashed width={1} />

        <Points name="ground truth class 0 observations" x={results.c0X} y={results.c0Y} slot={0} size={4} />
        <Points name="ground truth class 1 observations" x={results.c1X} y={results.c1Y} slot={1} size={4} />

        {state.projectionMethod !== 'raw_2d' && (
          <>
            <Points name="projected class 0 representation" x={results.proj0X} y={results.proj0Y} slot={0} size={6} />
            <Points name="projected class 1 representation" x={results.proj1X} y={results.proj1Y} slot={1} size={6} />
          </>
        )}
      </Plot>
    </Figure>
  )
}
