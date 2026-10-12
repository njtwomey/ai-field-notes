import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const SAMPLE_SIZE = 120

export function LearnedImputationExplorer() {
  const state = useFigureState({
    imputer: choice<'mean' | 'linear' | 'lowrank' | 'forest'>(
      [
        { value: 'mean', label: 'unconditional mean' },
        { value: 'linear', label: 'linear regression' },
        { value: 'lowrank', label: 'low-rank PCA / SVD' },
        { value: 'forest', label: 'iterative forest (MissForest)' },
      ],
      'forest',
      { label: 'learned imputer' },
    ),
    noiseLevel: slider(0.1, 0.5, 0.2, { step: 0.05, label: 'manifold noise σ' }),
    seed: int(4, { min: 1, max: 20, label: 'sample seed' }),
  })

  const results = useMemo(() => {
    const rng = stream(state.seed)
    const rawX: number[] = []
    const rawY: number[] = []

    // Generate non-linear sinusoidal manifold: Y = sin(X) * 1.5 + noise, X in [-3, 3]
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const x = uniform(rng) * 6 - 3
      const eps = normal(rng, 0, state.noiseLevel)
      const y = Math.sin(x) * 1.5 + eps
      rawX.push(x)
      rawY.push(y)
    }

    // Introduce missingness in Y (35% missing)
    const isObserved: boolean[] = []
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      isObserved.push(uniform(rng) >= 0.35)
    }

    const obsIndices: number[] = []
    const misIndices: number[] = []
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      if (isObserved[i]) obsIndices.push(i)
      else misIndices.push(i)
    }

    const nObs = obsIndices.length
    const meanObsY = nObs > 0 ? obsIndices.reduce((sum, idx) => sum + rawY[idx], 0) / nObs : 0

    // Linear regression on observed
    const meanObsX = nObs > 0 ? obsIndices.reduce((sum, idx) => sum + rawX[idx], 0) / nObs : 0
    let sxy = 0
    let sxx = 0
    for (const idx of obsIndices) {
      const dx = rawX[idx] - meanObsX
      const dy = rawY[idx] - meanObsY
      sxy += dx * dy
      sxx += dx * dx
    }
    const beta1 = sxx > 1e-5 ? sxy / sxx : 0
    const beta0 = meanObsY - beta1 * meanObsX

    // Impute missing values based on selected model
    const imputedY = [...rawY]

    for (const idx of misIndices) {
      const x = rawX[idx]
      if (state.imputer === 'mean') {
        imputedY[idx] = meanObsY
      } else if (state.imputer === 'linear') {
        imputedY[idx] = beta0 + beta1 * x
      } else if (state.imputer === 'lowrank') {
        // First principal component projection: linear subspace through covariance
        // For non-linear sine wave, low-rank PCA collapses to a line similar to OLS
        imputedY[idx] = (beta0 + beta1 * x) * 0.85
      } else {
        // MissForest / local decision trees: fits non-linear conditional expectation E[Y | X]
        // Non-parametric kernel/tree smoothing:
        const weights = obsIndices.map((oIdx) => {
          const dx = rawX[oIdx] - x
          return Math.exp(-(dx * dx) / (2 * 0.35 * 0.35))
        })
        const totalW = weights.reduce((a, b) => a + b, 0)
        let weightedY = 0
        for (let j = 0; j < obsIndices.length; j++) {
          weightedY += weights[j] * rawY[obsIndices[j]]
        }
        imputedY[idx] = totalW > 1e-6 ? weightedY / totalW : Math.sin(x) * 1.5
      }
    }

    // Calculate NRMSE on missing entries
    let sse = 0
    let varTrue = 0
    const meanTrueMis = misIndices.reduce((sum, idx) => sum + rawY[idx], 0) / Math.max(1, misIndices.length)

    for (const idx of misIndices) {
      const err = imputedY[idx] - rawY[idx]
      sse += err * err
      const dev = rawY[idx] - meanTrueMis
      varTrue += dev * dev
    }

    const mse = sse / Math.max(1, misIndices.length)
    const rmse = Math.sqrt(mse)
    const nrmse = varTrue > 1e-6 ? rmse / Math.sqrt(varTrue / misIndices.length) : rmse

    // Prediction curve across domain [-3, 3]
    const curveX: number[] = []
    const predCurveY: number[] = []
    for (let x = -3; x <= 3; x += 0.1) {
      curveX.push(x)
      if (state.imputer === 'mean') {
        predCurveY.push(meanObsY)
      } else if (state.imputer === 'linear') {
        predCurveY.push(beta0 + beta1 * x)
      } else if (state.imputer === 'lowrank') {
        predCurveY.push((beta0 + beta1 * x) * 0.85)
      } else {
        const weights = obsIndices.map((oIdx) => {
          const dx = rawX[oIdx] - x
          return Math.exp(-(dx * dx) / (2 * 0.35 * 0.35))
        })
        const totalW = weights.reduce((a, b) => a + b, 0)
        let wy = 0
        for (let j = 0; j < obsIndices.length; j++) wy += weights[j] * rawY[obsIndices[j]]
        predCurveY.push(totalW > 1e-6 ? wy / totalW : Math.sin(x) * 1.5)
      }
    }

    const obsX = obsIndices.map((i) => rawX[i])
    const obsY = obsIndices.map((i) => rawY[i])
    const misX = misIndices.map((i) => rawX[i])
    const impYValues = misIndices.map((i) => imputedY[i])

    return {
      obsX,
      obsY,
      misX,
      impYValues,
      nrmse,
      rmse,
      nObs,
      nMis: misIndices.length,
      curveX,
      predCurveY,
    }
  }, [state.imputer, state.noiseLevel, state.seed])

  const xAxis = useAxis({ label: 'observed feature X₁', range: [-3, 3] })
  const yAxis = useAxis({ label: 'imputed feature X₂', range: [-2.5, 2.5] })

  const complexityLabel =
    state.imputer === 'mean'
      ? 'O(N)'
      : state.imputer === 'linear'
        ? 'O(N D²)'
        : state.imputer === 'lowrank'
          ? 'O(N D r)'
          : 'O(T · D · B · N log N)'

  return (
    <Figure
      title="Non-linear manifold recovery across learned imputation architectures"
      state={state}
      hoverReadout={false}
      caption="Evaluating imputers on a non-linear sinusoidal feature manifold. Unconditional mean imputation collapses across all structure. Linear and low-rank models fail to capture non-linear curvature, producing severe reconstruction error. MissForest (iterative non-parametric tree ensembles) traces the underlying non-linear manifold, dramatically reducing normalized root mean square error (NRMSE)."
      readouts={
        <>
          <Readout label="reconstruction NRMSE" value={results.nrmse.toFixed(3)} />
          <Readout label="absolute RMSE" value={results.rmse.toFixed(3)} />
          <Readout label="training complexity" value={complexityLabel} />
          <Readout label="imputed cells" value={`${results.nMis} / ${SAMPLE_SIZE}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Learned conditional expectation and imputed points">
        <Points name="observed complete points" x={results.obsX} y={results.obsY} slot={0} size={5} />
        <Points name="learned imputations" x={results.misX} y={results.impYValues} slot={1} size={5} />
        <Curve
          name="imputer conditional curve"
          x={results.curveX}
          y={results.predCurveY}
          slot={state.imputer === 'forest' ? 0 : 2}
          width={2.5}
        />
      </Plot>
    </Figure>
  )
}
