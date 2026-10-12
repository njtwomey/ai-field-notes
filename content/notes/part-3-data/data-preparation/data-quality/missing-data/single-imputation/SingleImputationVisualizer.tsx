import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const SAMPLE_SIZE = 140

export function SingleImputationVisualizer() {
  const state = useFigureState({
    method: choice<'mean' | 'deterministic' | 'stochastic' | 'knn'>(
      [
        { value: 'mean', label: 'unconditional mean' },
        { value: 'deterministic', label: 'deterministic regression' },
        { value: 'stochastic', label: 'stochastic regression' },
        { value: 'knn', label: 'k-nearest neighbour (k=3)' },
      ],
      'mean',
      { label: 'imputation method' },
    ),
    missingRate: slider(0.1, 0.6, 0.35, { step: 0.05, label: 'missingness fraction' }),
    seed: int(5, { min: 1, max: 20, label: 'sample seed' }),
  })

  const results = useMemo(() => {
    const rng = stream(state.seed)
    const rawX: number[] = []
    const rawY: number[] = []

    // Generate correlated bivariate Gaussian: X ~ N(0, 1), Y = 0.8 * X + eps, eps ~ N(0, 0.6^2)
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const x = normal(rng, 0, 1)
      const eps = normal(rng, 0, 0.6)
      const y = 0.8 * x + eps
      rawX.push(x)
      rawY.push(y)
    }

    // Determine missingness in Y (MCAR coin flips)
    const isObserved: boolean[] = []
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      isObserved.push(uniform(rng) >= state.missingRate)
    }

    const obsIndices: number[] = []
    const misIndices: number[] = []
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      if (isObserved[i]) obsIndices.push(i)
      else misIndices.push(i)
    }

    // Calculate stats on observed subset
    const nObs = obsIndices.length
    const meanObsX = nObs > 0 ? obsIndices.reduce((sum, idx) => sum + rawX[idx], 0) / nObs : 0
    const meanObsY = nObs > 0 ? obsIndices.reduce((sum, idx) => sum + rawY[idx], 0) / nObs : 0

    let sxy = 0
    let sxx = 0
    let syy = 0
    for (const idx of obsIndices) {
      const dx = rawX[idx] - meanObsX
      const dy = rawY[idx] - meanObsY
      sxy += dx * dy
      sxx += dx * dx
      syy += dy * dy
    }
    const beta1 = sxx > 1e-6 ? sxy / sxx : 0.8
    const beta0 = meanObsY - beta1 * meanObsX
    const residualVar = Math.max(0.01, (syy - beta1 * sxy) / Math.max(1, nObs - 2))
    const residualStd = Math.sqrt(residualVar)

    // Compute imputed values
    const imputedY = [...rawY]

    for (const idx of misIndices) {
      const x = rawX[idx]
      if (state.method === 'mean') {
        imputedY[idx] = meanObsY
      } else if (state.method === 'deterministic') {
        imputedY[idx] = beta0 + beta1 * x
      } else if (state.method === 'stochastic') {
        imputedY[idx] = beta0 + beta1 * x + normal(rng, 0, residualStd)
      } else {
        // k-NN with k = 3
        const distances = obsIndices.map((obsIdx) => ({
          idx: obsIdx,
          dist: Math.abs(rawX[obsIdx] - x),
        }))
        distances.sort((a, b) => a.dist - b.dist)
        const kNeighbors = distances.slice(0, 3)
        const avgNeighborY = kNeighbors.reduce((sum, d) => sum + rawY[d.idx], 0) / kNeighbors.length
        imputedY[idx] = avgNeighborY
      }
    }

    // Full imputed dataset statistics
    const meanImpY = imputedY.reduce((a, b) => a + b, 0) / SAMPLE_SIZE
    let varImpY = 0
    let covImpXY = 0
    let varImpX = 0
    const meanX = rawX.reduce((a, b) => a + b, 0) / SAMPLE_SIZE

    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const dx = rawX[i] - meanX
      const dy = imputedY[i] - meanImpY
      varImpX += dx * dx
      varImpY += dy * dy
      covImpXY += dx * dy
    }
    varImpY /= SAMPLE_SIZE - 1
    varImpX /= SAMPLE_SIZE - 1
    covImpXY /= SAMPLE_SIZE - 1
    const corrImp = varImpX > 0 && varImpY > 0 ? covImpXY / Math.sqrt(varImpX * varImpY) : 0

    // True dataset statistics
    const trueMeanY = rawY.reduce((a, b) => a + b, 0) / SAMPLE_SIZE
    let trueVarY = 0
    let trueCovXY = 0
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const dx = rawX[i] - meanX
      const dy = rawY[i] - trueMeanY
      trueVarY += dy * dy
      trueCovXY += dx * dy
    }
    trueVarY /= SAMPLE_SIZE - 1
    trueCovXY /= SAMPLE_SIZE - 1
    const trueCorr = trueVarY > 0 && varImpX > 0 ? trueCovXY / Math.sqrt(varImpX * trueVarY) : 0

    const obsX = obsIndices.map((i) => rawX[i])
    const obsY = obsIndices.map((i) => rawY[i])
    const misX = misIndices.map((i) => rawX[i])
    const impYValues = misIndices.map((i) => imputedY[i])

    // Regression lines for comparison
    const curveX = [-3, -2, -1, 0, 1, 2, 3]
    const regressionLine = curveX.map((x) => beta0 + beta1 * x)
    const meanLine = curveX.map(() => meanObsY)

    return {
      obsX,
      obsY,
      misX,
      impYValues,
      varImpY,
      trueVarY,
      corrImp,
      trueCorr,
      nObs,
      nMis: misIndices.length,
      curveX,
      regressionLine,
      meanLine,
    }
  }, [state.method, state.missingRate, state.seed])

  const xAxis = useAxis({ label: 'feature X (observed)', range: [-3, 3] })
  const yAxis = useAxis({ label: 'target Y (observed or imputed)', range: [-3, 3] })

  const varRatio = results.trueVarY > 0 ? (results.varImpY / results.trueVarY) * 100 : 100

  return (
    <Figure
      title="Variance collapse and correlation distortion under single imputation"
      state={state}
      hoverReadout={false}
      caption="Unconditional mean imputation collapses imputed points onto a horizontal spike at the sample mean, artificially shrinking sample variance and diluting correlation. Deterministic regression places imputations directly on the fitted line, eliminating residual error and exaggerating correlation. Stochastic regression restores realistic dispersion by adding residual Gaussian noise. In all cases, treating the single imputed dataset as real data causes downstream standard errors to be severely underestimated."
      readouts={
        <>
          <Readout
            label="imputed variance Var(Y)"
            value={`${results.varImpY.toFixed(2)} (${varRatio.toFixed(0)}% of true)`}
          />
          <Readout
            label="correlation Corr(X, Y)"
            value={`${results.corrImp.toFixed(2)} (true: ${results.trueCorr.toFixed(2)})`}
          />
          <Readout label="complete cases" value={`${results.nObs} / ${SAMPLE_SIZE}`} />
          <Readout label="imputed values" value={`${results.nMis}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Single imputation points and reference geometry">
        <Points name="observed data points" x={results.obsX} y={results.obsY} slot={0} size={5} />
        <Points name="imputed values (synthetic)" x={results.misX} y={results.impYValues} slot={1} size={5} />
        {state.method === 'mean' && (
          <Curve name="unconditional mean level" x={results.curveX} y={results.meanLine} slot={1} dashed width={2} />
        )}
        {(state.method === 'deterministic' || state.method === 'stochastic') && (
          <Curve
            name="fitted regression line"
            x={results.curveX}
            y={results.regressionLine}
            slot={2}
            dashed
            width={2}
          />
        )}
      </Plot>
    </Figure>
  )
}
