import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { friedman1 } from 'aifn-methods/data/synthetic'
import { polynomialFeatures, standardScaler } from 'aifn-methods/learning/preprocessing'
import { linearRegression } from 'aifn-methods/learning/linear'
import { pipeline } from 'aifn-compute/learning/compose'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'
import { r2Score } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'

const DEGREES = [1, 2, 3]
const FOLDS = 5

/** Polynomial candidates of growing degree for a ridge regression, scored on the training folds and out of fold. */
export function CandidateExplosion() {
  const state = useFigureState({
    n: int(200, { ge: 60, le: 1500, suggestions: [100, 200, 500, 1000], label: 'rows' }),
    d: int(10, { ge: 5, le: 15, suggestions: [5, 10, 15], label: 'inputs (5 informative)' }),
    l2: float(1, { ge: 0, le: 1000, suggestions: [0, 1, 10, 100], label: 'ridge penalty' }),
    noise: float(1, { ge: 0, le: 5, suggestions: [0.5, 1, 2], label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { n, d, l2, noise, seed } = state
  const scores = useMemo(() => {
    const data = friedman1(stream(`automated-feature-engineering/${seed}`), { n, d, noise })
    return DEGREES.map((degree) => {
      const model = pipeline(
        standardScaler(),
        polynomialFeatures({ degree }),
        standardScaler(),
        linearRegression({ l2 }),
      )
      const cv = crossValidate(model, { ...data, y: data.y! }, kFold({ k: FOLDS, shuffle: true }), [r2Score], {
        stream: stream(`automated-feature-engineering/cv/${seed}`),
        trainMetrics: true,
      })
      const train = cv.folds.reduce((s, f) => s + (f.trainMetrics?.r2Score ?? NaN), 0) / cv.folds.length
      // The candidates the expansion generates: its output columns without the constant.
      const count = polynomialFeatures({ degree, includeBias: false }).fit({ x: data.x }).featureNames.length
      return { degree, count, train, test: cv.mean.r2Score }
    })
  }, [n, d, l2, noise, seed])
  const labels = scores.map((s) => `degree ≤ ${s.degree}: ${s.count} features`)
  const x = useAxis({ label: 'candidate set', categories: labels })
  const y = useAxis({ label: 'R²', range: [undefined, 1] })
  const positions = DEGREES.map((_, i) => i)
  const trainingRows = Math.round((n * (FOLDS - 1)) / FOLDS)
  return (
    <Figure
      title="More candidate features than rows"
      state={state}
      hoverReadout={false}
      caption={`Friedman's first benchmark: the target depends non-linearly on 5 inputs, and the others are noise. All monomials up to a degree are generated from the standardised inputs, as a blind generator would, and fitted by ridge regression. Training R² (grey) is measured on the rows each fold was fitted on; cross-validated R² (coloured) on the held-out rows. Each fold trains on ${trainingRows} rows. Once the candidates outnumber the training rows, the training fit becomes nearly perfect while the held-out score falls; more rows or a larger ridge penalty move the turning point.`}
      readouts={
        <>
          {scores.map((s) => (
            <Readout key={s.degree} label={`degree ≤ ${s.degree}, cross-validated R²`} value={formatNumber(s.test)} />
          ))}
          <Readout label="training rows per fold" value={String(trainingRows)} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Curve name="training R²" x={positions} y={scores.map((s) => s.train)} muted />
        <Points name="training R²" x={positions} y={scores.map((s) => s.train)} muted />
        <Curve name="cross-validated R²" x={positions} y={scores.map((s) => s.test)} slot={0} />
        <Points name="cross-validated R²" x={positions} y={scores.map((s) => s.test)} slot={0} />
      </Plot>
    </Figure>
  )
}
