import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { curve1d } from 'aifn-methods/data/synthetic'
import { linearRegression } from 'aifn-methods/learning/linear'
import { splineFeatures } from 'aifn-methods/learning/preprocessing'
import { logTarget, pipeline, powerTarget, transformTarget } from 'aifn-compute/learning/compose'
import { asTensor, dataset } from 'aifn-compute/learning/estimators'
import type { AnyUnivariate } from 'aifn-compute/probability/distributions'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, mean, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const MAPS = [
  { value: 'none', label: 'none: fit y' },
  { value: 'log', label: 'log y' },
  { value: 'box-cox', label: 'Box–Cox' },
] as const
const GRID = Array.from({ length: 101 }, (_, i) => i / 100)
const average = (t: Tensor) => Number(toFlat(asTensor(mean(t)))[0])

/** A regression on a transformed target, mapped back to y: the median, the mean, and the 90% predictive band. */
export function TransformedTargetFit() {
  const state = useFigureState({
    map: choice(MAPS, 'log', { label: 'target transform' }),
    noise: slider(0.1, 0.8, 0.6, { step: 0.05, label: 'noise (coefficient of variation)' }),
  })
  const { map, noise } = state
  const view = useMemo(() => {
    const train = curve1d(stream('encoding-targets/train'), { n: 300, case: 'gamma', noise })
    const test = curve1d(stream('encoding-targets/test'), { n: 2000, case: 'gamma', noise })
    const truth = curve1d(stream('encoding-targets/grid'), { n: GRID.length, case: 'gamma', noise })
    const regressor = () => pipeline(splineFeatures({ knots: 8, includeBias: false }), linearRegression())
    const data = dataset(train.x, train.y!)
    const model =
      map === 'none'
        ? regressor().fit(data)
        : transformTarget(regressor(), map === 'log' ? logTarget() : powerTarget({ method: 'box-cox' })).fit(data)
    const x = fromData(Float64Array.from(GRID), [GRID.length, 1])
    const predictive = model.predictive(x) as AnyUnivariate
    const quantile = (p: number) => Array.from(toFlat(asTensor(predictive.quantile(fromData(Float64Array.of(p), [])))))
    return {
      train,
      // curve1d draws x at random; the true mean is drawn on its own sorted inputs.
      truthX: Array.from(toFlat(truth.x)),
      truthY: Array.from(toFlat(truth.f!)),
      point: Array.from(toFlat(model.decide(x))),
      expected: Array.from(toFlat(model.expect(x))),
      lo: quantile(0.05),
      hi: quantile(0.95),
      trueMean: average(test.f!),
      pointMean: average(model.decide(test.x) as Tensor),
      expectedMean: average(model.expect(test.x) as Tensor),
    }
  }, [map, noise])
  const xa = useAxis({ label: 'x', range: [0, 1] })
  const ya = useAxis({ label: 'y', range: [-4, 14] })
  return (
    <Figure
      title="Back-transforming a transformed target"
      state={state}
      hoverReadout={false}
      caption="300 positive, right-skewed responses: gamma-distributed with a mean that varies with x (dashed). A regression on cubic spline features of x is fitted to y, to log y, or to a Box–Cox transform of y, and mapped back. The point prediction (solid) is the inverse transform of the fitted value: for a log target that is the median of y, which lies below the mean. The predictive mean (second curve) pushes the whole fitted distribution back through the inverse transform. The shaded band is the 90% predictive interval: fitting y directly gives a symmetric Gaussian band that runs below zero, which a positive quantity cannot do. Raising the noise widens the gap between median and mean. Averages are over 2000 test points."
      readouts={
        <>
          <Readout label="average true mean" value={formatNumber(view.trueMean)} />
          <Readout label="average point prediction" value={formatNumber(view.pointMean)} />
          <Readout label="average predictive mean" value={formatNumber(view.expectedMean)} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Area name="90% predictive band" x={GRID} y={view.hi} base={view.lo} slot={2} line={false} opacity={0.18} />
        <Points name="training data" x={toFlat(view.train.x)} y={toFlat(view.train.y!)} muted />
        <Curve name="true mean" x={view.truthX} y={view.truthY} emphasis dashed />
        <Curve name="point prediction (inverse transform)" x={GRID} y={view.point} slot={0} />
        <Curve name="predictive mean" x={GRID} y={view.expected} slot={1} />
      </Plot>
    </Figure>
  )
}
