import { useMemo } from 'react'
import {
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
import { linearRegressionData, withMissing } from 'aifn-methods/data/synthetic'
import { linearRegression } from 'aifn-methods/learning/linear'
import { simpleImputer } from 'aifn-methods/learning/preprocessing'
import { dataset } from 'aifn-compute/learning/estimators'
import { rootMeanSquaredError } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import { concat, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const NOISE = 0.3

/** y = x + noise, with x removed at random (MCAR) or more often when it is large (MNAR). */
function draw(name: string, n: number, rate: number, mechanism: 'mcar' | 'mnar') {
  const complete = linearRegressionData(stream(`missing-indicator/${name}`), { n, d: 1, weights: [1], noise: NOISE })
  return withMissing(stream(`missing-indicator/${name}/missing`), complete, { rate, mechanism })
}

/** Mean imputation alone, and mean imputation with a missing-value indicator column, under two mechanisms. */
export function MissingIndicatorFit() {
  const state = useFigureState({
    mechanism: choice(
      [
        { value: 'mcar', label: 'completely at random (MCAR)' },
        { value: 'mnar', label: 'large values go missing (MNAR)' },
      ],
      'mnar',
      { label: 'missingness' },
    ),
    rate: slider(0.05, 0.6, 0.3, { step: 0.05, label: 'share missing' }),
  })
  const mechanism = state.mechanism as 'mcar' | 'mnar'
  const { rate } = state
  const view = useMemo(() => {
    const train = draw('train', 300, rate, mechanism)
    const test = draw('test', 2000, rate, mechanism)
    const imputer = simpleImputer().fit({ x: train.x })
    const withIndicator = (x: Tensor, mask: Tensor) => concat([imputer.transform(x), mask], 1)
    const plain = linearRegression().fit(dataset(imputer.transform(train.x), train.y!))
    const indicated = linearRegression().fit(dataset(withIndicator(train.x, train.meta.missing!), train.y!))
    const fill = toFlat(imputer.statistics)[0]
    const x = Array.from(toFlat(imputer.transform(train.x)))
    const y = Array.from(toFlat(train.y!))
    const gone = Array.from(toFlat(train.meta.missing!)).map((m) => m === 1)
    const [slope, shift] = Array.from(toFlat(indicated.weights))
    // Both models' predictions along the observed range, and the indicator model's prediction for a missing row.
    const line = fromData(Float64Array.of(-3, 3), [2, 1])
    const missingRow = fromData(Float64Array.of(fill, 1), [1, 2])
    return {
      observed: { x: x.filter((_, i) => !gone[i]), y: y.filter((_, i) => !gone[i]) },
      imputed: { x: x.filter((_, i) => gone[i]), y: y.filter((_, i) => gone[i]) },
      plainLine: Array.from(toFlat(plain.decide(line))),
      indicatedLine: Array.from(toFlat(indicated.decide(withIndicator(line, fromData(Float64Array.of(0, 0), [2, 1]))))),
      missingPrediction: toFlat(indicated.decide(missingRow))[0],
      plainAtFill: toFlat(plain.decide(fromData(Float64Array.of(fill), [1, 1])))[0],
      fill,
      slope,
      shift,
      plainRmse: rootMeanSquaredError(toFlat(test.y!), toFlat(plain.decide(imputer.transform(test.x)))),
      indicatedRmse: rootMeanSquaredError(
        toFlat(test.y!),
        toFlat(indicated.decide(withIndicator(test.x, test.meta.missing!))),
      ),
    }
  }, [rate, mechanism])
  const xa = useAxis({ label: 'x (missing values filled with the mean)', range: [-3, 3] })
  const ya = useAxis({ label: 'y', range: [-4, 4] })
  return (
    <Figure
      title="A missing-value indicator"
      state={state}
      hoverReadout={false}
      caption={`300 rows of y = x + noise (standard deviation ${NOISE}). Some x values are removed, either completely at random or, under MNAR, more often when x is large, and are then filled with the mean of the observed x (the coloured column of points at x = ${formatNumber(view.fill)}). Mean imputation alone fits one line through all the rows. Adding an indicator column, 1 where x was missing, lets the model shift its prediction for the filled rows by a learned amount (the diamond): under MNAR the filled rows have large y, and the shift recovers most of it. Under MCAR the filled rows look like any other rows and the indicator adds nothing. Test errors use 2000 fresh rows with the same missingness.`}
      readouts={
        <>
          <Readout label="test RMSE, imputation only" value={formatNumber(view.plainRmse)} />
          <Readout label="test RMSE, with indicator" value={formatNumber(view.indicatedRmse)} />
          <Readout label="indicator weight" value={formatNumber(view.shift)} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Points name="observed rows" x={view.observed.x} y={view.observed.y} muted />
        {view.imputed.x.length > 0 && (
          <Points name="rows with x filled in" x={view.imputed.x} y={view.imputed.y} slot={1} thin />
        )}
        <Curve name="imputation only" x={[-3, 3]} y={view.plainLine} slot={2} dashed />
        <Curve name="with indicator: observed rows" x={[-3, 3]} y={view.indicatedLine} slot={0} />
        <Points
          name="with indicator: a filled row"
          x={[view.fill]}
          y={[view.missingPrediction]}
          slot={0}
          shape={3}
          size={12}
        />
      </Plot>
    </Figure>
  )
}
