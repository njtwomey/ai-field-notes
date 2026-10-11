import { useContext, useMemo, type ReactNode } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Points,
  Readout,
  Rug,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { regression1d } from 'aifn-methods/data/synthetic'
import { linearRegression } from 'aifn-methods/learning/linear'
import { splineFeatures } from 'aifn-methods/learning/preprocessing'
import { dataset } from 'aifn-compute/learning/estimators'
import { rootMeanSquaredError } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, toRows } from 'aifn-compute/foundation/tensor'

const PANEL = 260
const NOISE = 0.2
/** Each function's training range, as regression1d draws it, and the wider range plotted. */
const RANGES = { sine: [0, 2 * Math.PI], step: [-2, 2], bump: [-1, 1] } as const
type Fn = keyof typeof RANGES

const FUNCTIONS = [
  { value: 'sine', label: 'sine' },
  { value: 'step', label: 'step' },
  { value: 'bump', label: 'bump' },
] as const
const ENCODINGS = [
  { value: 'raw', label: 'raw value' },
  { value: '0', label: 'bins (degree 0)' },
  { value: '1', label: 'piecewise linear (degree 1)' },
  { value: '3', label: 'cubic spline (degree 3)' },
] as const

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** One number encoded as bins, hat functions or cubic B-splines, and a linear model fitted on the encoding. */
export function NumericEncodingFit() {
  const state = useFigureState({
    fn: choice(FUNCTIONS, 'sine', { label: 'true function' }),
    encoding: choice(ENCODINGS, '3', { label: 'encoding' }),
    knots: int(8, { ge: 2, le: 40, suggestions: [4, 8, 16], label: 'knots' }),
    placement: choice(
      [
        { value: 'uniform', label: 'uniform (equal width)' },
        { value: 'quantile', label: 'quantile (equal count)' },
      ],
      'uniform',
      { label: 'knot placement' },
    ),
    extrapolation: choice(
      [
        { value: 'constant', label: 'constant' },
        { value: 'continue', label: 'continue' },
      ],
      'constant',
      { label: 'outside the training range' },
    ),
  })
  const { fn, encoding, knots, placement, extrapolation } = state
  const [a, b] = RANGES[fn as Fn]
  const pad = 0.3 * (b - a)
  const view = useMemo(() => {
    const train = regression1d(stream('numeric-encoding/train'), { n: 120, fn: fn as Fn, noise: NOISE })
    const test = regression1d(stream('numeric-encoding/test'), { n: 400, fn: fn as Fn, noise: NOISE })
    // The true function on a grid wider than the training range, for the plot.
    const grid = regression1d(stream('numeric-encoding/grid'), {
      n: 241,
      fn: fn as Fn,
      noise: NOISE,
      range: [a - pad, b + pad],
      spacing: 'even',
    })
    const xs = Array.from(toFlat(grid.x))
    if (encoding === 'raw') {
      const model = linearRegression().fit(dataset(train.x, train.y!))
      return {
        train,
        xs,
        truth: Array.from(toFlat(grid.f!)),
        fit: Array.from(toFlat(model.decide(grid.x))),
        basis: [] as number[][],
        knotsAt: [] as number[],
        columns: 1,
        trainRmse: rootMeanSquaredError(toFlat(train.y!), toFlat(model.decide(train.x))),
        testRmse: rootMeanSquaredError(toFlat(test.y!), toFlat(model.decide(test.x))),
      }
    }
    const degree = Number(encoding)
    const encoder = splineFeatures({
      knots,
      degree,
      knotPlacement: placement as 'uniform' | 'quantile',
      extrapolation: extrapolation as 'constant' | 'continue',
      includeBias: false,
    }).fit({ x: train.x })
    const model = linearRegression({ l2: 1e-6 }).fit(dataset(encoder.transform(train.x), train.y!))
    const rows = toRows(encoder.transform(grid.x)) as number[][]
    const full = Array.from(toFlat(encoder.knots))
    return {
      train,
      xs,
      truth: Array.from(toFlat(grid.f!)),
      fit: Array.from(toFlat(model.decide(encoder.transform(grid.x)))),
      basis: Array.from({ length: rows[0].length }, (_, j) => rows.map((r) => r[j])),
      // The base knots: the full knot vector without the degree extra knots added at each end.
      knotsAt: full.slice(degree, full.length - degree),
      columns: encoder.perColumn,
      trainRmse: rootMeanSquaredError(toFlat(train.y!), toFlat(model.decide(encoder.transform(train.x)))),
      testRmse: rootMeanSquaredError(toFlat(test.y!), toFlat(model.decide(encoder.transform(test.x)))),
    }
  }, [fn, encoding, knots, placement, extrapolation, a, b, pad])
  const xBasis = useAxis({ label: 'x', range: [a - pad, b + pad] })
  const yBasis = useAxis({ label: 'feature value', range: [-0.1, 1.1] })
  const xFit = useAxis({ label: 'x', range: [a - pad, b + pad] })
  const yFit = useAxis({ label: 'y', hold: 'union', key: fn })
  const edges = fromData(Float64Array.of(a, b), [2])
  return (
    <Figure
      title="One number, several encodings"
      state={state}
      hoverReadout={false}
      caption={`Left: the encoding's features as functions of x, on a range 30% wider than the training data (between the dashed lines). Degree 0 B-splines are bin indicators, degree 1 are hat functions (a piecewise-linear encoding) and degree 3 are smooth cubic bumps; ticks mark the knots, and one feature is dropped because the model has an intercept. The raw value has no features to draw. Right: 120 noisy training points (noise standard deviation ${NOISE}), the true function (dashed) and a linear regression fitted on the features. Quantile knots follow the data's density; outside the training range, 'constant' holds every feature at its boundary value and 'continue' extends the end polynomials. The test error is measured inside the training range; ${NOISE} is the best achievable.`}
      readouts={
        <>
          <Readout label="model inputs" value={String(view.columns)} />
          <Readout label="training RMSE" value={formatNumber(view.trainRmse)} />
          <Readout label="test RMSE" value={formatNumber(view.testRmse)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xBasis} y={yBasis} title="Features" legend={false}>
            {view.basis.map((c, j) => (
              <Curve key={j} name={`feature ${j + 1}`} x={view.xs} y={c} slot={j % 8} thin />
            ))}
            {view.knotsAt.length > 0 && <Rug name="knots" values={view.knotsAt} emphasis />}
            <Curve name="training range" x={[a, a]} y={[-0.1, 1.1]} muted dashed silent />
            <Curve name="training range end" x={[b, b]} y={[-0.1, 1.1]} muted dashed silent />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xFit} y={yFit} title="Fit">
            <Points name="training data" x={toFlat(view.train.x)} y={toFlat(view.train.y!)} muted />
            <Curve name="true function" x={view.xs} y={view.truth} emphasis dashed />
            <Curve name="fitted" x={view.xs} y={view.fit} slot={0} />
            <Rug name="training range" values={toFlat(edges)} muted />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
