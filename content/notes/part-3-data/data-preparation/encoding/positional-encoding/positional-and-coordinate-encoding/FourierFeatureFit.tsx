import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { regression1d } from 'aifn-methods/data/synthetic'
import { linearRegression } from 'aifn-methods/learning/linear'
import { randomFourierFeatures } from 'aifn-methods/learning/preprocessing'
import { dataset } from 'aifn-compute/learning/estimators'
import { rootMeanSquaredError } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

const NOISE = 0.05

/** A coordinate encoded by random Fourier features of one lengthscale, and a ridge regression on the features. */
export function FourierFeatureFit() {
  const state = useFigureState({
    encoding: choice(
      [
        { value: 'raw', label: 'raw coordinate' },
        { value: 'fourier', label: 'Fourier features' },
      ],
      'fourier',
      { label: 'encoding' },
    ),
    lengthscale: float(0.03, {
      gt: 0,
      le: 1,
      scale: 'log10',
      suggestions: [0.1, 0.03, 0.01, 0.003],
      label: 'lengthscale ℓ',
    }),
    components: int(200, { ge: 2, le: 2000, suggestions: [20, 200, 800], label: 'features D' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { encoding, lengthscale, components, seed } = state
  const view = useMemo(() => {
    const train = regression1d(stream('fourier-features/train'), { n: 200, fn: 'doppler', noise: NOISE })
    const test = regression1d(stream('fourier-features/test'), { n: 500, fn: 'doppler', noise: NOISE })
    const grid = regression1d(stream('fourier-features/grid'), {
      n: 801,
      fn: 'doppler',
      noise: NOISE,
      spacing: 'even',
    })
    if (encoding === 'raw') {
      const model = linearRegression().fit(dataset(train.x, train.y!))
      return {
        train,
        grid,
        fit: toFlat(model.decide(grid.x)),
        trainRmse: rootMeanSquaredError(toFlat(train.y!), toFlat(model.decide(train.x))),
        testRmse: rootMeanSquaredError(toFlat(test.y!), toFlat(model.decide(test.x))),
      }
    }
    const encoder = randomFourierFeatures({ components, lengthscale }).fit(
      { x: train.x },
      { stream: stream(`fourier-features/draws/${seed}`) },
    )
    const model = linearRegression({ l2: 1e-3 }).fit(dataset(encoder.transform(train.x), train.y!))
    const predict = (x: typeof train.x) => toFlat(model.decide(encoder.transform(x)))
    return {
      train,
      grid,
      fit: predict(grid.x),
      trainRmse: rootMeanSquaredError(toFlat(train.y!), predict(train.x)),
      testRmse: rootMeanSquaredError(toFlat(test.y!), predict(test.x)),
    }
  }, [encoding, lengthscale, components, seed])
  const x = useAxis({ label: 'x', range: [0, 1] })
  const y = useAxis({ label: 'y', range: [-0.7, 0.7] })
  return (
    <Figure
      title="Fourier features resolve fine detail"
      state={state}
      hoverReadout={false}
      caption={`200 noisy samples (noise standard deviation ${NOISE}) of the Doppler function, which oscillates fast near x = 0 and slowly near x = 1. A least-squares fit on the raw coordinate can only be a line. A ridge regression on D random Fourier features cos(ωx + b), with frequencies ω drawn from a normal distribution of standard deviation 1/ℓ, fits detail down to a scale of about the lengthscale ℓ: a long ℓ smooths away the fast oscillations, a short ℓ fits the noise and the test error rises again. The test error uses 500 fresh samples; ${NOISE} is the best achievable.`}
      readouts={
        <>
          <Readout label="training RMSE" value={formatNumber(view.trainRmse)} />
          <Readout label="test RMSE" value={formatNumber(view.testRmse)} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Points name="training data" x={toFlat(view.train.x)} y={toFlat(view.train.y!)} muted />
        <Curve name="true function" x={toFlat(view.grid.x)} y={toFlat(view.grid.f!)} emphasis dashed />
        <Curve name="fitted" x={toFlat(view.grid.x)} y={view.fit} slot={0} />
      </Plot>
    </Figure>
  )
}
