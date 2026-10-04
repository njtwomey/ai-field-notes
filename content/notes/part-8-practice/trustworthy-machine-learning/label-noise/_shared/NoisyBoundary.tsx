import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Curve,
  Plot,
  Points,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { boundaryLine, corrupt, fitLogistic, lineAnchors, parametricTest, twoGaussians } from './noiseTests'

const XS = [-4.5, 4.5]
const SHOWN = 400
const CLEAN = XS.map((x) => -x)
const LABEL_NAMES = ['noisy label 0', 'noisy label 1']
const logit = (p: number) => Math.log(p / (1 - p))

/**
 * Two unit-variance Gaussians at (1, 1) and (−1, −1), labels flipped with rates α (positives) and β (negatives),
 * logistic regression fitted to the noisy labels. With `withTest`, anchors on the true boundary (posterior 1/2) and
 * the z-test of Poyiadzi et al. (2022) for H₀: α = β.
 */
export function NoisyBoundary({ withTest = false }: { withTest?: boolean }) {
  const state = useFigureState({
    alpha: float(0, { min: 0, max: 0.4, step: 0.01, label: 'α = P(flip | y = 1)' }),
    // Hidden while β is tied to α: the readouts and the boundaries then use α for both.
    beta: float(0.15, { min: 0, max: 0.4, step: 0.01, label: 'β = P(flip | y = 0)', when: (v) => !v.tied }),
    tied: setting(false, 'uniform noise (β = α)'),
    n: int(1000, { min: 200, max: 5000, step: 100, label: 'training points N' }),
    k: int(8, { min: 1, max: 32, step: 1, label: 'anchors k', when: () => withTest }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const b = state.tied ? state.alpha : state.beta

  const sample = useMemo(() => twoGaussians(state.n, state.seed), [state.n, state.seed])
  const noisy = useMemo(() => corrupt(sample, state.alpha, b), [sample, state.alpha, b])
  const fit = useMemo(() => fitLogistic(sample.X, noisy), [sample, noisy])
  const anchors = useMemo(() => lineAnchors(state.k, state.seed + 7), [state.k, state.seed])
  const test = useMemo(() => parametricTest(fit, anchors), [fit, anchors])

  // Clean posterior: logit η = 2(x₁ + x₂). The noisy Bayes rule thresholds η̃ at 1/2, i.e. η at (1/2 − β)/(1 − α − β).
  const etaStar = (0.5 - b) / (1 - state.alpha - b)
  const noisyBayes = logit(Math.min(Math.max(etaStar, 1e-6), 1 - 1e-6)) / 2

  const shown = useMemo(() => {
    const X = sample.X.slice(0, SHOWN)
    return { x: X.map((p) => p[0]), y: X.map((p) => p[1]), group: noisy.slice(0, SHOWN) }
  }, [sample, noisy])
  const noisyLine = useMemo(() => XS.map((x) => noisyBayes - x), [noisyBayes])
  const fitted = useMemo(() => boundaryLine(fit.theta, XS), [fit])
  const anchorXY = useMemo(() => ({ x: anchors.map((a) => a[0]), y: anchors.map((a) => a[1]) }), [anchors])
  const x1 = useAxis({ label: 'x₁', range: [-4.5, 4.5] })
  const x2 = useAxis({ label: 'x₂', range: [-4.5, 4.5], equal: x1 })

  return (
    <Figure
      title={withTest ? 'Testing for class-conditional noise at anchors' : 'Where the noisy decision boundary goes'}
      purpose="Change the label-flip rates and see how the noisy Bayes boundary and a fitted logistic regression move away from the clean boundary."
      state={state}
      caption={
        withTest
          ? 'Logistic regression is fitted to labels flipped with rates α (positives) and β (negatives). The anchors lie on the clean boundary, where the true posterior is 1/2. Under uniform noise (α = β) the fitted posterior there stays near 1/2; under class-conditional noise it moves towards (1 − α + β)/2 (further, in this example, because the logistic model is misspecified under noise), and the z-score grows with the number of points, the number of anchors and |α − β|. Tie β to α to see the null hold.'
          : 'Labels are flipped with rate α for positives and β for negatives. Under uniform noise (tie β to α) the noisy Bayes boundary, where the noisy posterior is 1/2, stays on the clean one and so does the fitted logistic regression. Under class-conditional noise both move into the region of the class whose labels are flipped more. The fitted line moves further than the noisy Bayes boundary, because the noisy posterior is no longer a logistic function of a linear score and the model is misspecified.'
      }
      readouts={
        withTest ? (
          <>
            <Readout label="mean fitted posterior at anchors" value={formatNumber(test.etaBar)} />
            <Readout label="noisy posterior there, (1 − α + β)/2" value={formatNumber((1 - state.alpha + b) / 2)} />
            <Readout label="standard error under H₀" value={formatNumber(test.se)} />
            <Readout label="z" value={formatNumber(test.z)} />
            <Readout label="two-sided p-value" value={test.p < 1e-4 ? '< 0.0001' : formatNumber(test.p)} />
          </>
        ) : (
          <>
            <Readout label="clean η on the noisy Bayes boundary" value={formatNumber(etaStar)} />
            <Readout label="noisy Bayes boundary crosses x₁ = x₂ at" value={formatNumber(noisyBayes / 2)} />
            <Readout
              label="fitted boundary crosses x₁ = x₂ at"
              value={formatNumber(-fit.theta[0] / (fit.theta[1] + fit.theta[2]))}
            />
          </>
        )
      }
    >
      <Plot x={x1} y={x2} ariaLabel="Two-class scatter with clean, noisy and fitted decision boundaries">
        <Points name="points" {...shown} groupNames={LABEL_NAMES} />
        <Curve name="clean Bayes boundary" x={XS} y={CLEAN} dashed muted />
        <Curve name="noisy Bayes boundary" x={XS} y={noisyLine} dashed slot={2} />
        <Curve name="fitted to noisy labels" x={XS} y={fitted} emphasis />
        {withTest && <Points name="anchors" {...anchorXY} emphasis />}
      </Plot>
    </Figure>
  )
}
