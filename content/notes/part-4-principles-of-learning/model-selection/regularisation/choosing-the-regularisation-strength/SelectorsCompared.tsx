import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linearRegressionData } from 'aifn-methods/data/synthetic'
import { evidenceMaximisation, linearRegression } from 'aifn-methods/learning/linear'
import { child, stream } from 'aifn-compute/foundation/random'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { fromData, matmul, norm, sub, toFlat, toRows, transpose } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { oneStandardError, ridgeCv } from 'aifn-compute/learning/selection'
import { gridSearch, kFold } from 'aifn-compute/learning/validate'
import { discrepancyPrinciple, lCurve, tikhonov, tikhonovPath } from 'aifn-compute/numerics/regularisation'

const P = 20
const TEST = 4000
const FOLDS = 10
/** 41 penalties, log-spaced from 10⁻³ to 10³, in the scaling ‖y − Xw‖² + λ‖w‖². */
const LAMBDAS = Array.from({ length: 41 }, (_, i) => Number((10 ** (-3 + i * 0.15)).toPrecision(3)))
const PENALTIES = space({ l2: oneOf(LAMBDAS) })
const SMOOTH = Array.from({ length: P }, (_, j) => Math.sin((Math.PI * (j + 1)) / (P + 1)))
const RULES = ['10-fold CV', 'one-SE rule', 'GCV', 'evidence', 'L-curve', 'discrepancy'] as const

const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))
const argmin = (v: ArrayLike<number>) => Array.from(v).reduce((b, x, i, a) => (x < a[b] ? i : b), 0)

/**
 * Six rules for the ridge penalty on one simulated regression, each choice marked on the true prediction error, with
 * the L-curve beside it.
 */
export function SelectorsCompared() {
  const state = useFigureState({
    coefficients: choice(['smooth', 'random'] as const, 'smooth', { label: 'true coefficients' }),
    n: int(50, { ge: 25, le: 400, suggestions: [25, 50, 100, 400], label: 'training rows n' }),
    noise: slider(0.25, 3, 1, { step: 0.25, label: 'noise σ' }),
    correlation: slider(0, 0.95, 0.9, { step: 0.05, label: 'neighbour correlation ρ' }),
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3], label: 'data set (seed)' }),
  })
  const { coefficients, n, noise, correlation, seed } = state

  const computed = useComputed(
    () => {
      const s = stream(`choosing-lambda/${seed}`)
      const weights =
        coefficients === 'smooth'
          ? SMOOTH
          : Array.from(linearRegressionData(child(s, 'weights'), { n: 1, d: P }).weights)
      const options = { weights, noise, correlation }
      const train = linearRegressionData(child(s, 'train'), { n, ...options })
      const test = linearRegressionData(child(stream('choosing-lambda'), 'test'), { n: TEST, ...options })
      const truth = fromData(Float64Array.from(weights))

      // The truth along the grid: prediction error on fresh rows and squared solution error.
      const path = tikhonovPath(train.x, train.y!, LAMBDAS)
      const preds = toRows(transpose(matmul(test.x, transpose(path.x)))) as number[][]
      const prediction = preds.map((row) => meanSquaredError(test.y!, row))
      const solution = (toRows(path.x) as number[][]).map((w) => norm(sub(fromData(Float64Array.from(w)), truth)) ** 2)
      const at = (lambda: number) => {
        // The discrepancy principle reports λ = ∞ when its target is above ‖y‖; the fit is then zero.
        const w = tikhonov(train.x, train.y!, { lambda: Number.isFinite(lambda) ? lambda : 1e15 }).x
        return {
          prediction: meanSquaredError(test.y!, toFlat(matmul(test.x, w))),
          solution: norm(sub(w, truth)) ** 2,
        }
      }

      const search = gridSearch((p) => linearRegression({ l2: p.l2, intercept: false }), PENALTIES, {
        metrics: [meanSquaredError],
        refit: false,
      })
      const cv = search.run(dataset(train.x, train.y!), kFold({ k: FOLDS, shuffle: true }), child(s, 'folds'))
      const ose = oneStandardError(cv, { simplicity: LAMBDAS.map((l) => -l) })
      const smoother = ridgeCv(train.x, train.y!, LAMBDAS, { intercept: false })
      const evidence = evidenceMaximisation(train.x, train.y!)
      const curve = lCurve(train.x, train.y!, LAMBDAS)
      const discrepancy = discrepancyPrinciple(train.x, train.y!, { sigma: noise })

      const chosen = [
        LAMBDAS[ose.best],
        LAMBDAS[ose.chosen],
        LAMBDAS[smoother.bestGcv],
        evidence.alpha / evidence.beta,
        curve.corner.lambda,
        discrepancy.lambda,
      ]
      return {
        prediction,
        solution,
        cv: Array.from(toFlat(ose.mean)),
        gcv: Array.from(toFlat(smoother.gcv)),
        chosen,
        scores: chosen.map(at),
        lcurve: {
          residual: Array.from(toFlat(curve.residualNorm)),
          solution: Array.from(toFlat(curve.solutionNorm)),
          corner: curve.corner.index,
        },
        discrepancy: { residual: discrepancy.residualNorm, solution: discrepancy.solutionNorm },
      }
    },
    [coefficients, n, noise, correlation, seed],
    { mode: 'release' },
  )
  const r = computed.value
  const best = useMemo(() => ({ prediction: argmin(r.prediction), solution: argmin(r.solution) }), [r])
  // A discrepancy target outside the attainable range gives λ = 0 or ∞, which a log axis cannot show.
  const shown = useMemo(
    () => RULES.map((_, i) => i).filter((i) => r.chosen[i] > 0 && Number.isFinite(r.chosen[i])),
    [r],
  )

  const lx = useAxis({ label: 'λ', log: true })
  // Zoom on the trough: from just below the best error to two and a half times it.
  const floor = r.prediction[best.prediction]
  const ey = useAxis({ label: 'prediction error', log: true, range: [0.95 * floor, 2.5 * floor] })
  const rx = useAxis({ label: '‖y − Xŵ‖', log: true })
  const wy = useAxis({ label: '‖ŵ‖', log: true })
  return (
    <Figure
      title="Six rules for one ridge penalty"
      state={state}
      caption={`Ridge regression without intercept on n rows of ${P} standard-normal predictors, neighbours correlated ρ, with smooth true coefficients sin(πj/21) or coefficients drawn from N(0, 1), and noise σ. Left: the true prediction error near its minimum of the fit at each of 41 penalties, measured on ${TEST} fresh rows, with the ${FOLDS}-fold cross-validation and GCV curves; the labelled points are the λ chosen by each rule, placed on the true error. The evidence fits both the prior precision and the noise; the discrepancy principle is given the true σ. Right: the L-curve, with its corner (largest curvature) and the discrepancy point. Switch the coefficients to random to see the evidence improve and the L-curve lose its corner.`}
      readouts={
        <>
          {RULES.map((name, i) => (
            <Readout
              key={name}
              label={name}
              value={`λ = ${fmt(r.chosen[i])}, error ${fmt(r.scores[i].prediction)}, ‖ŵ − w*‖² ${fmt(r.scores[i].solution)}`}
            />
          ))}
          <Readout
            label="best for prediction"
            value={`λ = ${fmt(LAMBDAS[best.prediction])}, error ${fmt(r.prediction[best.prediction])}`}
          />
          <Readout
            label="best for the coefficients"
            value={`λ = ${fmt(LAMBDAS[best.solution])}, ‖ŵ − w*‖² ${fmt(r.solution[best.solution])}`}
          />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={lx} y={ey}>
          <Curve name={`${FOLDS}-fold CV`} x={LAMBDAS} y={r.cv} slot={0} stale={computed.stale} />
          <Curve name="GCV" x={LAMBDAS} y={r.gcv} slot={1} dashed stale={computed.stale} />
          <Curve name="true prediction error" x={LAMBDAS} y={r.prediction} emphasis stale={computed.stale} />
          <Points
            name="chosen λ"
            x={shown.map((i) => r.chosen[i])}
            y={shown.map((i) => r.scores[i].prediction)}
            labels={shown.map((i) => RULES[i])}
            slot={2}
            stale={computed.stale}
          />
        </Plot>
        <Plot x={rx} y={wy}>
          <Curve name="L-curve" x={r.lcurve.residual} y={r.lcurve.solution} emphasis stale={computed.stale} />
          <Points
            name="corner and discrepancy"
            x={[r.lcurve.residual[r.lcurve.corner], r.discrepancy.residual]}
            y={[r.lcurve.solution[r.lcurve.corner], r.discrepancy.solution]}
            labels={['L-curve corner', 'discrepancy']}
            slot={2}
            stale={computed.stale}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
