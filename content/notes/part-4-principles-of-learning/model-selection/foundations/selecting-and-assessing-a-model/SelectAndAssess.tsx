import { useMemo, useState } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, square, sub, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { oneStandardError } from 'aifn-compute/learning/selection'
import { gridSearch, kFold, nested } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { freshSample, psplineLearner, sineData, testError } from '../../cross-validation/_shared/problems'

/** Rows drawn: the first 75 are the development set, the last 25 the test set. */
const ROWS = 100
const DEV = 75
const TEST = ROWS - DEV
const NOISE = 0.3
const FOLDS = 10
const FRESH = 5000
/** 25 penalties, log-spaced from 10⁻⁴ to 10⁴. */
const LAMBDAS = Array.from({ length: 25 }, (_, i) => Number((10 ** (-4 + (i * 8) / 24)).toPrecision(3)))
const PENALTIES = space({ lambda: oneOf(LAMBDAS) })
const ENDS = [LAMBDAS[0], LAMBDAS[LAMBDAS.length - 1]]
const STEPS = [
  'validation curve',
  '10-fold cross-validation',
  'one standard error',
  'optimism of the minimum',
  'nested cross-validation',
  'test set',
]
const ESTIMATES = ['training error', 'best CV score', 'nested CV', 'test set']

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const column = (v: ArrayLike<number>) => fromData(Float64Array.from(v), [v.length, 1])

/**
 * One regression problem taken through selection and assessment: a validation curve, 10-fold cross-validation with
 * standard errors, the one-standard-error choice, the optimism of the best score against the true error, nested
 * cross-validation and a held-out test set. Every number comes from engine calls; a Player walks the six steps.
 */
export function SelectAndAssess() {
  const state = useFigureState({
    seed: int(56, { ge: 1, le: 999, suggestions: [56, 1, 2, 3], label: 'data set (seed)' }),
  })
  const { seed } = state
  const [step, setStep] = useState(0)
  const fresh = useMemo(() => freshSample(stream('selecting-and-assessing'), NOISE, FRESH), [])
  const computed = useComputed(
    () => {
      const s = stream(`selecting-and-assessing/${seed}`)
      const all = sineData(child(s, 'data'), ROWS, NOISE)
      const dev = dataset(column(all.x.slice(0, DEV)), fromData(Float64Array.from(all.y.slice(0, DEV))))
      const test = { x: column(all.x.slice(DEV)), y: fromData(Float64Array.from(all.y.slice(DEV))) }
      const search = gridSearch((p) => psplineLearner(p.lambda), PENALTIES, {
        metrics: [meanSquaredError],
        refit: true,
      })
      // The search on all 75 development rows is nested's `unnested` run: the 10-fold curve of steps 2–4.
      const nest = nested(kFold({ k: FOLDS, shuffle: true }), kFold({ k: FOLDS, shuffle: true }), search, dev, {
        stream: child(s, 'cv'),
      })
      const cv = nest.unnested
      const rule = oneStandardError(cv, { simplicity: (i) => -LAMBDAS[i] })
      const fits = LAMBDAS.map((lambda) => psplineLearner(lambda).fit(dev))
      const final = fits[rule.best]
      const squared = toFlat(square(sub(test.y, final.decide(test.x))))
      return {
        train: fits.map((m) => testError(m, dev)),
        truth: fits.map((m) => testError(m, fresh)),
        edf: fits.map((m) => m.fit.edf),
        cv: cv.rows.map((r) => r.mean),
        se: Array.from(toFlat(rule.se)),
        best: rule.best,
        chosen: rule.chosen,
        threshold: rule.threshold,
        nestedScore: nest.nestedScore,
        nestedSe: nest.nestedStd / Math.sqrt(FOLDS),
        perFold: nest.perFold.map((f) => (f.params as { lambda: number }).lambda),
        test: testError(final, test),
        testSe: standardDeviation(squared, { sample: true }) / Math.sqrt(TEST),
      }
    },
    [seed, fresh],
    { mode: 'release' },
  )
  const r = computed.value
  const view = useMemo(() => {
    const bars = LAMBDAS.map((l, i) => ({ from: [l, r.cv[i] - r.se[i]] as const, to: [l, r.cv[i] + r.se[i]] as const }))
    return { bars, lambdaCv: LAMBDAS[r.best], lambda1se: LAMBDAS[r.chosen], errFinal: r.truth[r.best] }
  }, [r])

  const lx = useAxis({ label: 'penalty λ (more regularised →)', log: true })
  const ey = useAxis({ label: 'mean squared error', log: true, range: [0.05, 0.35] })
  const cx = useAxis({ label: 'estimate of the final model’s error', categories: ESTIMATES })
  const cy = useAxis({ label: 'mean squared error', range: [0.04, 0.16] })
  const at = (k: number) => [k]
  return (
    <Figure
      title="Selecting and assessing a P-spline penalty"
      state={state}
      caption={`${ROWS} points of y = sin(πx) + ${NOISE}ε with x uniform on [−1, 1]; the first ${DEV} are the development set and the last ${TEST} the test set. The learner is a P-spline (cubic B-splines on 15 intervals, second-difference penalty λ) at 25 penalties from 10⁻⁴ to 10⁴. Step through with the player. Left: the training error and the ${FOLDS}-fold cross-validation error on the development set (step 1), ±1 standard error bars (step 2), the one-standard-error threshold and the choice it gives (step 3), and the true error of each fit, measured on ${FRESH} fresh points (step 4). Right: estimates of the error of the final model, the fit at the cross-validation minimum refitted on all ${DEV} development rows, beside its true error (dashed): its training error and the best cross-validation score (step 2), nested 10 × 10-fold cross-validation (step 5) and the ${TEST}-row test set (step 6), with ±1 standard error. Change the seed to draw another data set and watch the estimates scatter about the truth.`}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={STEPS.length}
          label="step"
          format={(v) => `${v + 1}. ${STEPS[v]}`}
        />
      }
      readouts={
        <>
          {step === 0 && (
            <>
              <Readout label="training error at λ = 10⁻⁴" value={`${fmt(r.train[0])}, df ${fmt(r.edf[0])}`} />
              <Readout
                label="training error at λ = 10⁴"
                value={`${fmt(r.train[LAMBDAS.length - 1])}, df ${fmt(r.edf[LAMBDAS.length - 1])}`}
              />
            </>
          )}
          {step >= 1 && (
            <Readout
              label="CV minimum"
              value={`λ = ${fmt(view.lambdaCv)}: ${fmt(r.cv[r.best])} ± ${fmt(r.se[r.best])}, df ${fmt(r.edf[r.best])}`}
            />
          )}
          {step >= 2 && (
            <Readout
              label="one-standard-error choice"
              value={`λ = ${fmt(view.lambda1se)}: ${fmt(r.cv[r.chosen])} ≤ ${fmt(r.threshold)}, df ${fmt(r.edf[r.chosen])}`}
            />
          )}
          {step >= 3 && (
            <>
              <Readout label="true error at the CV minimum" value={fmt(view.errFinal)} />
              <Readout label="true error at the 1-SE choice" value={fmt(r.truth[r.chosen])} />
              <Readout label="best CV score minus true error" value={fmt(r.cv[r.best] - view.errFinal)} />
            </>
          )}
          {step >= 4 && (
            <>
              <Readout label="nested estimate" value={`${fmt(r.nestedScore)} ± ${fmt(r.nestedSe)}`} />
              <Readout label="λ chosen in each outer fold" value={r.perFold.map(fmt).join(', ')} />
            </>
          )}
          {step >= 5 && <Readout label="test error (25 rows)" value={`${fmt(r.test)} ± ${fmt(r.testSe)}`} />}
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={lx} y={ey} title={`${STEPS[step]}`}>
          {step >= 1 && <Segments name="±1 SE" segments={view.bars} slot={0} />}
          <Curve name="training error" x={LAMBDAS} y={r.train} slot={1} stale={computed.stale} />
          <Curve name={`${FOLDS}-fold CV`} x={LAMBDAS} y={r.cv} slot={0} showPoints stale={computed.stale} />
          {step >= 2 && <Curve name="min + 1 SE" x={ENDS} y={[r.threshold, r.threshold]} slot={0} dashed />}
          {step >= 3 && <Curve name="true error" x={LAMBDAS} y={r.truth} emphasis dashed stale={computed.stale} />}
          {step >= 1 && <Annotation x={view.lambdaCv} text="CV min" dashed slot={0} />}
          {step >= 2 && <Annotation at={[view.lambda1se, r.cv[r.chosen]]} text="1-SE choice" />}
        </Plot>
        <Plot x={cx} y={cy} title="the final model">
          {step >= 3 && (
            <Curve
              name="true error"
              x={[-0.5, ESTIMATES.length - 0.5]}
              y={[view.errFinal, view.errFinal]}
              emphasis
              dashed
            />
          )}
          {step >= 1 && (
            <>
              <Points name="training error" x={at(0)} y={[r.train[r.best]]} slot={1} stale={computed.stale} />
              <Points name="best CV score" x={at(1)} y={[r.cv[r.best]]} slot={0} stale={computed.stale} />
            </>
          )}
          {step >= 4 && (
            <>
              <Segments
                name="nested ± 1 SE"
                segments={[{ from: [2, r.nestedScore - r.nestedSe], to: [2, r.nestedScore + r.nestedSe] }]}
                slot={2}
              />
              <Points name="nested CV" x={at(2)} y={[r.nestedScore]} slot={2} stale={computed.stale} />
            </>
          )}
          {step >= 5 && (
            <>
              <Segments
                name="test ± 1 SE"
                segments={[{ from: [3, r.test - r.testSe], to: [3, r.test + r.testSe] }]}
                slot={2}
              />
              <Points name="test set" x={at(3)} y={[r.test]} slot={2} stale={computed.stale} />
            </>
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
