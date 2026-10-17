import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Rug,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'
import { semiSupervisedMixtureSteps } from 'aifn-methods/unsupervised/clustering'
import {
  bayesError,
  drawLineSample,
  mixtureDensity,
  ruleBoundaries,
  ruleError,
  type LineMixtureProblem,
} from '../_shared/line-mixtures'

/** Three problems on the line, equal priors; the model always fits one Gaussian per class. */
const PROBLEMS: Record<string, LineMixtureProblem> = {
  correct: { classes: [[[1, -1, 1]], [[1, 1.5, 1]]] },
  bimodal: {
    classes: [
      [
        [0.5, -2.5, 0.6],
        [0.5, 2.0, 0.6],
      ],
      [[1, 0.5, 0.6]],
    ],
  },
  skewed: {
    classes: [
      [
        [0.7, -1.5, 0.6],
        [0.3, 1.2, 1.2],
      ],
      [[1, 2.5, 0.7]],
    ],
  },
}
const U_GRID = [10, 30, 100, 300, 1000, 3000]
const REPEATS = 12
const MAX_STEPS = 300
const X_RANGE: [number, number] = [-6, 6]
const XS = Array.from({ length: 241 }, (_, i) => X_RANGE[0] + ((X_RANGE[1] - X_RANGE[0]) * i) / 240)
const gauss = (x: number, m: number, v: number) =>
  Math.exp((-0.5 * ((x - m) * (x - m))) / v) / Math.sqrt(2 * Math.PI * v)

/**
 * Semi-supervised EM with one Gaussian per class, on a problem where that model is right and on two where it is wrong:
 * the fitted densities and boundary as EM runs, and the error against the number of unlabelled points.
 */
export function MisspecifiedEm() {
  const fig = useFigureState({
    problem: choice(
      [
        { value: 'correct', label: 'each class one Gaussian (model correct)' },
        { value: 'bimodal', label: 'class 0 has two modes' },
        { value: 'skewed', label: 'class 0 is skewed' },
      ],
      'bimodal',
      { label: 'true classes' },
    ),
    perClass: int(4, { min: 1, max: 50, label: 'labels per class', suggestions: [2, 4, 10, 30] }),
    lambda: float(1, { min: 0, max: 1, step: 0.05, label: 'weight λ of the unlabelled data' }),
    u: int(300, { min: 1, max: 3000, scale: 'log10', label: 'unlabelled points', onChart: true }),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { problem: key, perClass, lambda, u, seed } = fig
  const problem = PROBLEMS[key]
  const bayes = useMemo(() => bayesError(problem), [problem])

  // Mean error over repeats for every unlabelled count, and the supervised error (λ = 0).
  const curve = useComputed(
    () => {
      const errors = (count: number, weight: number) => {
        let e = 0
        for (let r = 0; r < REPEATS; r++) {
          const smp = drawLineSample(problem, child(stream(seed), 'repeat', r), perClass, count)
          const alg = semiSupervisedMixtureSteps(
            { x: smp.x.map((v) => [v]), labels: smp.labels, classes: 2 },
            {
              covariance: 'diagonal',
              start: 'per-class',
              regularisation: 0,
              varianceFloor: 0.01,
              objective: 'per-row',
              unlabelledWeight: weight,
              tolerance: 1e-7,
            },
          )
          e += ruleError(problem, run(alg, undefined, MAX_STEPS))
        }
        return e / REPEATS
      }
      return { supervised: errors(0, 0), ssl: U_GRID.map((c) => errors(c, lambda)) }
    },
    [problem, perClass, lambda, seed],
    { mode: 'release' },
  )

  // One sample at the chosen u, traced through EM.
  const sample = useMemo(
    () => drawLineSample(problem, child(stream(seed), 'shown'), perClass, u),
    [problem, seed, perClass, u],
  )
  const states = useMemo(() => {
    const alg = semiSupervisedMixtureSteps(
      { x: sample.x.map((v) => [v]), labels: sample.labels, classes: 2 },
      {
        covariance: 'diagonal',
        start: 'per-class',
        regularisation: 0,
        varianceFloor: 0.01,
        objective: 'per-row',
        unlabelledWeight: lambda,
        tolerance: 1e-7,
      },
    )
    // Keep only the parameters of each iteration (the states also carry n × 2 responsibilities).
    const keep = (st: ReturnType<typeof alg.init>) => ({
      t: st.t,
      weights: st.weights,
      means: st.means,
      covariances: st.covariances,
      error: ruleError(problem, st),
      boundaries: ruleBoundaries(st),
    })
    const root = stream(0)
    let st = alg.init(undefined, root)
    const out = [keep(st)]
    for (let t = 0; t < MAX_STEPS && !st.converged; t++) {
      st = alg.step(st, { t, stream: root })
      out.push(keep(st))
    }
    return out
  }, [sample, lambda, problem])
  const settings = `${key}/${perClass}/${lambda}/${u}/${seed}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const at = pos.settings === settings ? Math.min(pos.at, states.length - 1) : 0
  const s = states[at]

  const xa = useAxis({ label: 'x', range: X_RANGE })
  const ya = useAxis({ label: 'density × prior', range: [0, 0.45] })
  const xb = useAxis({ label: 'unlabelled points', log: true, range: [8, 4000] })
  const yb = useAxis({ label: 'test error', range: [0, 0.45] })

  const labelled = [0, 1].map((k) => sample.x.filter((_, i) => sample.labels[i] === k))
  const unlabelled = sample.x.filter((_, i) => sample.labels[i] < 0)
  const vlines = (bs: number[]) => bs.filter((b) => b > X_RANGE[0] && b < X_RANGE[1])

  return (
    <Figure
      title="Unlabelled data pull a wrong model further from the truth"
      state={fig}
      caption="A two-class problem on the line with equal priors, fitted by semi-supervised EM with one Gaussian per class. Left: the true class densities times ½ (dashed), the fitted ones (solid) and the fitted boundary (ink), for one sample with the chosen numbers of labelled (dots) and unlabelled points (ticks). The player steps through EM from iteration 0, the fit to the labels alone. Right: the exact test error after EM, averaged over 12 samples, against the number of unlabelled points; the dashed line is the labels alone and the dotted one the Bayes error. Drag the vertical line to set the number of unlabelled points. When each class is one Gaussian, more unlabelled points move the error towards the Bayes error. When class 0 has two modes, EM spends one Gaussian on each cluster of the input density, names the right-hand cluster class 1 and misclassifies class 0's right mode: with 4 labels per class the error rises from 0.22 to about 0.26, and with 30 labels per class from 0.12 to about 0.26. When class 0 is skewed, a few hundred unlabelled points help with 4 labels per class and more of them hurt. A small λ delays the rise; it does not stop it."
      controls={
        <Player value={at} onChange={(v) => setPos({ settings, at: v })} count={states.length} label="EM iteration" />
      }
      readouts={
        <>
          <Readout label="EM iteration" value={s.t} />
          <Readout label="error of this fit" value={formatNumber(s.error)} />
          <Readout label="labels alone (mean)" value={formatNumber(curve.value.supervised)} />
          <Readout label="Bayes error" value={formatNumber(bayes)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          {[0, 1].map((k) => (
            <Curve
              key={`true${k}`}
              name={`class ${k}, true`}
              x={XS}
              y={XS.map((x) => 0.5 * mixtureDensity(problem.classes[k], x))}
              slot={k}
              dashed
              thin
            />
          ))}
          {[0, 1].map((k) => (
            <Curve
              key={`fit${k}`}
              name={`class ${k}, fitted`}
              x={XS}
              y={XS.map((x) => s.weights[k] * gauss(x, s.means[k], s.covariances[k]))}
              slot={k}
            />
          ))}
          {vlines(s.boundaries).map((b, i) => (
            <Curve key={`b${i}`} name="fitted boundary" x={[b, b]} y={[0, 0.45]} emphasis />
          ))}
          {unlabelled.length > 0 && <Rug name="unlabelled" values={unlabelled} muted length={6} />}
          {[0, 1].map((k) =>
            labelled[k].length > 0 ? (
              <Points
                key={`l${k}`}
                name={`labelled class ${k}`}
                x={labelled[k]}
                y={labelled[k].map(() => 0.02)}
                slot={k}
                size={9}
              />
            ) : null,
          )}
        </Plot>
        <Plot x={xb} y={yb}>
          <Curve
            name={`semi-supervised EM, λ = ${formatNumber(lambda)}`}
            x={U_GRID}
            y={curve.value.ssl}
            slot={2}
            showPoints
            stale={curve.stale}
          />
          <Curve name="labels alone" x={[8, 4000]} y={[curve.value.supervised, curve.value.supervised]} dashed muted />
          <Curve name="Bayes error" x={[8, 4000]} y={[bayes, bayes]} muted thin />
          <Handle
            kind="x"
            at={u}
            label="unlabelled"
            onDrag={(v) => fig.set('u', Math.round(Math.min(3000, Math.max(1, v))))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
