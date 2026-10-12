import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'
import {
  classifierLogRatio,
  classifierTrace,
  drawSamples,
  featureRows,
  logRatio,
  pickCentres,
  ratioErrors,
  type Features,
  type Shift,
} from '../_shared/ratio'

const LO = -4
const HI = 5
const X = toFlat(linspace(LO, HI, 241))
const SIGMA = 0.6
const CENTRES = 15

/**
 * Logistic regression between a sample from p (label 1) and a sample from q (label 0), trained by gradient descent and
 * scrubbed step by step. The classifier's log odds, shifted by log(n_q/n_p), is the estimate of log r(x).
 */
export function ClassifierRatio() {
  const state = useFigureState({
    features: choice<Features>(
      [
        { value: 'linear', label: 'linear: 1, x' },
        { value: 'quadratic', label: 'quadratic: 1, x, x²' },
        { value: 'kernel', label: `${CENTRES} Gaussian bumps` },
      ],
      'quadratic',
      { label: 'features φ(x)' },
    ),
    mean: float(1, { min: -2, max: 3, step: 0.05, label: 'mean of p' }),
    sd: float(0.6, { min: 0.3, max: 1.6, step: 0.05, label: 'standard deviation of p' }),
    np: int(100, { min: 10, max: 1000, label: 'points from p', suggestions: [30, 100, 1000] }),
    nq: int(200, { min: 10, max: 1000, label: 'points from q', suggestions: [30, 200, 1000] }),
    lambda: float(0.001, { min: 0, max: 1, step: 0.001, label: 'L2 penalty λ', suggestions: [0, 0.001, 0.1] }),
    steps: int(150, { min: 10, max: 2000, label: 'gradient steps', suggestions: [50, 150, 1000] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const shift: Shift = useMemo(() => ({ mean: state.mean, sd: state.sd }), [state.mean, state.sd])

  const run = useMemo(() => {
    const { xp, xq } = drawSamples(state.seed, state.np, state.nq, shift)
    const centres = pickCentres(
      [...xp, ...xq].sort((a, b) => a - b),
      CENTRES,
    )
    const rows = (xs: readonly number[]) => featureRows(xs, state.features, centres, SIGMA)
    const tr = classifierTrace(rows(xp), rows(xq), state.lambda, state.steps)
    return { xp, xq, grid: rows(X), tr }
  }, [state.seed, state.np, state.nq, shift, state.features, state.lambda, state.steps])

  // The walk-through restarts at step 0 whenever the run changes.
  const [position, setPosition] = useState({ run, k: 0 })
  const k = position.run === run ? Math.min(position.k, run.tr.weights.length - 1) : 0
  const w = run.tr.weights[k]

  const logR = useMemo(() => classifierLogRatio(run.grid, w, state.np, state.nq), [run, w, state.np, state.nq])
  const truth = useMemo(() => X.map((x) => logRatio(x, shift)), [shift])
  // The classifier's probability that x came from p, and the Bayes-optimal one for these sample sizes.
  const prob = useMemo(() => logR.map((l) => sigmoid(l + Math.log(state.np / state.nq))), [logR, state.np, state.nq])
  const bayes = useMemo(() => truth.map((l) => sigmoid(l + Math.log(state.np / state.nq))), [truth, state.np, state.nq])
  const errors = ratioErrors(X, logR, shift)

  // Labels on the probability chart: 1 for the p sample, 0 for the q sample, spread a little so the points separate.
  const jitter = (i: number) => 0.04 * Math.sin(12.9898 * i)
  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'P(from p | x)', range: [-0.1, 1.1] })
  const xb = useAxis({ label: 'x', range: [LO, HI] })
  const yb = useAxis({ label: 'log r(x)', range: [-8, 4] })
  return (
    <Figure
      title="A classifier between the two samples"
      state={state}
      caption="Left: the sample from p at label 1 and the sample from q at label 0, the trained classifier's probability that a point came from p, and the best possible one for these sample sizes (dashed). Right: the classifier's log odds plus log(n_q/n_p), which estimates log r(x), against the truth. Play to watch gradient descent fit it. Two Gaussians have a quadratic log ratio, so quadratic features can match it exactly; linear features can only draw a straight line, and Gaussian bumps fit it where the data are and flatten beyond them."
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ run, k: v })}
          count={run.tr.weights.length}
          label="step"
          format={(v) => String(run.tr.index[v])}
        />
      }
      readouts={
        <>
          <Readout label="step" value={run.tr.index[k]} />
          <Readout label="penalised log loss" value={formatNumber(run.tr.loss[k])} />
          <Readout label="squared error ½ E_q[(r̂ − r)²]" value={formatNumber(errors.squared)} />
          <Readout label="KL error" value={formatNumber(errors.kl)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={300}>
          <Points name="from q (label 0)" x={run.xq} y={run.xq.map((_, i) => jitter(i))} slot={0} thin />
          <Points name="from p (label 1)" x={run.xp} y={run.xp.map((_, i) => 1 + jitter(i + 7))} slot={1} thin />
          <Curve name="best possible" x={X} y={bayes} emphasis dashed />
          <Curve name="classifier" x={X} y={prob} slot={2} width={3} />
        </Plot>
        <Plot x={xb} y={yb} height={300}>
          <Curve name="log r(x)" x={X} y={truth} emphasis dashed />
          <Curve name="log r̂(x)" x={X} y={logR} slot={2} width={3} />
        </Plot>
      </div>
    </Figure>
  )
}
