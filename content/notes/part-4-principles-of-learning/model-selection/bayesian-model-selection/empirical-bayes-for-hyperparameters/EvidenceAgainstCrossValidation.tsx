import { Curve, Figure, float, int, Plot, Readout, useAxis, useComputed, useFigureState } from 'aifn-render'
import { add, matmul } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { ridgeCv } from 'aifn-compute/learning/selection'
import { bayesianLinearEvidence, evidenceMaximisation } from 'aifn-methods/learning/linear'
import { design, drawData, flat, fmt } from '../_shared/blr'

const LOG_LAMBDAS = Array.from({ length: 33 }, (_, i) => -6 + i * 0.25)
const LAMBDAS = LOG_LAMBDAS.map((l) => 10 ** l)

/**
 * Empirical Bayes for a ridge penalty: MacKay's evidence maximisation on a polynomial design, beside leave-one-out
 * cross-validation of the same ridge fits and their true test error on a large fresh sample.
 */
export function EvidenceAgainstCrossValidation() {
  const state = useFigureState({
    n: int(30, { ge: 10, le: 300, suggestions: [15, 30, 100, 300], label: 'data points n' }),
    degree: int(9, { ge: 2, le: 15, suggestions: [5, 9, 12], label: 'polynomial degree' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
  })
  const { n, degree, noise, seed } = state
  const run = useComputed(
    () => {
      const data = drawData(`eb/${seed}`, n, noise)
      const fresh = drawData('eb/fresh', 2000, noise)
      const { X, map } = design(data.x, degree, false)
      const Xf = map(fresh.x)
      const eb = evidenceMaximisation(X, data.y, { intercept: true })
      const evidence: number[] = []
      const truth: number[] = []
      for (const lambda of LAMBDAS) {
        const ev = bayesianLinearEvidence(X, data.y, { alpha: lambda * eb.beta, beta: eb.beta, intercept: true })
        evidence.push(ev.logEvidence)
        truth.push(meanSquaredError(fresh.y, add(matmul(Xf, ev.mean), ev.intercept)))
      }
      const cv = ridgeCv(X, data.y, LAMBDAS, { intercept: true })
      const ebTruth = meanSquaredError(fresh.y, add(matmul(Xf, eb.mean), eb.intercept))
      return {
        evidence,
        truth,
        loo: flat(cv.loo),
        bestLoo: cv.bestLoo,
        eb: {
          logLambda: Math.log10(eb.alpha / eb.beta),
          sd: 1 / Math.sqrt(eb.beta),
          gamma: eb.gamma,
          steps: eb.steps,
          truth: ebTruth,
        },
      }
    },
    [n, degree, noise, seed],
    { mode: 'release' },
  )
  const v = run.value
  const bestTruth = v.truth.reduce((b, x, i) => (x < v.truth[b] ? i : b), 0)
  const key = `${n}/${degree}/${noise}/${seed}`
  const lx = useAxis({ label: 'log₁₀ ridge penalty λ = α/β', range: [-6, 2], nice: false })
  const ey = useAxis({ label: 'log evidence (nats)', hold: 'union', key })
  const my = useAxis({ label: 'mean squared error (log scale)', log: true, hold: 'union', key: `m/${key}` })
  const vline = (x: number, y: number[]) => ({ x: [x, x], y: [Math.min(...y), Math.max(...y)] })
  return (
    <Figure
      title="The evidence and cross-validation choose a ridge penalty"
      state={state}
      hoverReadout={false}
      caption="A polynomial of the chosen degree fitted to n points of y = sin(πx) + σε by Bayesian linear regression with an unpenalised intercept; the posterior mean is the ridge fit with penalty λ = α/β. Left: the log evidence against the penalty, with the noise precision β held at its empirical Bayes value; the vertical line marks the penalty that MacKay's updates reach by maximising over both precisions. Right: the leave-one-out error of the same ridge fits and their true error on 2,000 fresh points, with the empirical Bayes penalty marked. Both criteria usually land near the minimum of the true error. With few points and a high degree, change the dataset to see the evidence and leave-one-out disagree, each in its own direction."
      readouts={
        <>
          <Readout
            label="log₁₀ λ by empirical Bayes / leave-one-out"
            value={`${fmt(v.eb.logLambda)} / ${LOG_LAMBDAS[v.bestLoo]}`}
          />
          <Readout label="log₁₀ λ with the smallest true error" value={`${LOG_LAMBDAS[bestTruth]}`} />
          <Readout
            label="true error at the empirical Bayes / leave-one-out choice"
            value={`${fmt(v.eb.truth)} / ${fmt(v.truth[v.bestLoo])}`}
          />
          <Readout label="noise sd estimated by empirical Bayes (true σ)" value={`${fmt(v.eb.sd)} (${noise})`} />
          <Readout
            label={`effective parameters γ of ${degree}, after MacKay steps`}
            value={`${fmt(v.eb.gamma)}, ${v.eb.steps}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={lx} y={ey} title="log evidence at the estimated noise level">
          <Curve name="log evidence" x={LOG_LAMBDAS} y={v.evidence} slot={0} stale={run.stale} />
          <Curve name="empirical Bayes λ" {...vline(v.eb.logLambda, v.evidence)} emphasis dashed />
        </Plot>
        <Plot x={lx} y={my} title="cross-validation and the truth">
          <Curve name="leave-one-out error" x={LOG_LAMBDAS} y={v.loo} slot={1} stale={run.stale} />
          <Curve name="true error" x={LOG_LAMBDAS} y={v.truth} slot={2} dashed stale={run.stale} />
          <Curve name="empirical Bayes λ" {...vline(v.eb.logLambda, [...v.loo, ...v.truth])} emphasis dashed />
        </Plot>
      </div>
    </Figure>
  )
}
