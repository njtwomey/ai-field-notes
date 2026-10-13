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
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linearRegressionData } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, norm, sub, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { filterFactors, landweberSteps, tikhonovPath } from 'aifn-compute/numerics/regularisation'

const P = 20
const N = 40
const T = 5000
const SMOOTH = Array.from({ length: P }, (_, j) => Math.sin((Math.PI * (j + 1)) / (P + 1)))
/** The steps the player visits: 0, then about 120 log-spaced steps up to T. */
const STEPS = [
  0,
  ...new Set(Array.from({ length: 121 }, (_, i) => Math.max(1, Math.round(10 ** ((i * Math.log10(T)) / 120))))),
]

const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/**
 * Landweber iteration on a correlated regression: its filter factors at step k beside Tikhonov's at λ = 1/(ωk), and
 * the semiconvergence of the error, with the discrepancy-principle stop.
 */
export function LandweberAgainstTikhonov() {
  const state = useFigureState({
    noise: slider(0.25, 3, 1, { step: 0.25, label: 'noise σ' }),
    correlation: slider(0, 0.99, 0.95, { step: 0.01, label: 'neighbour correlation ρ' }),
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3], label: 'data set (seed)' }),
  })
  const { noise, correlation, seed } = state
  const [position, setPosition] = useState(0)

  const computed = useComputed(
    () => {
      const data = linearRegressionData(child(stream(`landweber/${seed}`), 'train'), {
        n: N,
        weights: SMOOTH,
        noise,
        correlation,
      })
      const truth = fromData(Float64Array.from(SMOOTH))
      const run = trace(landweberSteps({ A: data.x, b: data.y! }), undefined, T, { keep: 'all' })
      const states = STEPS.map((k) => run.steps[Math.min(k, run.steps.length - 1)])
      const omega = states[0].step
      const error = states.map((s) => norm(sub(s.x, truth)))
      const residual = states.map((s) => s.residualNorm)
      // At step 0 nothing is fitted, which Tikhonov matches only as λ → ∞.
      const lambdas = STEPS.map((k) => (k === 0 ? 1e12 : 1 / (omega * k)))
      const path = tikhonovPath(data.x, data.y!, lambdas)
      const tikhonovError = (toRows(path.x) as number[][]).map((w) => norm(sub(fromData(Float64Array.from(w)), truth)))
      const singular = Array.from(toFlat(path.singularValues))
      const delta = noise * Math.sqrt(N)
      const stop = residual.findIndex((v) => v <= delta)
      return {
        omega,
        singular,
        landweber: states.map((s) => Array.from(toFlat(s.filterFactors))),
        tikhonov: lambdas.map((l) => Array.from(toFlat(filterFactors(singular, l)))),
        error,
        residual,
        tikhonovError,
        lambdas,
        delta,
        stop,
      }
    },
    [noise, correlation, seed],
    { mode: 'release' },
  )
  const r = computed.value
  const i = Math.min(position, STEPS.length - 1)
  const bestStep = useMemo(() => r.error.reduce((b, v, j) => (v < r.error[b] ? j : b), 0), [r])
  const bestTikhonov = useMemo(() => Math.min(...r.tikhonovError), [r])

  const sx = useAxis({ label: 'singular value dⱼ', log: true })
  const fy = useAxis({ label: 'filter factor φⱼ', range: [0, 1.05] })
  const kx = useAxis({ label: 'step k', log: true, range: [1, T] })
  const ny = useAxis({ label: 'norm', log: true })
  const later = STEPS.slice(1)
  return (
    <Figure
      title="Landweber iteration filters like Tikhonov"
      state={state}
      caption={`Landweber iteration (gradient descent with step ω = 1/d₁²) from zero on ${N} rows of ${P} standard-normal predictors, neighbours correlated ρ, with smooth true coefficients sin(πj/21) and noise σ. Left: the filter factor of each singular direction after k steps, 1 − (1 − ωd_j²)^k, beside Tikhonov's d_j²/(d_j² + λ) at λ = 1/(ωk). Right: the error ‖w_k − w*‖ falls and then rises (semiconvergence) while the residual ‖y − Xw_k‖ falls throughout; dashed, the error of Tikhonov at λ = 1/(ωk). The discrepancy principle stops at the first step whose residual is below δ = σ√n. Play the steps to watch the weak directions being fitted.`}
      readouts={
        <>
          <Readout label="step k" value={String(STEPS[i])} />
          <Readout label="matching λ = 1/(ωk)" value={STEPS[i] === 0 ? '∞' : fmt(r.lambdas[i])} />
          <Readout label="error at k" value={fmt(r.error[i])} />
          <Readout label="best step" value={`k = ${STEPS[bestStep]}, error ${fmt(r.error[bestStep])}`} />
          <Readout
            label="discrepancy stop"
            value={r.stop < 0 ? 'not reached' : `k = ${STEPS[r.stop]}, error ${fmt(r.error[r.stop])}`}
          />
          <Readout label="best Tikhonov error on the grid" value={fmt(bestTikhonov)} />
        </>
      }
    >
      <Player label="step" value={i} count={STEPS.length} onChange={setPosition} format={(p) => String(STEPS[p])} />
      <Plots cols={2}>
        <Plot x={sx} y={fy}>
          <Curve name="Landweber" x={r.singular} y={r.landweber[i]} slot={0} showPoints stale={computed.stale} />
          <Curve
            name="Tikhonov, λ = 1/(ωk)"
            x={r.singular}
            y={r.tikhonov[i]}
            slot={1}
            dashed
            showPoints
            stale={computed.stale}
          />
        </Plot>
        <Plot x={kx} y={ny}>
          <Curve name="error ‖w_k − w*‖" x={later} y={r.error.slice(1)} emphasis stale={computed.stale} />
          <Curve name="Tikhonov error" x={later} y={r.tikhonovError.slice(1)} slot={1} dashed stale={computed.stale} />
          <Curve name="residual ‖y − Xw_k‖" x={later} y={r.residual.slice(1)} slot={2} stale={computed.stale} />
          <Annotation y={r.delta} text="δ = σ√n" dashed muted />
          {r.stop > 0 && <Annotation x={STEPS[r.stop]} text="discrepancy stop" dashed slot={2} />}
          {i > 0 && <Annotation x={STEPS[i]} text={`k = ${STEPS[i]}`} />}
        </Plot>
      </Plots>
    </Figure>
  )
}
