import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Player,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  contour,
  elbo,
  exactPosterior,
  logEvidence,
  logExact,
  logQ,
  statistics,
  updateMu,
  updateTau,
  type Factors,
  type Prior,
} from './cavi'
import { normal, stream } from 'aifn/foundation/random'

/** Half-steps of coordinate ascent computed ahead of time; the player walks through them a sweep at a time. */
const MAX_SWEEPS = 25

/**
 * CAVI for a Gaussian with unknown mean and precision. Each sweep updates q(τ) and then q(μ); the ELBO rises
 * monotonically towards log p(x), and the factorised q ends up narrower in μ than the exact Normal-Gamma posterior.
 */
export function CaviNormalGamma() {
  const state = useFigureState({
    n: int(10, { min: 2, max: 50, step: 1, label: 'observations N' }),
    trueMu: float(1, { min: -2, max: 2, step: 0.1, label: 'true mean μ' }),
    trueTau: float(2, { min: 0.25, max: 4, step: 0.25, label: 'true precision τ' }),
    lambda0: float(1, { min: 0.1, max: 10, step: 0.1, label: 'prior strength λ₀' }),
    init: float(0.1, { min: 0.05, max: 5, step: 0.05, label: 'initial guess E[τ]' }),
    seed: int(1, { min: 0, max: 20, step: 1, label: 'seed' }),
  })

  const run = useMemo(() => {
    const g = stream(state.seed)
    const x = Array.from({ length: state.n }, () => state.trueMu + normal(g) / Math.sqrt(state.trueTau))
    const s = statistics(x)
    const prior: Prior = { mu0: 0, lambda0: state.lambda0, a0: 1, b0: 1 }
    // Start q(τ) with mean equal to the initial guess, then set q(μ) from it.
    const a = prior.a0 + (s.n + 1) / 2
    let q: Factors = updateMu({ mu: 0, lambda: 1, a, b: a / state.init }, s, prior)
    const states: Factors[] = [q]
    for (let i = 0; i < MAX_SWEEPS; i++) {
      q = updateTau(q, s, prior)
      states.push(q)
      q = updateMu(q, s, prior)
      states.push(q)
    }
    const elbos = states.map((st) => elbo(st, s, prior))
    // First half-step after which the ELBO no longer moves: the end of Run.
    const converged = elbos.findIndex((e, i) => i > 0 && Math.abs(e - elbos[i - 1]) < 1e-9)
    return {
      states,
      elbos,
      converged: converged < 0 ? states.length - 1 : converged,
      exact: exactPosterior(s, prior),
      evidence: logEvidence(s, prior),
    }
  }, [state.n, state.trueMu, state.trueTau, state.lambda0, state.init, state.seed])

  // The visible half-step, reset whenever the problem changes.
  const key = `${state.n}|${state.trueMu}|${state.trueTau}|${state.lambda0}|${state.init}|${state.seed}`
  const [cursor, setCursor] = useState({ key, sweep: 0 })
  const sweep = cursor.key === key ? cursor.sweep : 0
  const sweeps = Math.ceil(run.converged / 2) + 1
  const at = Math.min(2 * sweep, run.converged)
  const q = run.states[at]

  const contours: SeriesSpec[] = useMemo(() => {
    const e = run.exact
    const exactMode: [number, number] = [e.mu, (e.a - 0.5) / e.b]
    const exactScale: [number, number] = [1 / Math.sqrt(e.lambda * exactMode[1]), Math.sqrt(e.a) / e.b]
    const qMode: [number, number] = [q.mu, (q.a - 1) / q.b]
    const qScale: [number, number] = [1 / Math.sqrt(q.lambda), Math.sqrt(q.a) / q.b]
    return [
      ...[1, 2].map((k): SeriesSpec => ({
        name: 'exact posterior',
        type: 'line',
        ...contour(logExact(e), exactMode, exactScale, k),
        slot: 0,
        dashed: k === 2,
      })),
      ...[1, 2].map((k): SeriesSpec => ({
        name: 'mean-field q',
        type: 'line',
        ...contour(logQ(q), qMode, qScale, k),
        slot: 1,
        dashed: k === 2,
      })),
      { name: 'true (μ, τ)', type: 'scatter', x: [state.trueMu], y: [state.trueTau], emphasis: true },
    ]
  }, [run.exact, q, state.trueMu, state.trueTau])

  const trace = useMemo(() => {
    const steps = run.elbos.slice(0, at + 1)
    const xs = steps.map((_, i) => i / 2)
    return [
      { name: 'ELBO', x: xs, y: steps, slot: 1 },
      { name: 'ELBO at each update', x: xs, y: steps, slot: 1 },
      {
        name: 'log p(x)',
        x: [0, Math.max(at / 2, 1)],
        y: [run.evidence, run.evidence],
        slot: 0,
        dashed: true,
      },
    ] as const
  }, [run.elbos, run.evidence, at])

  const e = run.exact
  const exactSdMu = Math.sqrt(e.b / ((e.a - 1) * e.lambda))
  const xAxis = useAxis({ label: 'mean μ', hold: 'union' })
  const yAxis = useAxis({ label: 'precision τ', hold: 'union' })
  const xAxis2 = useAxis({ label: 'sweeps', hold: 'union' })
  const yAxis2 = useAxis({ label: 'ELBO', hold: 'union' })
  return (
    <Figure
      title="Coordinate ascent for a Gaussian's mean and precision"
      state={state}
      caption="Left: contours of the exact Normal-Gamma posterior (blue) and of the factorised approximation q(μ)q(τ) (orange) over the mean μ and precision τ; solid and dashed are one and two 'standard deviations' from each mode. Right: the ELBO after each half-update, rising to just below log p(x). Each step of the player is one sweep, updating q(τ) and then q(μ). The initial guess for E[τ] starts far off so the first sweeps are visible. At convergence q cannot tilt with the posterior, so it is narrower in μ."
      controls={<Player value={sweep} onChange={(v) => setCursor({ key, sweep: v })} count={sweeps} label="sweep" />}
      readouts={
        <>
          <Readout label="sweeps" value={formatNumber(at / 2)} />
          <Readout label="ELBO" value={formatNumber(run.elbos[at])} />
          <Readout label="log p(x)" value={formatNumber(run.evidence)} />
          <Readout label="gap KL(q ‖ posterior)" value={formatNumber(run.evidence - run.elbos[at])} />
          <Readout label="E[τ]: q vs exact" value={`${formatNumber(q.a / q.b)} vs ${formatNumber(e.a / e.b)}`} />
          <Readout
            label="sd of μ: q vs exact"
            value={`${formatNumber(1 / Math.sqrt(q.lambda))} vs ${formatNumber(exactSdMu)}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          {seriesLayers(contours)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve {...trace[0]} />
          <Points {...trace[1]} />
          <Curve {...trace[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
