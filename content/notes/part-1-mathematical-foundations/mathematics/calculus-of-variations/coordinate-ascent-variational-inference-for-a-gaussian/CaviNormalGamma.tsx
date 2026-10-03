import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  StepControls,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
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

/** Half-steps of coordinate ascent computed ahead of time; Step and Run reveal them. */
const MAX_SWEEPS = 25

/**
 * CAVI for a Gaussian with unknown mean and precision. Each sweep updates q(τ) and then q(μ); the ELBO rises
 * monotonically towards log p(x), and the factorised q ends up narrower in μ than the exact Normal-Gamma posterior.
 */
export function CaviNormalGamma() {
  const n = useParam(10, { min: 2, max: 50, step: 1 })
  const trueMu = useParam(1, { min: -2, max: 2, step: 0.1 })
  const trueTau = useParam(2, { min: 0.25, max: 4, step: 0.25 })
  const lambda0 = useParam(1, { min: 0.1, max: 10, step: 0.1 })
  const init = useParam(0.1, { min: 0.05, max: 5, step: 0.05 })
  const seed = useParam(1, { min: 0, max: 20, step: 1 })

  const run = useMemo(() => {
    const g = rng(seed.value)
    const x = Array.from({ length: n.value }, () => trueMu.value + g.normal() / Math.sqrt(trueTau.value))
    const s = statistics(x)
    const prior: Prior = { mu0: 0, lambda0: lambda0.value, a0: 1, b0: 1 }
    // Start q(τ) with mean equal to the initial guess, then set q(μ) from it.
    const a = prior.a0 + (s.n + 1) / 2
    let q: Factors = updateMu({ mu: 0, lambda: 1, a, b: a / init.value }, s, prior)
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
  }, [n.value, trueMu.value, trueTau.value, lambda0.value, init.value, seed.value])

  // The visible half-step, reset whenever the problem changes.
  const key = `${n.value}|${trueMu.value}|${trueTau.value}|${lambda0.value}|${init.value}|${seed.value}`
  const [cursor, setCursor] = useState({ key, at: 0 })
  const at = cursor.key === key ? cursor.at : 0
  const q = run.states[at]
  const done = at >= run.converged
  const go = (next: number) => setCursor({ key, at: Math.min(next, run.converged) })

  const contours: XYSeries[] = useMemo(() => {
    const e = run.exact
    const exactMode: [number, number] = [e.mu, (e.a - 0.5) / e.b]
    const exactScale: [number, number] = [1 / Math.sqrt(e.lambda * exactMode[1]), Math.sqrt(e.a) / e.b]
    const qMode: [number, number] = [q.mu, (q.a - 1) / q.b]
    const qScale: [number, number] = [1 / Math.sqrt(q.lambda), Math.sqrt(q.a) / q.b]
    return [
      ...[1, 2].map((k): XYSeries => ({
        name: 'exact posterior',
        type: 'line',
        ...contour(logExact(e), exactMode, exactScale, k),
        slot: 0,
        dashed: k === 2,
      })),
      ...[1, 2].map((k): XYSeries => ({
        name: 'mean-field q',
        type: 'line',
        ...contour(logQ(q), qMode, qScale, k),
        slot: 1,
        dashed: k === 2,
      })),
      { name: 'true (μ, τ)', type: 'scatter', x: [trueMu.value], y: [trueTau.value], emphasis: true },
    ]
  }, [run.exact, q, trueMu.value, trueTau.value])

  const trace: XYSeries[] = useMemo(() => {
    const steps = run.elbos.slice(0, at + 1)
    const xs = steps.map((_, i) => i / 2)
    return [
      { name: 'ELBO', type: 'line', x: xs, y: steps, slot: 1 },
      { name: 'ELBO at each update', type: 'scatter', x: xs, y: steps, slot: 1 },
      {
        name: 'log p(x)',
        type: 'line',
        x: [0, Math.max(at / 2, 1)],
        y: [run.evidence, run.evidence],
        slot: 0,
        dashed: true,
      },
    ]
  }, [run.elbos, run.evidence, at])

  const e = run.exact
  const exactSdMu = Math.sqrt(e.b / ((e.a - 1) * e.lambda))
  return (
    <Interactive
      title="Coordinate ascent for a Gaussian's mean and precision"
      caption="Left: contours of the exact Normal-Gamma posterior (blue) and of the factorised approximation q(μ)q(τ) (orange) over the mean μ and precision τ; solid and dashed are one and two 'standard deviations' from each mode. Right: the ELBO after each half-update, rising to just below log p(x). Step runs one sweep, updating q(τ) and then q(μ). The initial guess for E[τ] starts far off so the first sweeps are visible. At convergence q cannot tilt with the posterior, so it is narrower in μ."
      controls={
        <>
          <ParamSlider label="observations N" param={n} />
          <ParamSlider label="true mean μ" param={trueMu} />
          <ParamSlider label="true precision τ" param={trueTau} />
          <ParamSlider label="prior strength λ₀" param={lambda0} />
          <ParamSlider label="initial guess E[τ]" param={init} />
          <ParamSlider label="seed" param={seed} />
          <StepControls onStep={() => go(at + 2)} onRun={() => go(run.converged)} onReset={() => go(0)} done={done} />
        </>
      }
      readout={
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
        <XYChart series={contours} xLabel="mean μ" yLabel="precision τ" height={340} />
        <XYChart series={trace} xLabel="sweeps" yLabel="ELBO" height={340} />
      </div>
    </Interactive>
  )
}
