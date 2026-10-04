import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

const EPS = Array.from({ length: 200 }, (_, i) => 0.005 + (0.995 * i) / 199)
const LOGS = 10000

/**
 * Second moment of the importance weight for an ε-greedy logging policy over K arms and a deterministic target that
 * disagrees with the logging policy's greedy arm on a fraction ρ of contexts.
 */
const secondMoment = (eps: number, k: number, rho: number) => (1 - rho) / (1 - eps + eps / k) + (rho * k) / eps

/** Effective sample size per logged row, 1 / E[w²], against the logging policy's exploration rate. */
export function LoggingOverlap() {
  const state = useFigureState({
    eps: float(0.1, { min: 0.005, max: 1, step: 0.005, label: 'logging exploration ε', format: (v) => v.toFixed(3) }),
    rho: float(0.2, { min: 0, max: 1, step: 0.01, label: 'disagreement ρ' }),
    arms: int(4, { min: 2, max: 20, step: 1, label: 'arms K', format: (v) => String(v) }),
  })
  const k = state.arms
  const r = state.rho

  const series = useMemo(() => {
    const curve = (value: number) => EPS.map((e) => 1 / secondMoment(e, k, value))
    return [
      { name: `target disagrees on ${Math.round(100 * r)}% of contexts`, x: EPS, y: curve(r), slot: 0 },
      { name: 'target = logging greedy arm', x: EPS, y: curve(0), muted: true },
      { name: 'target always disagrees', x: EPS, y: curve(1), muted: true },
    ] as const
  }, [k, r])

  const m2 = secondMoment(state.eps, k, r)
  const xAxis = useAxis({ label: 'logging exploration ε', range: [0, 1] })
  const yAxis = useAxis({ label: 'effective sample size per row', range: [0, 1] })
  return (
    <Figure
      title="What exploration buys for later evaluation"
      state={state}
      caption="An ε-greedy logging policy over K arms: it plays its greedy arm with probability 1 − ε + ε/K and each other arm with probability ε/K. A deterministic target policy agrees with the greedy arm on some contexts and picks another arm on a fraction ρ of them. The chart shows the effective sample size per logged row, 1/E[w²], as ε varies; drag the guide to set ε. Where the target disagrees, each matching row carries weight K/ε, so a small ε makes the log almost useless for evaluating a different policy. The muted curves are the extremes ρ = 0 and ρ = 1."

      readouts={
        <>
          <Readout label="weight where the target agrees" value={formatNumber(1 / (1 - state.eps + state.eps / k))} />
          <Readout label="weight where it disagrees, K/ε" value={formatNumber(k / state.eps)} />
          <Readout label="E[w²]" value={formatNumber(m2)} />
          <Readout
            label={`effective size of ${LOGS.toLocaleString()} logs`}
            value={Math.round(LOGS / m2).toLocaleString()}
          />
          <Readout label="IPS standard error at most" value={formatNumber(Math.sqrt(m2 / LOGS))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('eps', { label: 'ε' })} />
      </Plot>
    </Figure>
  )
}
