import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const MAX_DAYS = 56

/** The effect decays exponentially from τ₀ to τ∞ with time constant λ; its average over [0, T] is the estimate. */
function averageEffect(t0: number, tInf: number, lambda: number, T: number): number {
  return tInf + ((t0 - tInf) * lambda * (1 - Math.exp(-T / lambda))) / T
}

/**
 * A novelty (τ₀ > τ∞) or primacy (τ₀ < τ∞) effect, with every user first exposed on day 0. The experiment's estimate
 * after T days is the average of the daily effect over those days.
 */
export function NoveltyDecay() {
  const state = useFigureState({
    days: int(14, { min: 1, max: MAX_DAYS, step: 1, label: 'experiment length T (days)' }),
    t0: float(5, { min: -5, max: 10, step: 0.5, label: 'initial effect τ₀ (%)' }),
    tInf: float(1, { min: -5, max: 5, step: 0.5, label: 'long-run effect τ∞ (%)' }),
    lambda: slider(0.5, 14, 3, { step: 0.5, label: 'time constant λ (days)' }),
  })

  const series = useMemo(() => {
    const ts = Array.from({ length: MAX_DAYS * 4 + 1 }, (_, i) => i / 4)
    const later = ts.filter((t) => t > 0)
    return [
      {
        name: 'effect on day t',
        x: ts,
        y: ts.map((t) => state.tInf + (state.t0 - state.tInf) * Math.exp(-t / state.lambda)),
        slot: 0,
      },
      {
        name: 'estimate from days 0 to t',
        x: later,
        y: later.map((t) => averageEffect(state.t0, state.tInf, state.lambda, t)),
        slot: 1,
      },
      { name: 'long-run effect', x: [0, MAX_DAYS], y: [state.tInf, state.tInf], emphasis: true, dashed: true },
    ] as const
  }, [state.t0, state.tInf, state.lambda])

  const estimate = averageEffect(state.t0, state.tInf, state.lambda, state.days)

  const xAxis = useAxis({ label: 'days since first exposure', range: [0, MAX_DAYS] })
  const yAxis = useAxis({ label: 'treatment effect (%)', hold: 'union' })
  return (
    <Figure
      title="A decaying effect and the estimate it produces"
      state={state}
      caption="The treatment effect on a user t days after first exposure moves from τ₀ to the long-run value τ∞ with time constant λ. With τ₀ above τ∞ it is a novelty effect; below, a primacy effect. An experiment that runs for T days and averages over them reports the second curve, which approaches τ∞ only slowly, as λ/T. Drag the line labelled T, or use its slider, to change the experiment's length."

      readouts={
        <>
          <Readout label="estimate after T days (%)" value={formatNumber(estimate)} />
          <Readout label="long-run effect (%)" value={formatNumber(state.tInf)} />
          <Readout label="bias (points)" value={formatNumber(estimate - state.tInf)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle kind="x" at={state.days} label="T" onDrag={(x) => state.set('days', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
