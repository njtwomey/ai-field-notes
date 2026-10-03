import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
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
  const [t0, setT0] = useState(5)
  const [tInf, setTInf] = useState(1)
  const [lambda, setLambda] = useState(3)
  const days = useParam(14, { min: 1, max: MAX_DAYS, step: 1 })

  const series = useMemo<XYSeries[]>(() => {
    const ts = Array.from({ length: MAX_DAYS * 4 + 1 }, (_, i) => i / 4)
    const later = ts.filter((t) => t > 0)
    return [
      {
        name: 'effect on day t',
        type: 'line',
        x: ts,
        y: ts.map((t) => tInf + (t0 - tInf) * Math.exp(-t / lambda)),
        slot: 0,
      },
      {
        name: 'estimate from days 0 to t',
        type: 'line',
        x: later,
        y: later.map((t) => averageEffect(t0, tInf, lambda, t)),
        slot: 1,
      },
      { name: 'long-run effect', type: 'line', x: [0, MAX_DAYS], y: [tInf, tInf], emphasis: true, dashed: true },
    ]
  }, [t0, tInf, lambda])

  const estimate = averageEffect(t0, tInf, lambda, days.value)
  const handles: Handle[] = [{ kind: 'x', at: days.value, label: 'T', onDrag: (x) => days.set(Math.round(x)) }]

  return (
    <Interactive
      title="A decaying effect and the estimate it produces"
      caption="The treatment effect on a user t days after first exposure moves from τ₀ to the long-run value τ∞ with time constant λ. With τ₀ above τ∞ it is a novelty effect; below, a primacy effect. An experiment that runs for T days and averages over them reports the second curve, which approaches τ∞ only slowly, as λ/T. Drag the line labelled T, or use its slider, to change the experiment's length."
      controls={
        <>
          <ParamSlider label="experiment length T (days)" param={days} />
          <ParamSlider label="initial effect τ₀ (%)" value={t0} onChange={setT0} min={-5} max={10} step={0.5} />
          <ParamSlider label="long-run effect τ∞ (%)" value={tInf} onChange={setTInf} min={-5} max={5} step={0.5} />
          <ParamSlider
            label="time constant λ (days)"
            value={lambda}
            onChange={setLambda}
            min={0.5}
            max={14}
            step={0.5}
          />
        </>
      }
      readout={
        <>
          <Readout label="estimate after T days (%)" value={formatNumber(estimate)} />
          <Readout label="long-run effect (%)" value={formatNumber(tInf)} />
          <Readout label="bias (points)" value={formatNumber(estimate - tInf)} />
        </>
      }
    >
      <XYChart
        height={300}
        series={series}
        xRange={[0, MAX_DAYS]}
        xLabel="days since first exposure"
        yLabel="treatment effect (%)"
        handles={handles}
      />
    </Interactive>
  )
}
