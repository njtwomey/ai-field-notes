import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const Q = toFlat(linspace(0.005, 0.995, 199))

/**
 * Excess expected loss S(q, p) − S(p, p) of reporting probability q for a binary event whose true probability is p,
 * for the log, Brier and spherical losses (proper: minimised at q = p) and the linear loss 1 − q_y (improper).
 */
export function ExpectedScore() {
  const state = useFigureState({
    p: slider(0.02, 0.98, 0.7, { step: 0.01, label: 'true probability p' }),
  })

  const r = useMemo(() => {
    const P = state.p
    const log = (q: number) => -(P * Math.log(q) + (1 - P) * Math.log(1 - q))
    const brier = (q: number) => 2 * (P * (1 - q) ** 2 + (1 - P) * q ** 2)
    const sph = (q: number) => 1 - (P * q + (1 - P) * (1 - q)) / Math.hypot(q, 1 - q)
    const lin = (q: number) => 1 - (P * q + (1 - P) * (1 - q))
    const excess = (f: (q: number) => number) => Q.map((q) => f(q) - f(P))
    return {
      log: excess(log),
      brier: excess(brier),
      sph: excess(sph),
      lin: excess(lin),
      linAt1: lin(P > 0.5 ? 0.995 : 0.005) - lin(P),
    }
  }, [state.p])

  const xAxis = useAxis({ label: 'reported probability q', range: [0, 1] })
  const yAxis = useAxis({ label: 'excess expected loss', range: [-0.5, 1] })
  return (
    <Figure
      title="Expected loss of each possible report"
      state={state}
      caption="The event happens with probability p. Each curve is the expected loss of reporting q minus the expected loss of reporting p honestly. The log, Brier and spherical losses are proper: every curve is non-negative with its minimum at q = p. The linear loss 1 − q_y is improper: its curve dips below zero, rewarding a report of 0 or 1 whichever outcome is more likely."

      readouts={
        <>
          <Readout label="gain from exaggerating, linear loss" value={formatNumber(-r.linAt1)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve name="log loss" x={Q} y={r.log} slot={0} />
        <Curve name="Brier score" x={Q} y={r.brier} slot={1} />
        <Curve name="spherical loss" x={Q} y={r.sph} slot={2} />
        <Curve name="linear loss (improper)" x={Q} y={r.lin} slot={3} dashed />
        <Curve name="q = p" x={[state.p, state.p]} y={[-0.5, 1]} muted dashed />
      </Plot>
    </Figure>
  )
}
