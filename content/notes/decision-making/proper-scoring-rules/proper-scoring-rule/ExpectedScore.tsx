import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { linspace } from '@/lib/math'

const Q = linspace(0.005, 0.995, 199)

/**
 * Excess expected loss S(q, p) − S(p, p) of reporting probability q for a binary event whose true probability is p,
 * for the log, Brier and spherical losses (proper: minimised at q = p) and the linear loss 1 − q_y (improper).
 */
export function ExpectedScore() {
  const p = useParam(0.7, { min: 0.02, max: 0.98, step: 0.01 })

  const r = useMemo(() => {
    const P = p.value
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
  }, [p.value])

  return (
    <Interactive
      title="Expected loss of each possible report"
      caption="The event happens with probability p. Each curve is the expected loss of reporting q minus the expected loss of reporting p honestly. The log, Brier and spherical losses are proper: every curve is non-negative with its minimum at q = p. The linear loss 1 − q_y is improper: its curve dips below zero, rewarding a report of 0 or 1 whichever outcome is more likely."
      controls={<ParamSlider label="true probability p" param={p} />}
      readout={
        <>
          <Readout label="gain from exaggerating, linear loss" value={formatNumber(-r.linAt1)} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="reported probability q"
        yLabel="excess expected loss"
        xRange={[0, 1]}
        yRange={[-0.5, 1]}
        series={[
          { name: 'log loss', type: 'line', x: Q, y: r.log, slot: 0 },
          { name: 'Brier score', type: 'line', x: Q, y: r.brier, slot: 1 },
          { name: 'spherical loss', type: 'line', x: Q, y: r.sph, slot: 2 },
          { name: 'linear loss (improper)', type: 'line', x: Q, y: r.lin, slot: 3, dashed: true },
          { name: 'q = p', type: 'line', x: [p.value, p.value], y: [-0.5, 1], muted: true, dashed: true },
        ]}
      />
    </Interactive>
  )
}
