import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { logGamma } from 'aifn/numerics/special'

const logBeta = (a: number, b: number) => logGamma(a) + logGamma(b) - logGamma(a + b)
/** log p(k successes in a given order out of n | Beta(1, 1) rate) = log B(1 + k, 1 + n − k). */
const logEvidence = (k: number, n: number) => logBeta(1 + k, 1 + n - k)

/** Bayesian model selection through a gate: the posterior probability that the treatment has an effect. */
export function TrialEvidence() {
  const state = useFigureState({
    nc: int(5, { ge: 1, le: 60, suggestions: [5, 10, 20, 50], label: 'control group size' }),
    kc: int(1, { ge: 0, le: 60, label: 'controls recovered (at most the group size)' }),
    nt: int(5, { ge: 1, le: 60, suggestions: [5, 10, 20, 50], label: 'treated group size' }),
    kt: int(4, { ge: 0, le: 60, label: 'treated recovered (at most the group size)' }),
  })
  const nc = state.nc
  const nt = state.nt
  const c = Math.min(state.kc, nc)
  const t = Math.min(state.kt, nt)

  const effect = logEvidence(c, nc) + logEvidence(t, nt)
  const none = logEvidence(c + t, nc + nt)
  const pEffect = 1 / (1 + Math.exp(none - effect))

  const series = useMemo(() => {
    const x = Array.from({ length: 101 }, (_, i) => i / 100)
    const dens = (k: number, n: number) =>
      x.map((r) =>
        Math.exp(
          k * Math.log(Math.max(r, 1e-12)) + (n - k) * Math.log(Math.max(1 - r, 1e-12)) - logBeta(1 + k, 1 + n - k),
        ),
      )
    return [
      { name: 'control rate ρc (if effect)', x, y: dens(c, nc), slot: 0 },
      { name: 'treated rate ρt (if effect)', x, y: dens(t, nt), slot: 1 },
      {
        name: 'shared rate ρ (if no effect)',
        x,
        y: dens(c + t, nc + nt),
        slot: 2,
        dashed: true,
      },
    ] as const
  }, [c, t, nc, nt])

  const xAxis = useAxis({ label: 'recovery rate', range: [0, 1] })
  const yAxis = useAxis({ label: 'posterior density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Does the treatment work?"
      state={state}
      caption="Set the size of each group and how many recovered. The gate's message to the selector is each sub-model's evidence, a ratio of Beta functions; the posterior probability of an effect follows from a prior of 0.5. The chart shows the posterior recovery rates under each sub-model. The starting values are the Infer.NET tutorial's: 1 of 5 controls and 4 of 5 treated patients recovered."
      readouts={
        <>
          <Readout label="recovered" value={`${c} of ${nc} controls, ${t} of ${nt} treated`} />
          <Readout label="P(effect | data)" value={formatNumber(pEffect)} />
          <Readout label="log Bayes factor" value={formatNumber(effect - none)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
