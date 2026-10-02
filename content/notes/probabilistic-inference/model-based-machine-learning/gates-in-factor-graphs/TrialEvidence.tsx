import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { logGamma } from '@/lib/math/special'

const logBeta = (a: number, b: number) => logGamma(a) + logGamma(b) - logGamma(a + b)
/** log p(k successes in a given order out of n | Beta(1, 1) rate) = log B(1 + k, 1 + n − k). */
const logEvidence = (k: number, n: number) => logBeta(1 + k, 1 + n - k)

/** Bayesian model selection through a gate: the posterior probability that the treatment has an effect. */
export function TrialEvidence() {
  const nc = useParam(5, { min: 1, max: 60, step: 1 })
  const kc = useParam(1, { min: 0, max: 60, step: 1 })
  const nt = useParam(5, { min: 1, max: 60, step: 1 })
  const kt = useParam(4, { min: 0, max: 60, step: 1 })
  const c = Math.min(kc.value, nc.value)
  const t = Math.min(kt.value, nt.value)

  const effect = logEvidence(c, nc.value) + logEvidence(t, nt.value)
  const none = logEvidence(c + t, nc.value + nt.value)
  const pEffect = 1 / (1 + Math.exp(none - effect))

  const series = useMemo((): XYSeries[] => {
    const x = Array.from({ length: 101 }, (_, i) => i / 100)
    const dens = (k: number, n: number) =>
      x.map((r) =>
        Math.exp(
          k * Math.log(Math.max(r, 1e-12)) + (n - k) * Math.log(Math.max(1 - r, 1e-12)) - logBeta(1 + k, 1 + n - k),
        ),
      )
    return [
      { name: 'control rate ρc (if effect)', type: 'line', x, y: dens(c, nc.value), slot: 0 },
      { name: 'treated rate ρt (if effect)', type: 'line', x, y: dens(t, nt.value), slot: 1 },
      {
        name: 'shared rate ρ (if no effect)',
        type: 'line',
        x,
        y: dens(c + t, nc.value + nt.value),
        slot: 2,
        dashed: true,
      },
    ]
  }, [c, t, nc.value, nt.value])

  return (
    <Interactive
      title="Does the treatment work?"
      caption="Set the size of each group and how many recovered. The gate's message to the selector is each sub-model's evidence, a ratio of Beta functions; the posterior probability of an effect follows from a prior of 0.5. The chart shows the posterior recovery rates under each sub-model. The starting values are the Infer.NET tutorial's: 1 of 5 controls and 4 of 5 treated patients recovered."
      controls={
        <>
          <ParamSlider label="control group size" param={nc} format={(v) => String(v)} />
          <ParamSlider label="controls recovered" param={kc} format={(v) => String(Math.min(v, nc.value))} />
          <ParamSlider label="treated group size" param={nt} format={(v) => String(v)} />
          <ParamSlider label="treated recovered" param={kt} format={(v) => String(Math.min(v, nt.value))} />
        </>
      }
      readout={
        <>
          <Readout label="P(effect | data)" value={formatNumber(pEffect)} />
          <Readout label="log Bayes factor" value={formatNumber(effect - none)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="recovery rate"
        yLabel="posterior density"
        xRange={[0, 1]}
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}
