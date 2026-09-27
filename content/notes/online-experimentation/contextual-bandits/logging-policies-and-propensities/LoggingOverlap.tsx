import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

const EPS = Array.from({ length: 200 }, (_, i) => 0.005 + (0.995 * i) / 199)
const LOGS = 10000

/**
 * Second moment of the importance weight for an ε-greedy logging policy over K arms and a deterministic target that
 * disagrees with the logging policy's greedy arm on a fraction ρ of contexts.
 */
const secondMoment = (eps: number, k: number, rho: number) => (1 - rho) / (1 - eps + eps / k) + (rho * k) / eps

/** Effective sample size per logged row, 1 / E[w²], against the logging policy's exploration rate. */
export function LoggingOverlap() {
  const eps = useParam(0.1, { min: 0.005, max: 1, step: 0.005 })
  const rho = useParam(0.2, { min: 0, max: 1, step: 0.01 })
  const arms = useParam(4, { min: 2, max: 20, step: 1 })
  const k = arms.value
  const r = rho.value

  const series = useMemo((): XYSeries[] => {
    const curve = (value: number) => EPS.map((e) => 1 / secondMoment(e, k, value))
    return [
      { name: `target disagrees on ${Math.round(100 * r)}% of contexts`, type: 'line', x: EPS, y: curve(r), slot: 0 },
      { name: 'target = logging greedy arm', type: 'line', x: EPS, y: curve(0), muted: true },
      { name: 'target always disagrees', type: 'line', x: EPS, y: curve(1), muted: true },
    ]
  }, [k, r])

  const m2 = secondMoment(eps.value, k, r)
  return (
    <Interactive
      title="What exploration buys for later evaluation"
      caption="An ε-greedy logging policy over K arms: it plays its greedy arm with probability 1 − ε + ε/K and each other arm with probability ε/K. A deterministic target policy agrees with the greedy arm on some contexts and picks another arm on a fraction ρ of them. The chart shows the effective sample size per logged row, 1/E[w²], as ε varies; drag the guide to set ε. Where the target disagrees, each matching row carries weight K/ε, so a small ε makes the log almost useless for evaluating a different policy. The muted curves are the extremes ρ = 0 and ρ = 1."
      controls={
        <>
          <ParamSlider label="logging exploration ε" param={eps} format={(v) => v.toFixed(3)} />
          <ParamSlider label="disagreement ρ" param={rho} />
          <ParamSlider label="arms K" param={arms} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="weight where the target agrees" value={formatNumber(1 / (1 - eps.value + eps.value / k))} />
          <Readout label="weight where it disagrees, K/ε" value={formatNumber(k / eps.value)} />
          <Readout label="E[w²]" value={formatNumber(m2)} />
          <Readout
            label={`effective size of ${LOGS.toLocaleString()} logs`}
            value={Math.round(LOGS / m2).toLocaleString()}
          />
          <Readout label="IPS standard error at most" value={formatNumber(Math.sqrt(m2 / LOGS))} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="logging exploration ε"
        yLabel="effective sample size per row"
        xRange={[0, 1]}
        yRange={[0, 1]}
        handles={[{ kind: 'x', at: eps.value, label: 'ε', onDrag: (x) => eps.set(x) }]}
      />
    </Interactive>
  )
}
