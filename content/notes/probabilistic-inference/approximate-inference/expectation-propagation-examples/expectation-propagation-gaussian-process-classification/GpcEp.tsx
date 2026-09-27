import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { grid } from '../_shared/ep'
import { gpcEp, gpcEpPredict, gpcLaplace, seKernel } from '../_shared/gpc'

const XS = grid(-6, 6, 121)
const X_RANGE: [number, number] = [-6, 6]
const F_RANGE: [number | undefined, number | undefined] = [-6, 6]
const P_RANGE: [number | undefined, number | undefined] = [-0.05, 1.05]
const INITIAL = [-4, -3.2, -2.5, -1.8, -0.6, 0.3, 1.1, 2, 2.9, 3.6]
const LABELS = [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1]
const SWEEPS = 8

/**
 * EP for 1-D GP classification, sweep by sweep, against the Laplace approximation. The sites' pseudo-observations
 * ν̃/τ̃ show EP as GP regression on made-up targets with noise variance 1/τ̃.
 */
export function GpcEp() {
  const [xs, setXs] = useState(INITIAL)
  const sweep = useParam(1, { min: 0, max: SWEEPS, step: 1 })
  const ell = useParam(1, { min: 0.3, max: 3, step: 0.05 })
  const sf = useParam(2, { min: 0.5, max: 5, step: 0.1 })

  const r = useMemo(() => {
    const k = seKernel(ell.value, sf.value)
    const states = gpcEp(xs, LABELS, k, SWEEPS)
    return {
      states,
      preds: states.map((s) => gpcEpPredict(xs, k, s.tau, s.nu, XS)),
      laplace: gpcLaplace(xs, LABELS, k, XS),
    }
  }, [xs, ell.value, sf.value])

  const state = r.states[sweep.value]
  const ep = r.preds[sweep.value]
  const la = r.laplace.predict
  const band = (m: number[], v: number[], sign: 1 | -1) => m.map((mi, i) => mi + sign * 2 * Math.sqrt(v[i]))
  const pseudo = state.tau
    .map((t, i) => ({ x: xs[i], y: t > 0 ? state.nu[i] / t : NaN }))
    .filter((p) => Number.isFinite(p.y))

  const latent: XYSeries[] = [
    { name: 'EP mean', type: 'line', x: XS, y: ep.mean, slot: 0 },
    { name: 'EP ± 2 sd', type: 'line', x: XS, y: band(ep.mean, ep.variance, 1), slot: 0, dashed: true },
    { name: 'EP ± 2 sd', type: 'line', x: XS, y: band(ep.mean, ep.variance, -1), slot: 0, dashed: true },
    { name: 'Laplace mean', type: 'line', x: XS, y: la.mean, slot: 1 },
    { name: 'Laplace ± 2 sd', type: 'line', x: XS, y: band(la.mean, la.variance, 1), slot: 1, dashed: true },
    { name: 'Laplace ± 2 sd', type: 'line', x: XS, y: band(la.mean, la.variance, -1), slot: 1, dashed: true },
    { name: 'site means ν̃/τ̃', type: 'scatter', x: pseudo.map((p) => p.x), y: pseudo.map((p) => p.y), slot: 2 },
  ]
  const probability: XYSeries[] = [
    { name: 'EP', type: 'line', x: XS, y: ep.prob, slot: 0 },
    { name: 'Laplace', type: 'line', x: XS, y: la.prob, slot: 1 },
  ]
  const handles: Handle[] = xs.map((x, i) => ({
    kind: 'point',
    at: [x, LABELS[i] > 0 ? 1 : 0],
    label: `x${i + 1}`,
    onDrag: ([nx]) =>
      setXs((cur) => cur.map((v, j) => (j === i ? Math.round(Math.min(5.8, Math.max(-5.8, nx)) * 20) / 20 : v))),
  }))
  const at0 = XS.indexOf(0)

  return (
    <Interactive
      title="EP for Gaussian process classification, sweep by sweep"
      caption="Top: the latent function f with ±2 sd bands under EP after the chosen number of sweeps, and under the Laplace approximation. The dots are the sites' means ν̃/τ̃: EP's posterior is exactly GP regression on these pseudo-targets with noise variances 1/τ̃. Bottom: the predictive probability of class +1. The labelled points sit at 0 and 1; drag them sideways. EP's latent function is larger in magnitude than Laplace's, and its probabilities more confident."
      controls={
        <>
          <ParamSlider label="EP sweeps" param={sweep} withArrows format={(v) => String(v)} />
          <ParamSlider label="length-scale ℓ" param={ell} />
          <ParamSlider label="signal sd σ_f" param={sf} />
          <ParamButton onClick={() => setXs(INITIAL)}>Reset points</ParamButton>
        </>
      }
      readout={
        <>
          <Readout
            label="f(0): EP mean, sd"
            value={`${formatNumber(ep.mean[at0])}, ${formatNumber(Math.sqrt(ep.variance[at0]))}`}
          />
          <Readout
            label="Laplace mean, sd"
            value={`${formatNumber(la.mean[at0])}, ${formatNumber(Math.sqrt(la.variance[at0]))}`}
          />
          <Readout
            label="p(y = +1 | x = 0): EP, Laplace"
            value={`${formatNumber(ep.prob[at0])}, ${formatNumber(la.prob[at0])}`}
          />
        </>
      }
    >
      <XYChart series={latent} xLabel="x" yLabel="latent f" xRange={X_RANGE} yRange={F_RANGE} height={300} />
      <XYChart
        series={probability}
        xLabel="x"
        yLabel="p(y = +1)"
        xRange={X_RANGE}
        yRange={P_RANGE}
        handles={handles}
        height={240}
      />
    </Interactive>
  )
}
