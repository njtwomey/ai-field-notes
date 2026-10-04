import { useMemo, useState } from 'react'
import {
  Button,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
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
  const state = useFigureState({
    ell: slider(0.3, 3, 1, { step: 0.05, label: 'length-scale ℓ' }),
    sf: slider(0.5, 5, 2, { step: 0.1, label: 'signal sd σ_f' }),
  })
  // EP sweeps made: the walk-through position. It counts sweeps, so it keeps its meaning when the inputs change.
  const [sweep, setSweep] = useState(0)

  const r = useMemo(() => {
    const k = seKernel(state.ell, state.sf)
    const states = gpcEp(xs, LABELS, k, SWEEPS)
    return {
      states,
      preds: states.map((s) => gpcEpPredict(xs, k, s.tau, s.nu, XS)),
      laplace: gpcLaplace(xs, LABELS, k, XS),
    }
  }, [xs, state.ell, state.sf])

  const site = r.states[sweep]
  const ep = r.preds[sweep]
  const la = r.laplace.predict
  const band = (m: number[], v: number[], sign: 1 | -1) => m.map((mi, i) => mi + sign * 2 * Math.sqrt(v[i]))
  const pseudo = site.tau
    .map((t, i) => ({ x: xs[i], y: t > 0 ? site.nu[i] / t : NaN }))
    .filter((p) => Number.isFinite(p.y))

  const latent = [
    { name: 'EP mean', x: XS, y: ep.mean, slot: 0 },
    { name: 'EP ± 2 sd', x: XS, y: band(ep.mean, ep.variance, 1), slot: 0, dashed: true },
    { name: 'EP ± 2 sd', x: XS, y: band(ep.mean, ep.variance, -1), slot: 0, dashed: true },
    { name: 'Laplace mean', x: XS, y: la.mean, slot: 1 },
    { name: 'Laplace ± 2 sd', x: XS, y: band(la.mean, la.variance, 1), slot: 1, dashed: true },
    { name: 'Laplace ± 2 sd', x: XS, y: band(la.mean, la.variance, -1), slot: 1, dashed: true },
    { name: 'site means ν̃/τ̃', x: pseudo.map((p) => p.x), y: pseudo.map((p) => p.y), slot: 2 },
  ] as const
  const probability = [
    { name: 'EP', x: XS, y: ep.prob, slot: 0 },
    { name: 'Laplace', x: XS, y: la.prob, slot: 1 },
  ] as const
  const handles: Handle[] = xs.map((x, i) => ({
    kind: 'point',
    at: [x, LABELS[i] > 0 ? 1 : 0],
    label: `x${i + 1}`,
    onDrag: ([nx]) =>
      setXs((cur) => cur.map((v, j) => (j === i ? Math.round(Math.min(5.8, Math.max(-5.8, nx)) * 20) / 20 : v))),
  }))
  const at0 = XS.indexOf(0)

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'latent f', range: F_RANGE })
  const yAxis2 = useAxis({ label: 'p(y = +1)', range: P_RANGE })
  return (
    <Figure
      title="EP for Gaussian process classification, sweep by sweep"
      state={state}
      caption="Top: the latent function f with ±2 sd bands under EP after the chosen number of sweeps, and under the Laplace approximation. The dots are the sites' means ν̃/τ̃: EP's posterior is exactly GP regression on these pseudo-targets with noise variances 1/τ̃. Bottom: the predictive probability of class +1. The labelled points sit at 0 and 1; drag them sideways. EP's latent function is larger in magnitude than Laplace's, and its probabilities more confident."
      controls={
        <>
          <Player
            value={sweep}
            onChange={setSweep}
            count={SWEEPS + 1}
            label="EP sweeps"
            format={(k) => `${k} of ${SWEEPS}`}
          />
          <Button variant="outline" size="sm" onClick={() => setXs(INITIAL)}>
            Reset points
          </Button>
        </>
      }
      readouts={
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
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...latent[0]} />
        <Curve {...latent[1]} />
        <Curve {...latent[2]} />
        <Curve {...latent[3]} />
        <Curve {...latent[4]} />
        <Curve {...latent[5]} />
        <Points {...latent[6]} />
      </Plot>
      <Plot x={xAxis} y={yAxis2} height={240}>
        <Curve {...probability[0]} />
        <Curve {...probability[1]} />
        {handles.map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
