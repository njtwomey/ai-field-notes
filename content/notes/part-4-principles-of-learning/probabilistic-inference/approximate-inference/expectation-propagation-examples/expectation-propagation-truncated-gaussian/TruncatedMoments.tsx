import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { grid, logNormalCdf, normalLogPdf, normalPdf, probitTilted, stepTilted, toNat, vFn, wFn } from '../_shared/ep'

const XS = grid(-6, 8, 351)
const TS = grid(-4, 4, 161)
const X_RANGE: [number, number] = [-6, 8]
const T_RANGE: [number, number] = [-4, 4]
const Y_DENSITY: [number | undefined, number | undefined] = [0, undefined]
const Y_FACTOR: [number | undefined, number | undefined] = [0, 1.6]
const Y_VW: [number | undefined, number | undefined] = [0, 2]
const V_CURVE = TS.map(vFn)
const W_CURVE = TS.map(wFn)

/**
 * A Gaussian cavity times a step (or probit) factor: the tilted distribution, its moment-matched Gaussian and the site
 * that EP stores, with the v and w functions that give the moments in closed form.
 */
export function TruncatedMoments() {
  const state = useFigureState({
    mu: float(-1, { min: -4, max: 5, step: 0.05, label: 'cavity mean μ' }),
    sd: float(1, { min: 0.3, max: 3, step: 0.05, label: 'cavity sd σ' }),
    eps: slider(-3, 4, 0, { step: 0.05, label: 'threshold ε' }),
    noise: float(0, { min: 0, max: 2, step: 0.05, label: 'noise s (0 = hard step)' }),
  })

  const r = useMemo(() => {
    const variance = state.sd ** 2
    const s2 = state.noise ** 2
    // A hard step is the probit factor with no noise; the two share one formula with σ² + s² in the denominator.
    const t = s2 > 0 ? probitTilted(state.mu, variance, 1, state.eps, s2) : stepTilted(state.mu, variance, state.eps)
    const z = (state.mu - state.eps) / Math.sqrt(variance + s2)
    const factor = XS.map((x) =>
      s2 > 0 ? Math.exp(logNormalCdf((x - state.eps) / Math.sqrt(s2))) : x > state.eps ? 1 : 0,
    )
    const cavity = XS.map((x) => normalPdf(x, state.mu, variance))
    const Z = Math.exp(t.logZ)
    const tilted = XS.map((_, i) => (cavity[i] * factor[i]) / Z)
    const q = XS.map((x) => normalPdf(x, t.mean, t.variance))
    // The site is Z q / cavity: it matches the factor where the cavity has mass and ignores it elsewhere.
    const site = XS.map((x) =>
      Math.min(5, Math.exp(t.logZ + normalLogPdf(x, t.mean, t.variance) - normalLogPdf(x, state.mu, variance))),
    )
    const nat = toNat(t)
    const siteTau = nat.tau - 1 / variance
    const siteNu = nat.nu - state.mu / variance
    return { t, z, factor, cavity, tilted, q, site, siteTau, siteNu, Z }
  }, [state.mu, state.sd, state.noise, state.eps])

  const densities = [
    { name: 'cavity', x: XS, y: r.cavity, muted: true },
    { name: 'tilted (exact)', x: XS, y: r.tilted, emphasis: true },
    { name: 'projection q', x: XS, y: r.q, slot: 0 },
  ] as const
  const factors = [
    { name: 'factor', x: XS, y: r.factor, emphasis: true },
    { name: 'site', x: XS, y: r.site, slot: 4, dashed: true },
    { name: 'cavity (scaled)', x: XS, y: r.cavity.map((c) => c * state.sd * 2.5), muted: true },
  ] as const
  const vw = [
    { name: 'v(t)', x: TS, y: V_CURVE, slot: 0 },
    { name: 'w(t)', x: TS, y: W_CURVE, slot: 1 },
    { name: 'current t', x: [r.z, r.z], y: [vFn(r.z), wFn(r.z)], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'density', range: Y_DENSITY })
  const xAxis2 = useAxis({ label: 'x', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'factor value', range: Y_FACTOR })
  const xAxis3 = useAxis({ label: 't = (μ − ε)/√(σ² + s²)', range: T_RANGE })
  const yAxis3 = useAxis({ range: Y_VW })
  return (
    <Figure
      title="Moments of a Gaussian times a step"
      state={state}
      caption="The cavity N(μ, σ²) is multiplied by the factor 𝟙(x > ε), or by Φ((x − ε)/s) when the noise s is above zero. Left: the exact tilted density and the Gaussian q with its mean and variance. Right: the factor and the Gaussian site Z·q/cavity that EP stores in its place. Drag the threshold ε on either chart. Move μ below ε and the site narrows: it copies the factor only where the cavity has mass."

      readouts={
        <>
          <Readout label="t" value={formatNumber(r.z)} />
          <Readout label="Z = Φ(t)" value={formatNumber(r.Z)} />
          <Readout label="v(t), w(t)" value={`${formatNumber(vFn(r.z))}, ${formatNumber(wFn(r.z))}`} />
          <Readout label="q mean, variance" value={`${formatNumber(r.t.mean)}, ${formatNumber(r.t.variance)}`} />
          <Readout
            label="site mean, variance"
            value={`${formatNumber(r.siteNu / r.siteTau)}, ${formatNumber(1 / r.siteTau)}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...densities[0]} />
          <Curve {...densities[1]} />
          <Curve {...densities[2]} />
          <Handle {...state.handle('eps', { label: 'threshold ε' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Curve {...factors[0]} />
          <Curve {...factors[1]} />
          <Curve {...factors[2]} />
          <Handle {...state.handle('eps', { label: 'threshold ε' })} />
        </Plot>
      </div>
      <Plot x={xAxis3} y={yAxis3} height={220}>
        <Curve {...vw[0]} />
        <Curve {...vw[1]} />
        <Points {...vw[2]} />
      </Plot>
    </Figure>
  )
}
